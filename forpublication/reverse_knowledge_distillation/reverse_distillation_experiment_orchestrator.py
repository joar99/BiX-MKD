"""
Reverse Knowledge-Distillation Sweep
====================================
The forward sweep distilled two frozen teachers into a multimodal student.
Here we flip it: each (text, waveform) teacher is unfrozen and fine-tuned
on the multimodal student's (train ∪ test) cohort, modality-sliced. The
multimodal student becomes the soft-label source.

Three variants per teacher per seed:
  1. baseline                  — pure CE, no KD (the baseline lift)
  2. kd_from_baseline_student  — CE + KD against the no-KD multimodal student
  3. kd_from_kd_student        — CE + KD against the kd_alpha_03_tau_3 student

Per-seed determinism: identical init (loaded from the SAME teacher
checkpoint), identical shuffle (set_seed before DataLoader), identical
dropout RNG (set_seed before each variant). Only the loss differs.

Resumability: each (seed, teacher, variant) cell writes a results.json on
completion. On rerun, cells with existing results.json are skipped and
their cached metrics replayed into the per-seed and cross-seed summaries.
This makes the script behave like run_kd_sweep.py — extending the SEEDS
list or adding configs only re-trains what's missing.

Outputs (Models/BackToTeachers/NEW/checkpoints/):
  soft_labels/
    seed{SEED}/
      {student_config}_logits.pt        # cached student raw logits per hadm_id
  seed{SEED}/
    {teacher_kind}/{variant}/
      best_model.pt                     # best-val-AUC checkpoint
      results.json                      # config + history + metrics + bootstrap CI
      val_predictions.csv               # on teacher's OWN private val
      test_predictions.csv              # on teacher's OWN private test
      epoch_logits/val_epoch_NN.pt      # per-epoch raw val logits
    sweep_summary.json
  sweep_summary_seeds_*.json            # cross-seed aggregate

Expected wall time on MPS:
  text     ~17 min/run × 30 runs ≈ 8.5h
  waveform  ~2 min/run × 30 runs ≈ 1h
  soft labels ~30s × 20 (10 seeds × 2 student variants) ≈ 10 min
  total                                 ≈ 10h, fully resumable
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# ─────────────────────────────────────────────────────────────────────── #
# Path setup — pull modules from sibling project dirs
# ─────────────────────────────────────────────────────────────────────── #
THIS_FILE     = Path(__file__).resolve()
PROJECT_ROOT  = THIS_FILE.parents[1]          # the forpublication/ root (FORPUB)
NEW_DIR       = THIS_FILE.parent
FEATURES_PATH = PROJECT_ROOT / "db" / "mimic_data" / "waveform_features_v2.parquet"
CHECKPOINTS_ROOT = NEW_DIR / "checkpoints"
SOFT_LABELS_ROOT = CHECKPOINTS_ROOT / "soft_labels"

# ── Wiring: module folders this orchestrator imports from ────────────────
for _sub in ("common", "db", "text_teacher", "waveform_teacher", "multimodal_student",
             "preprocessing/waveform_data_preprocessing"):
    _p = str(PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.append(_p)

# Reverse-KD modules (this directory)
from reverse_distillation_experiment_configurations import (                    # noqa: E402
    EXPERIMENT_CONFIGS, CONFIGS_TO_RUN, SEEDS, TEACHERS,
    TEXT_TEACHER_REGIME, WAVEFORM_TEACHER_REGIME,
    ORIGINAL_TEACHER_PATHS, WAVEFORM_SCALER_COHORT,
    STUDENT_CKPT_TEMPLATE, STUDENT_MODEL_CONFIG,
    CONSTANTS,
)
from TextTeacherTrainer import TextTeacherTrainer                                     # noqa: E402
from WaveformTeacherTrainer import WaveformTeacherTrainer                             # noqa: E402

# Existing project modules
from class_weights import compute_class_weights                                  # noqa: E402
from TextTeacherDataset import MortalityTextDataset                                        # noqa: E402
from TextTeacher import TextClassifier                                         # noqa: E402
from WaveformTeacherDataset import MortalityWaveformDataset                        # noqa: E402
from WaveformTeacher import WaveformClassifier                                 # noqa: E402
from waveform_features_v2 import get_feature_columns                            # noqa: E402
from MultimodalStudent import MultimodalStudent                                 # noqa: E402
from MultimodalStudentDataset import MultimodalStudentDataset                   # noqa: E402
from EicuTextDataset import EicuTextDataset                                     # noqa: E402
from EicuWaveformDataset import EicuWaveformDataset                             # noqa: E402
from EicuStudentDataset import EicuStudentDataset                              # noqa: E402

# Central data access (db/fetch_data.py)
from fetch_data import (                                                        # noqa: E402
    Split, DataSource, get_text_teacher_data, get_waveform_teacher_data,
    get_multimodal_student_data,
)


# ─────────────────────────────────────────────────────────────────────── #
# Knobs
# ─────────────────────────────────────────────────────────────────────── #
N_BOOTSTRAP = CONSTANTS["n_bootstrap"]
DEVICE      = CONSTANTS["device"]


# ─────────────────────────────────────────────────────────────────────── #
# Helpers
# ─────────────────────────────────────────────────────────────────────── #
def set_seed(seed: int) -> None:
    """Reset python / numpy / torch RNGs.

    Called at the top of each (seed, variant) cell so that within-seed
    determinism holds across the 3 variants — same DataLoader shuffle,
    same dropout pattern, identical model init (loaded from disk anyway).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ratio_class_weights(target_ratio: float, num_classes: int = 2) -> torch.Tensor:
    """Mirror Models/WaveformModels/MLPV2/train_strong_drop4_v2.py:115."""
    w_neg = num_classes / (1.0 + target_ratio)
    w_pos = target_ratio * w_neg
    return torch.tensor([w_neg, w_pos], dtype=torch.float)


def bootstrap_ci_auc(y_true: np.ndarray, y_prob: np.ndarray,
                     n_bootstrap: int = N_BOOTSTRAP,
                     seed: int = 42) -> Dict:
    """Bootstrap 95% CI for test AUC. Mirrors run_kd_sweep.py:178-196."""
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    aucs = np.array(aucs)
    return {
        "test_auc":    float(roc_auc_score(y_true, y_prob)),
        "ci_lower":    float(np.percentile(aucs, 2.5)),
        "ci_upper":    float(np.percentile(aucs, 97.5)),
        "std":         float(aucs.std()),
        "n_bootstrap": int(len(aucs)),
    }


# ─────────────────────────────────────────────────────────────────────── #
# Soft-label generation (per-seed × per-student-variant cache)
# ─────────────────────────────────────────────────────────────────────── #
def generate_or_load_soft_logits(student_config_name: str,
                                 seed: int,
                                 paired_student_dataset,
                                 source: DataSource,
                                 waveform_input_dim: int,
                                 ) -> Dict[int, torch.Tensor]:
    """
    Run the multimodal student over the paired (train ∪ test) cohort and
    cache its raw logits, keyed by hadm_id, to disk. Cached logits are
    `(num_classes,)` float CPU tensors. The cache is independent of any
    KD-knob choice — temperature is applied *inside the loss*, not here
    (decision (b) in the planning step).

    Cache path:
      Models/BackToTeachers/NEW/checkpoints/soft_labels/seed{SEED}/{config}_logits.pt
    """
    cache_dir = SOFT_LABELS_ROOT / source.value / f"seed{seed}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{student_config_name}_logits.pt"

    if cache_path.exists():
        print(f"  [cache] soft logits for ({student_config_name}, seed={seed}) "
              f"already on disk: {cache_path}")
        return torch.load(cache_path)

    ckpt_rel = STUDENT_CKPT_TEMPLATE.format(source=source.value, seed=seed, config=student_config_name)
    ckpt_path = PROJECT_ROOT / ckpt_rel
    assert ckpt_path.exists(), f"Missing student checkpoint: {ckpt_path}"

    print(f"  [generate] soft logits for ({student_config_name}, seed={seed}) "
          f"<- {ckpt_path}")

    student = MultimodalStudent(**{**STUDENT_MODEL_CONFIG, "waveform_input_dim": waveform_input_dim})
    student.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    student.to(DEVICE)
    student.eval()

    # batch_size=32 matches the student's training-time batch
    loader = DataLoader(paired_student_dataset, batch_size=32,
                        shuffle=False, num_workers=0)

    out: Dict[int, torch.Tensor] = {}
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            input_ids        = batch["input_ids"].to(DEVICE)
            attention_mask   = batch["attention_mask"].to(DEVICE)
            waveform_features = batch["waveform_features"].to(DEVICE)
            logits = student(input_ids, attention_mask, waveform_features).cpu()
            for i, hadm in enumerate(batch["hadm_id"]):
                out[int(hadm)] = logits[i].float()
    elapsed = time.time() - t0
    print(f"  [generate] {len(out)} soft-label entries in {elapsed:.1f}s")

    torch.save(out, cache_path)
    print(f"  [cache] wrote {cache_path}")

    # Free student weights — we only need the logits dict from here on.
    del student
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return out


# ─────────────────────────────────────────────────────────────────────── #
# Shared resources (loaded once)
# ─────────────────────────────────────────────────────────────────────── #
def load_shared_resources(source: DataSource) -> dict:
    print("=" * 70)
    print(f"Loading shared resources (data + splits + scaler + class weights) — source={source.value}")
    print("=" * 70)

    # ── Student train ∪ val pool (the reverse training set) ───────────────
    # test is held out entirely — it's the clean set for the student-vs-teacher
    # comparison, so the teachers must never fine-tune on it. (val is fair game
    # now that forward calibration — which used to touch it — is gone.)
    # get_multimodal_student_data enforces the PAIRED restriction (both
    # modalities) at the source, so train ∪ val IS the paired training pool.
    s_tr = get_multimodal_student_data(Split.TRAIN, source)
    s_va = get_multimodal_student_data(Split.VAL, source)

    # ── Teacher private val/test (each teacher's own held-out split) ──────
    t_val  = get_text_teacher_data(Split.VAL, source)
    t_test = get_text_teacher_data(Split.TEST, source)
    w_val  = get_waveform_teacher_data(Split.VAL, source)
    w_test = get_waveform_teacher_data(Split.TEST, source)

    # ── Waveform feature columns (317 MIMIC / 236 eICU) ──────────────────
    if source is DataSource.MIMIC:
        feature_columns = get_feature_columns()
    else:
        feature_columns = [c for c in w_val.columns if c not in ("patientunitstayid", "mortality")]

    # ── Waveform scaler — from the ORIGINAL waveform-teacher train split ──
    # Keeps the loaded teacher weights coherent with input statistics on epoch 0.
    wave_train_features = get_waveform_teacher_data(Split.TRAIN, source)[feature_columns].values.astype(np.float32)
    waveform_scaler = {
        "mean": np.mean(wave_train_features, axis=0),
        "std":  np.std (wave_train_features, axis=0),
    }
    print(f"Waveform scaler: from {len(wave_train_features)} {WAVEFORM_SCALER_COHORT} train samples")

    # ── Datasets (source-specific assembly) ──────────────────────────────
    ml = CONSTANTS["max_length"]
    print("\nBuilding teacher + student datasets ...")
    if source is DataSource.MIMIC:
        paired_notes = pd.concat([s_tr.notes,      s_va.notes],      ignore_index=True)
        paired_adm   = pd.concat([s_tr.admissions, s_va.admissions], ignore_index=True)
        paired_wave  = pd.concat([s_tr.features,   s_va.features],   ignore_index=True)
        agg = TEXT_TEACHER_REGIME["aggregation"]
        tok = TEXT_TEACHER_REGIME["model_name"]
        text_train_ds = MortalityTextDataset(notes_df=paired_notes, admissions_df=paired_adm,
                                             tokenizer_name=tok, max_length=ml, aggregation=agg)
        text_val_ds   = MortalityTextDataset(notes_df=t_val.notes, admissions_df=t_val.admissions,
                                             tokenizer_name=tok, max_length=ml, aggregation=agg)
        text_test_ds  = MortalityTextDataset(notes_df=t_test.notes, admissions_df=t_test.admissions,
                                             tokenizer_name=tok, max_length=ml, aggregation=agg)
        wave_train_ds = MortalityWaveformDataset(features_df=paired_wave, feature_columns=feature_columns, scaler=waveform_scaler)
        wave_val_ds   = MortalityWaveformDataset(features_df=w_val,       feature_columns=feature_columns, scaler=waveform_scaler)
        wave_test_ds  = MortalityWaveformDataset(features_df=w_test,      feature_columns=feature_columns, scaler=waveform_scaler)
        paired_student_ds = MultimodalStudentDataset(
            notes_df=paired_notes, admissions_df=paired_adm, waveform_features_df=paired_wave,
            feature_columns=feature_columns, aggregation=agg, max_length=ml, scaler=waveform_scaler)
        pool_size = len(paired_adm)
        cohort_meta = {
            "paired_train_pool_size":     pool_size,
            "paired_in_student_train":    len(s_tr.admissions),
            "paired_in_student_val":      len(s_va.admissions),
            "text_teacher_val_size":      len(t_val.admissions),
            "text_teacher_test_size":     len(t_test.admissions),
            "waveform_teacher_val_size":  len(w_val),
            "waveform_teacher_test_size": len(w_test),
        }
    else:  # eICU — the student fetch is already one paired DataFrame per stay
        pool = pd.concat([s_tr, s_va], ignore_index=True)
        text_train_ds = EicuTextDataset(pool,   max_length=ml)
        text_val_ds   = EicuTextDataset(t_val,  max_length=ml)
        text_test_ds  = EicuTextDataset(t_test, max_length=ml)
        wave_train_ds = EicuWaveformDataset(pool,   feature_columns=feature_columns, scaler=waveform_scaler)
        wave_val_ds   = EicuWaveformDataset(w_val,  feature_columns=feature_columns, scaler=waveform_scaler)
        wave_test_ds  = EicuWaveformDataset(w_test, feature_columns=feature_columns, scaler=waveform_scaler)
        paired_student_ds = EicuStudentDataset(pool, feature_columns=feature_columns, max_length=ml, scaler=waveform_scaler)
        pool_size = len(pool)
        cohort_meta = {
            "paired_train_pool_size":     pool_size,
            "paired_in_student_train":    len(s_tr),
            "paired_in_student_val":      len(s_va),
            "text_teacher_val_size":      len(t_val),
            "text_teacher_test_size":     len(t_test),
            "waveform_teacher_val_size":  len(w_val),
            "waveform_teacher_test_size": len(w_test),
        }
    print(f"Reverse training pool (student train ∪ val, paired): {pool_size}")
    print(f"  text train/val/test: {len(text_train_ds)}/{len(text_val_ds)}/{len(text_test_ds)}")
    print(f"  wave train/val/test: {len(wave_train_ds)}/{len(wave_val_ds)}/{len(wave_test_ds)}")

    # ── Class weights ─────────────────────────────────────────────────────
    # Text: inverse-frequency on the new (paired) training pool.
    text_train_labels = torch.tensor(text_train_ds.labels, dtype=torch.long)
    text_class_weights = compute_class_weights(text_train_labels)
    print(f"\nText class weights (inverse-freq on new pool): "
          f"dist={torch.bincount(text_train_labels).tolist()}  "
          f"weights={[f'{w:.4f}' for w in text_class_weights.tolist()]}")

    # Waveform: fixed ratio 2.0, distribution-independent.
    wave_class_weights = ratio_class_weights(2.0)
    print(f"Wave class weights (ratio=2.0): "
          f"weights={[f'{w:.4f}' for w in wave_class_weights.tolist()]}")

    return {
        "text_train_ds":      text_train_ds,
        "text_val_ds":        text_val_ds,
        "text_test_ds":       text_test_ds,
        "wave_train_ds":      wave_train_ds,
        "wave_val_ds":        wave_val_ds,
        "wave_test_ds":       wave_test_ds,
        "paired_student_ds":  paired_student_ds,
        "text_class_weights": text_class_weights,
        "wave_class_weights": wave_class_weights,
        "cohort_meta":        cohort_meta,
        "waveform_scaler":    waveform_scaler,
        "waveform_input_dim": len(feature_columns),
    }


# ─────────────────────────────────────────────────────────────────────── #
# Per-cell run (one (seed, teacher, variant) experiment)
# ─────────────────────────────────────────────────────────────────────── #
def make_teacher_model(teacher_kind: str, seed: int, source: DataSource,
                       waveform_input_dim: int) -> torch.nn.Module:
    """Instantiate teacher fresh and load its SEED-MATCHED frozen state_dict.
    reverse-seed-i re-trains teacher-seed-i."""
    orig_path = PROJECT_ROOT / ORIGINAL_TEACHER_PATHS[teacher_kind].format(source=source.value, seed=seed)
    assert orig_path.exists(), f"Missing original teacher checkpoint: {orig_path}"

    if teacher_kind == "text_teacher":
        cfg = TEXT_TEACHER_REGIME
        model = TextClassifier(
            model_name=cfg["model_name"],
            num_classes=CONSTANTS["num_classes"],
            dropout=cfg["dropout"],
            freeze_layers=cfg["freeze_layers"] if cfg["freeze_layers"] else None,
        )
    elif teacher_kind == "waveform_teacher":
        cfg = WAVEFORM_TEACHER_REGIME
        model = WaveformClassifier(
            input_dim=waveform_input_dim,
            hidden_dims=cfg["hidden_dims"],
            num_classes=CONSTANTS["num_classes"],
            dropout=cfg["dropout"],
            batch_norm=cfg["batch_norm"],
        )
    else:
        raise ValueError(teacher_kind)

    model.load_state_dict(torch.load(orig_path, map_location="cpu"))
    return model


def run_one_cell(seed: int,
                 teacher_kind: str,
                 config_name: str,
                 shared: dict,
                 soft_labels_by_source: Dict[Optional[str], Optional[Dict[int, torch.Tensor]]],
                 cell_dir: Path,
                 source: DataSource) -> dict:
    """
    Run one (seed, teacher_kind, config_name) experiment, with skip-if-exists.
    Returns a compact summary row for the per-seed table.
    """
    cell_dir.mkdir(parents=True, exist_ok=True)
    results_path = cell_dir / "results.json"

    # ── Resume: replay cached metrics if results.json exists ──
    if results_path.exists():
        print(f"[SKIP] {teacher_kind}/{config_name}/seed{seed} — results.json "
              f"exists, reusing cached metrics")
        r = json.loads(results_path.read_text())
        return {
            "seed":         seed,
            "teacher_kind": teacher_kind,
            "config_name":  config_name,
            "test_auc":     r["test_metrics"]["auc"],
            "test_f1":      r["test_metrics"]["f1"],
            "test_acc":     r["test_metrics"]["accuracy"],
            "ci_lower":     r.get("bootstrap_ci", {}).get("ci_lower", float("nan")),
            "ci_upper":     r.get("bootstrap_ci", {}).get("ci_upper", float("nan")),
            "best_epoch":   r["training"]["best_epoch"],
            "total_epochs": r["training"]["total_epochs"],
            "elapsed_sec":  r["training"].get("elapsed_sec", 0.0),
        }

    # ── Set seed BEFORE model load + DataLoader build (within-seed determinism) ──
    set_seed(seed)

    # ── Pick datasets / regime / forward fn / class weights by teacher kind ──
    if teacher_kind == "text_teacher":
        train_ds = shared["text_train_ds"]
        val_ds   = shared["text_val_ds"]
        test_ds  = shared["text_test_ds"]
        regime   = TEXT_TEACHER_REGIME
        trainer_cls = TextTeacherTrainer
        class_weights = shared["text_class_weights"]
    else:  # waveform_teacher
        train_ds = shared["wave_train_ds"]
        val_ds   = shared["wave_val_ds"]
        test_ds  = shared["wave_test_ds"]
        regime   = WAVEFORM_TEACHER_REGIME
        trainer_cls = WaveformTeacherTrainer
        class_weights = shared["wave_class_weights"]

    # ── Build loaders fresh per cell (shuffle pinned by set_seed above) ──
    train_loader = DataLoader(train_ds, batch_size=regime["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=regime["batch_size"], shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=regime["batch_size"], shuffle=False, num_workers=0)

    # ── Model: load original teacher state_dict (deterministic init) ──
    model = make_teacher_model(teacher_kind, seed, source, shared["waveform_input_dim"])

    # ── KD knobs + optional soft labels (looked up from config) ──
    cfg         = EXPERIMENT_CONFIGS[config_name]
    src_name    = cfg["soft_label_source"]      # None for baseline
    soft_logits = soft_labels_by_source[src_name]

    # ── Trainer ──
    trainer_kwargs = dict(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=regime["learning_rate"],
        weight_decay=regime["weight_decay"],
        num_epochs=regime["num_epochs"],
        patience=regime["patience"],
        class_weights=class_weights,
        label_smoothing=regime["label_smoothing"],
        soft_logits=soft_logits,
        alpha=cfg["alpha"],
        temperature=cfg["temperature"],
        device=DEVICE,
        checkpoint_dir=str(cell_dir),
        epoch_logits_dir=str(cell_dir / "epoch_logits"),
    )
    if regime["scheduler_kind"] == "linear_warmup":
        trainer_kwargs["warmup_ratio"] = regime["warmup_ratio"]

    trainer = trainer_cls(**trainer_kwargs)

    # ── Train ──
    t0 = time.time()
    history = trainer.train()
    elapsed = time.time() - t0

    # ── Evaluate using best-val-AUC checkpoint ──
    trainer.load_checkpoint("best_model.pt")
    val_metrics  = trainer.evaluate(val_loader)
    test_metrics = trainer.evaluate(test_loader)

    # Train predictions: build a non-shuffled loader so the CSV is in
    # dataset order (the training loader was shuffled). This lets us
    # inspect the teacher's behaviour on its OWN training data after
    # fine-tuning — e.g. how confident it is on the soft-labelled
    # admissions, how it disagrees with the multimodal student, etc.
    # Also useful for post-hoc splitting via cohort_meta (which paired
    # admissions came from student_train vs student_test).
    train_eval_loader = DataLoader(train_ds, batch_size=regime["batch_size"],
                                   shuffle=False, num_workers=0)
    train_preds = trainer.get_predictions(train_eval_loader)
    val_preds   = trainer.get_predictions(val_loader)
    test_preds  = trainer.get_predictions(test_loader)
    pd.DataFrame(train_preds).to_csv(cell_dir / "train_predictions.csv", index=False)
    pd.DataFrame(val_preds  ).to_csv(cell_dir / "val_predictions.csv",   index=False)
    pd.DataFrame(test_preds ).to_csv(cell_dir / "test_predictions.csv",  index=False)

    # ── Bootstrap CI on test ──
    y_true = np.array(test_preds["true_label"])
    y_prob = np.array(test_preds["pred_prob"])
    bootstrap = bootstrap_ci_auc(y_true, y_prob)

    # ── Persist results.json ──
    best_epoch = int(np.argmax(history["val_auc"]) + 1)
    results = {
        "experiment":     f"reverse_kd/{teacher_kind}/{config_name}",
        "source":         source.value,
        "seed":           seed,
        "teacher_kind":   teacher_kind,
        "config_name":    config_name,
        "kd": {
            "alpha":                  cfg["alpha"],
            "temperature":            cfg["temperature"],
            "soft_label_source":      src_name,               # None for baseline
            "soft_label_calibration": "none",                 # explicit per spec
        },
        "regime":         regime,
        "constants":      CONSTANTS,
        "original_teacher_path": ORIGINAL_TEACHER_PATHS[teacher_kind].format(source=source.value, seed=seed),
        "student_ckpt_path":     (
            STUDENT_CKPT_TEMPLATE.format(source=source.value, seed=seed, config=src_name)
            if src_name is not None else None
        ),
        "cohort_meta":    shared["cohort_meta"],
        "training": {
            "history":      history,
            "best_epoch":   best_epoch,
            "total_epochs": len(history["train_loss"]),
            "elapsed_sec":  float(elapsed),
        },
        "val_metrics":    val_metrics,
        "test_metrics":   test_metrics,
        "bootstrap_ci":   bootstrap,
        "timestamp":      datetime.now().isoformat(),
    }
    results_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"  Saved {results_path}")

    return {
        "seed":         seed,
        "teacher_kind": teacher_kind,
        "config_name":  config_name,
        "test_auc":     test_metrics["auc"],
        "test_f1":      test_metrics["f1"],
        "test_acc":     test_metrics["accuracy"],
        "ci_lower":     bootstrap["ci_lower"],
        "ci_upper":     bootstrap["ci_upper"],
        "best_epoch":   best_epoch,
        "total_epochs": len(history["train_loss"]),
        "elapsed_sec":  float(elapsed),
    }


# ─────────────────────────────────────────────────────────────────────── #
# Main loop
# ─────────────────────────────────────────────────────────────────────── #
def main(source: DataSource) -> None:
    CHECKPOINTS_ROOT.mkdir(parents=True, exist_ok=True)
    SOFT_LABELS_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Project root:      {PROJECT_ROOT}")
    print(f"Checkpoints root:  {CHECKPOINTS_ROOT}")
    print(f"Seeds to run:      {SEEDS}")
    print(f"Teachers per seed: {TEACHERS}")
    print(f"Configs per cell:  {CONFIGS_TO_RUN}")
    print(f"Total cells:       {len(SEEDS) * len(TEACHERS) * len(CONFIGS_TO_RUN)}")
    print()

    # Load notes / waveform features / cohorts / scaler / class weights once.
    shared = load_shared_resources(source)

    # Soft-label sources we need (deduplicate Nones — baseline doesn't need them).
    needed_student_sources = sorted({
        EXPERIMENT_CONFIGS[c]["soft_label_source"]
        for c in CONFIGS_TO_RUN
        if EXPERIMENT_CONFIGS[c]["soft_label_source"] is not None
    })
    print(f"\nStudent soft-label sources to materialise per seed: "
          f"{needed_student_sources}")

    all_rows = []
    cell_counter = 0
    total_cells = len(SEEDS) * len(TEACHERS) * len(CONFIGS_TO_RUN)
    total_t0 = time.time()

    for seed in SEEDS:
        seed_root = CHECKPOINTS_ROOT / source.value / f"seed{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)

        print("\n" + "▓" * 70)
        print(f"▓ SEED {seed}   (output: {seed_root})")
        print("▓" * 70)

        # ── Soft labels per student variant (cached on disk) ──
        print(f"\n[seed {seed}] Materialising soft labels ...")
        soft_labels_by_source: Dict[Optional[str], Optional[Dict[int, torch.Tensor]]] = {None: None}
        for src in needed_student_sources:
            soft_labels_by_source[src] = generate_or_load_soft_logits(
                src, seed, shared["paired_student_ds"], source, shared["waveform_input_dim"],
            )

        seed_rows = []
        for teacher_kind in TEACHERS:
            for config_name in CONFIGS_TO_RUN:
                cell_counter += 1
                cell_dir = seed_root / teacher_kind / config_name
                print(f"\n[{cell_counter}/{total_cells}] "
                      f"seed={seed}  teacher={teacher_kind}  config={config_name}")

                row = run_one_cell(
                    seed=seed,
                    teacher_kind=teacher_kind,
                    config_name=config_name,
                    shared=shared,
                    soft_labels_by_source=soft_labels_by_source,
                    cell_dir=cell_dir,
                    source=source,
                )
                seed_rows.append(row)
                all_rows.append(row)

        # ── Per-seed sweep summary ──
        seed_summary = {
            "seed":           seed,
            "checkpoints":    str(seed_root),
            "timestamp":      datetime.now().isoformat(),
            "teachers":       TEACHERS,
            "configs_run":    CONFIGS_TO_RUN,
            "configs":        {c: EXPERIMENT_CONFIGS[c] for c in CONFIGS_TO_RUN},
            "rows":           seed_rows,
        }
        (seed_root / "sweep_summary.json").write_text(
            json.dumps(seed_summary, indent=2, default=float)
        )

        # Per-seed table
        print("\n" + "-" * 110)
        print(f"SEED {seed} SUMMARY  (sorted by test AUC within teacher)")
        print("-" * 110)
        for tk in TEACHERS:
            print(f"  {tk}")
            tk_rows = sorted([r for r in seed_rows if r["teacher_kind"] == tk],
                             key=lambda x: -x["test_auc"])
            for r in tk_rows:
                ci = f"[{r['ci_lower']:.3f}-{r['ci_upper']:.3f}]"
                print(f"    {r['config_name']:<46} AUC {r['test_auc']:.4f}  {ci:<20}  "
                      f"F1 {r['test_f1']:.4f}  best {r['best_epoch']}/{r['total_epochs']}")

    total_elapsed = time.time() - total_t0

    # ── Cross-seed aggregate ──
    print("\n" + "=" * 110)
    print(f"CROSS-SEED SUMMARY  (across seeds {SEEDS})")
    print("=" * 110)
    header = (f"{'rank':<4} {'teacher':<18} {'config_name':<46} "
              f"{'test_auc (mean±std)':<25} {'test_f1 (mean±std)':<25} {'n_seeds':<10}")
    print(header)
    print("-" * len(header))

    agg_rows = []
    per_cell_auc: Dict[tuple, list] = {}
    per_cell_f1:  Dict[tuple, list] = {}
    for r in all_rows:
        key = (r["teacher_kind"], r["config_name"])
        per_cell_auc.setdefault(key, []).append(r["test_auc"])
        per_cell_f1 .setdefault(key, []).append(r["test_f1"])

    for tk in TEACHERS:
        for c in CONFIGS_TO_RUN:
            key = (tk, c)
            aucs = np.array(per_cell_auc.get(key, []))
            f1s  = np.array(per_cell_f1 .get(key, []))
            agg_rows.append({
                "teacher_kind": tk,
                "config_name":  c,
                "auc_mean":     float(aucs.mean()) if len(aucs) else float("nan"),
                "auc_std":      float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
                "f1_mean":      float(f1s.mean())  if len(f1s)  else float("nan"),
                "f1_std":       float(f1s.std (ddof=1)) if len(f1s)  > 1 else 0.0,
                "n_seeds":      int(len(aucs)),
            })

    # Print rank within teacher (so each teacher's best config is obvious)
    def rank_within(tk: str):
        sub = [r for r in agg_rows if r["teacher_kind"] == tk]
        sub.sort(key=lambda r: -r["auc_mean"])
        for i, r in enumerate(sub, 1):
            auc = f"{r['auc_mean']:.4f} ± {r['auc_std']:.4f}"
            f1  = f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}"
            print(f"{i:<4} {r['teacher_kind']:<18} {r['config_name']:<46} "
                  f"{auc:<25} {f1:<25} {r['n_seeds']:<10}")
    for tk in TEACHERS:
        rank_within(tk)

    # ── Top-level cross-seed JSON ──
    cross_path = CHECKPOINTS_ROOT / source.value / f"sweep_summary_seeds_{'_'.join(str(s) for s in SEEDS)}.json"
    cross_path.write_text(json.dumps({
        "seeds":             SEEDS,
        "teachers":          TEACHERS,
        "configs_run":       CONFIGS_TO_RUN,
        "configs":           {c: EXPERIMENT_CONFIGS[c] for c in CONFIGS_TO_RUN},
        "all_rows":          all_rows,
        "aggregated":        agg_rows,
        "total_elapsed_sec": float(total_elapsed),
        "timestamp":         datetime.now().isoformat(),
    }, indent=2, default=float))
    print(f"\nCross-seed summary: {cross_path}")
    print(f"Total time: {total_elapsed/60:.1f} min "
          f"({total_elapsed/max(cell_counter, 1)/60:.1f} min/cell avg)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Reverse-KD: re-train seed-matched teachers from student soft labels, on MIMIC or eICU.")
    ap.add_argument("--source", required=True, choices=[s.value for s in DataSource],
                    help="which dataset to run on (mimic | eicu)")
    args = ap.parse_args()
    main(DataSource(args.source))
