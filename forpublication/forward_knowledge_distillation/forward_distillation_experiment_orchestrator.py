"""
Knowledge-distillation α × τ sweep for the multimodal student.
==============================================================
Port of the original notebook sweep (OldStuff/StudentModel/test_student.ipynb)
to a single batch script.

What changed from the v3 sweep:
  - Waveform teacher: waveform_low_dropout_v2 → waveform_strong_drop4_v2
    (the bs=64, seed=99999, dropout=0.4 teacher from the health-sweep work).
  - Student's waveform-branch dropout: 0.1 → 0.4 (now matches the teacher).
  - TEACHER_CONFIGS["waveform_teacher"]["dropout"]: 0.1 → 0.4 (hygiene —
    was incorrect before but benign since teachers are in .eval()).
  - **Per-branch AdamW parameter groups**: student branches inherit their
    corresponding teacher's training hyperparameters where possible.
      - text_encoder + text_projection: lr=2e-5, wd=0.01 (BERT convention,
        matches text teacher)
      - waveform_encoder: lr=5e-4, wd=1e-3 (matches waveform_strong_drop4_v2
        teacher)
      - fusion: lr=5e-4, wd=1e-3 (no teacher analog; uses waveform-MLP
        regime as a similar-scale randomly-initialized small MLP)
    This gives a methodologically cleaner defense than a unified BERT-LR
    across a model where 99% of params are BERT but the non-BERT components
    were tuned at completely different regimes.
  - **No teacher calibration.** Under the federated premise the reverse
    direction cannot Guo-calibrate (it would need the teachers' validation
    data on the student side), so the forward direction doesn't either —
    keeping both directions methodologically symmetric. Teachers use raw
    logits with only the Hinton τ softening.
  - Text teacher and text branch unchanged.

Sweep (7 configs, single seed each — matches original v3 methodology):
  - multimodal_baseline                    alpha=0.0, tau=1.0  (no KD)
  - kd_alpha_03_tau_3                      alpha=0.3, tau=3.0
  - kd_alpha_05_tau_3                      alpha=0.5, tau=3.0
  - kd_alpha_07_tau_3                      alpha=0.7, tau=3.0
  - kd_alpha_05_tau_1                      alpha=0.5, tau=1.0
  - kd_alpha_05_tau_5                      alpha=0.5, tau=5.0
  - kd_alpha_05_tau_10                     alpha=0.5, tau=10.0

What is loaded once (not per config):
  - Notes, admissions, waveform features
  - Cohort splits (student cohort)
  - Waveform scaler (from waveform teacher's training set)
  - Text teacher (frozen)   — BERT is ~400MB; no point rebuilding 7 times
  - Waveform teacher (frozen)
  - Student datasets (train/val/test)

What is fresh per config:
  - Student model (random init)
  - DistillationTrainer (new optimizer / scheduler state)

Outputs (Models/StudentModel/checkpoints/v4/{config_name}/):
  - best_model.pt        state_dict at best-val-AUC epoch
  - results.json         config + history + test metrics + bootstrap CI
  - val_predictions.csv  hadm_id, true_label, pred_prob, pred_label
  - test_predictions.csv hadm_id, true_label, pred_prob, pred_label

A combined `sweep_summary.json` and a printed comparison table land at the
end for quick inspection.

Expected wall time: BERT dominates — ~2 min/epoch × 8 epochs × 7 configs
≈ 2 hours on MPS.
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# --------------------------------------------------------------------------- #
# Path setup
# --------------------------------------------------------------------------- #
THIS_FILE    = Path(__file__).resolve()
FORPUB       = THIS_FILE.parents[1]          # the forpublication/ root
FEATURES_PATH = FORPUB / "db" / "mimic_data" / "waveform_features_v2.parquet"
CHECKPOINTS_ROOT = THIS_FILE.parent / "checkpoints"   # per-seed subdirs created at runtime

# ── Wiring: module folders this orchestrator imports from ────────────────
for _sub in ("common", "db", "text_teacher", "waveform_teacher", "multimodal_student",
             "preprocessing/waveform_data_preprocessing"):
    _p = str(FORPUB / _sub)
    if _p not in sys.path:
        sys.path.append(_p)

# Student modules
from forward_distillation_experiment_configurations import (                                    # noqa: E402
    EXPERIMENT_CONFIGS, CONFIGS_TO_RUN, SEEDS, N_BOOTSTRAP,
    CONSTANTS, TEACHER_PATHS, TEACHER_CONFIGS,
)
from MultimodalStudent import MultimodalStudent                                                 # noqa: E402
from MultimodalStudentDataset import MultimodalStudentDataset                                    # noqa: E402
from EicuStudentDataset import EicuStudentDataset                                                # noqa: E402
from StudentTrainer import DistillationTrainer                                                   # noqa: E402

# Teacher data helpers
from class_weights import compute_class_weights                                                  # noqa: E402

# Teacher model classes
from TextTeacher import TextClassifier                                                          # noqa: E402
from WaveformTeacher import WaveformClassifier                                                  # noqa: E402

# Waveform feature helpers
from waveform_features_v2 import get_feature_columns                                             # noqa: E402

# Central data access (db/fetch_data.py)
from fetch_data import (                                                                         # noqa: E402
    Split, DataSource, get_multimodal_student_data, get_waveform_teacher_data,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bootstrap_ci_auc(y_true: np.ndarray, y_prob: np.ndarray,
                     n_bootstrap: int = N_BOOTSTRAP,
                     seed: int = 42) -> Dict:
    """Bootstrap 95% CI for test AUC. Mirrors the notebook's logic."""
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


# --------------------------------------------------------------------------- #
# Shared data / teacher loading (runs once)
# --------------------------------------------------------------------------- #
def load_shared_resources(source: DataSource):
    print("=" * 70)
    print(f"Loading shared resources (data + teachers) — source={source.value}")
    print("=" * 70)

    # --- Student data per split (cohort/split + note leakage-filter in fetch_data) ─
    base = EXPERIMENT_CONFIGS[CONFIGS_TO_RUN[0]]

    s_train = get_multimodal_student_data(Split.TRAIN, source)
    s_val   = get_multimodal_student_data(Split.VAL, source)
    s_test  = get_multimodal_student_data(Split.TEST, source)

    # Waveform feature columns: MIMIC 317-dim recipe, eICU 236-dim vitals recipe.
    if source is DataSource.MIMIC:
        feature_columns = get_feature_columns()
    else:
        _wf = get_waveform_teacher_data(Split.TRAIN, source)
        feature_columns = [c for c in _wf.columns if c not in ("patientunitstayid", "mortality")]

    # --- Scaler from the waveform-teacher training set ------------------
    # Guarantees the student sees waveform features standardized with the same
    # stats the frozen waveform teacher saw at its own training time.
    wave_train_features = get_waveform_teacher_data(Split.TRAIN, source)[feature_columns].values.astype(np.float32)
    waveform_scaler = {
        "mean": np.mean(wave_train_features, axis=0),
        "std":  np.std (wave_train_features, axis=0),
    }
    print(f"Scaler computed from {len(wave_train_features)} waveform-teacher training samples")

    # --- Student datasets ------------------------------------------------
    print("\nBuilding paired (text, waveform) datasets ...")
    if source is DataSource.MIMIC:
        def _student_ds(s):
            return MultimodalStudentDataset(
                notes_df=s.notes, admissions_df=s.admissions,
                waveform_features_df=s.features, feature_columns=feature_columns,
                aggregation=base["aggregation"], max_length=CONSTANTS["max_length"],
                scaler=waveform_scaler,
            )
    else:
        def _student_ds(s):
            return EicuStudentDataset(
                s, feature_columns=feature_columns,
                max_length=CONSTANTS["max_length"], scaler=waveform_scaler,
            )
    train_dataset = _student_ds(s_train)
    val_dataset   = _student_ds(s_val)
    test_dataset  = _student_ds(s_test)

    # (Teachers are loaded per-seed in load_teachers() — seed-matched.)

    # --- Class weights (fixed across configs, from train dataset) -------
    train_labels = torch.tensor(train_dataset.labels)
    class_weights = compute_class_weights(train_labels)
    print(f"Train class dist: {torch.bincount(train_labels).tolist()}  "
          f"weights: {[f'{w:.3f}' for w in class_weights.tolist()]}")

    # NOTE: no teacher calibration. Under the federated premise the reverse
    # direction cannot Guo-calibrate (it would need the teachers' val data on
    # the student side), so the forward direction doesn't either — keeping the
    # two directions methodologically symmetric. Teachers use raw logits with
    # only the Hinton KD temperature (tau).

    return {
        "train_dataset":      train_dataset,
        "val_dataset":        val_dataset,
        "test_dataset":       test_dataset,
        "class_weights":      class_weights,
        "waveform_input_dim": len(feature_columns),
    }


def load_teachers(seed: int, source: DataSource, waveform_input_dim: int) -> dict:
    """Load the seed-matched frozen teachers for one seed.
    student-seed-i distills from text-teacher-seed-i + waveform-teacher-seed-i."""
    device = CONSTANTS["device"]

    text_cfg = TEACHER_CONFIGS["text_teacher"]
    text_teacher = TextClassifier(
        model_name=text_cfg["model_name"],
        num_classes=text_cfg["num_classes"],
        dropout=text_cfg["dropout"],
    )
    text_path = TEACHER_PATHS["text_teacher"].format(source=source.value, seed=seed)
    text_teacher.load_state_dict(torch.load(text_path, map_location="cpu"))
    text_teacher.eval()
    for p in text_teacher.parameters(): p.requires_grad = False
    text_teacher.to(device)

    wave_cfg = TEACHER_CONFIGS["waveform_teacher"]
    waveform_teacher = WaveformClassifier(
        input_dim=waveform_input_dim,
        hidden_dims=wave_cfg["hidden_dims"],
        num_classes=wave_cfg["num_classes"],
        dropout=wave_cfg["dropout"],
        batch_norm=wave_cfg["batch_norm"],
    )
    wave_path = TEACHER_PATHS["waveform_teacher"].format(source=source.value, seed=seed)
    waveform_teacher.load_state_dict(torch.load(wave_path, map_location="cpu"))
    waveform_teacher.eval()
    for p in waveform_teacher.parameters(): p.requires_grad = False
    waveform_teacher.to(device)

    print(f"  seed-{seed} teachers:  text <- {text_path}")
    print(f"                         wave <- {wave_path}")
    return {"text_teacher": text_teacher, "waveform_teacher": waveform_teacher}


# --------------------------------------------------------------------------- #
# Single-config run
# --------------------------------------------------------------------------- #
def run_one_config(config_name: str, config: dict, shared: dict,
                   seed: int, seed_root: Path) -> Dict:
    print("\n" + "#" * 70)
    print(f"# CONFIG: {config_name}   seed={seed}   "
          f"(alpha={config['alpha']}, tau={config['temperature']})")
    print("#" * 70)

    ckpt_dir = seed_root / config_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Skip if this seed×config already has saved results -----------
    # Lets the sweep script act as a "resume": re-running with an extended
    # SEEDS list (or adding configs) won't redo work already on disk.
    results_path = ckpt_dir / "results.json"
    if results_path.exists():
        print(f"[SKIP] seed={seed} config={config_name} — results.json exists, "
              f"reusing cached metrics")
        r = json.loads(results_path.read_text())
        return {
            "experiment":   config_name,
            "alpha":        config["alpha"],
            "temperature":  config["temperature"],
            "test_auc":     r["test_metrics"]["auc"],
            "test_f1":      r["test_metrics"]["f1"],
            "test_acc":     r["test_metrics"].get("accuracy",
                             r["test_metrics"].get("acc", float("nan"))),
            "ci_lower":     r.get("bootstrap_ci", {}).get("ci_lower", float("nan")),
            "ci_upper":     r.get("bootstrap_ci", {}).get("ci_upper", float("nan")),
            "best_epoch":   r["training"]["best_epoch"],
            "total_epochs": r["training"].get("total_epochs",
                             len(r["training"]["history"]["train_loss"])),
            "elapsed_sec":  r["training"].get("elapsed_sec", 0.0),
        }

    set_seed(seed)

    # Loaders (fresh each config — shuffle order depends on seed; rebuilt here
    # so every config sees the same shuffle pattern given the same SEED)
    train_loader = torch.utils.data.DataLoader(
        shared["train_dataset"], batch_size=config["batch_size"], shuffle=True,  num_workers=0,
    )
    val_loader   = torch.utils.data.DataLoader(
        shared["val_dataset"],   batch_size=config["batch_size"], shuffle=False, num_workers=0,
    )
    test_loader  = torch.utils.data.DataLoader(
        shared["test_dataset"],  batch_size=config["batch_size"], shuffle=False, num_workers=0,
    )

    # Fresh student per config (random init)
    student = MultimodalStudent(
        text_model_name      =config["text_model_name"],
        text_proj_dim        =config["text_proj_dim"],
        text_dropout         =config["text_dropout"],
        freeze_bert          =config["freeze_bert"],
        freeze_bert_layers   =config["freeze_bert_layers"],
        waveform_input_dim   =shared["waveform_input_dim"],
        waveform_hidden_dims =config["waveform_hidden_dims"],
        wave_proj_dim        =config["wave_proj_dim"],
        waveform_dropout     =config["waveform_dropout"],
        waveform_batch_norm  =config["waveform_batch_norm"],
        fusion_hidden_dim    =config["fusion_hidden_dim"],
        fusion_dropout       =config["fusion_dropout"],
        num_classes          =CONSTANTS["num_classes"],
    )

    # Only attach teachers when alpha > 0
    is_kd = config["alpha"] > 0
    trainer = DistillationTrainer(
        student=student,
        text_teacher     =shared["text_teacher"]     if is_kd else None,
        waveform_teacher =shared["waveform_teacher"] if is_kd else None,
        train_loader=train_loader, val_loader=val_loader,
        alpha            =config["alpha"],
        temperature      =config["temperature"],
        # Text-branch hyperparameters (BERT encoder + text_projection)
        learning_rate    =config["learning_rate"],
        weight_decay     =config["weight_decay"],
        # Per-branch overrides: each branch inherits its teacher's regime
        waveform_learning_rate=config.get("waveform_learning_rate"),
        waveform_weight_decay =config.get("waveform_weight_decay"),
        fusion_learning_rate  =config.get("fusion_learning_rate"),
        fusion_weight_decay   =config.get("fusion_weight_decay"),
        num_epochs       =config["num_epochs"],
        warmup_ratio     =config["warmup_ratio"],
        patience         =config["patience"],
        checkpoint_dir   =str(ckpt_dir),
        class_weights    =shared["class_weights"],
        device           =CONSTANTS["device"],
        # Calibration dropped (federated symmetry) — raw teacher logits, KD tau only.
        text_teacher_temperature    =None,
        waveform_teacher_temperature=None,
        text_kd_weight    =config.get("text_kd_weight",     0.5),
        waveform_kd_weight=config.get("waveform_kd_weight", 0.5),
    )

    t0 = time.time()
    history = trainer.train()
    elapsed = time.time() - t0

    # --- Evaluate on test using best-val-AUC checkpoint -----------------
    trainer.load_checkpoint("best_model.pt")
    test_metrics = trainer.evaluate(test_loader)

    # Predictions (hadm_id, true_label, pred_prob, pred_label)
    val_preds  = trainer.get_test_predictions(val_loader)
    test_preds = trainer.get_test_predictions(test_loader)
    pd.DataFrame(val_preds ).to_csv(ckpt_dir / "val_predictions.csv",  index=False)
    pd.DataFrame(test_preds).to_csv(ckpt_dir / "test_predictions.csv", index=False)

    # Bootstrap CI for test AUC
    y_true = np.array(test_preds["true_label"])
    y_prob = np.array(test_preds["pred_prob"])
    bootstrap = bootstrap_ci_auc(y_true, y_prob)

    # --- results.json (shape mirrors the old notebook output) ----------
    results = {
        "experiment":    config_name,
        "config":        config,
        "constants":     CONSTANTS,
        "teacher_paths": TEACHER_PATHS if is_kd else None,
        "seed":          seed,
        "dataset": {
            "cohort":              "student",
            "train_samples":       len(shared["train_dataset"]),
            "val_samples":         len(shared["val_dataset"]),
            "test_samples":        len(shared["test_dataset"]),
            "class_weights":       shared["class_weights"].tolist(),
        },
        "model":         student.count_parameters(),
        "training": {
            "history":     history,
            "best_epoch":  int(np.argmax(history["val_auc"]) + 1),
            "total_epochs":len(history["train_loss"]),
            "elapsed_sec": float(elapsed),
        },
        "test_metrics":  test_metrics,
        "bootstrap_ci":  bootstrap,
        "timestamp":     datetime.now().isoformat(),
    }
    (ckpt_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved: {ckpt_dir}/results.json")

    # Compact return for the summary table
    return {
        "experiment":  config_name,
        "alpha":       config["alpha"],
        "temperature": config["temperature"],
        "test_auc":    test_metrics["auc"],
        "test_f1":     test_metrics["f1"],
        "test_acc":    test_metrics["accuracy"],
        "ci_lower":    bootstrap["ci_lower"],
        "ci_upper":    bootstrap["ci_upper"],
        "best_epoch":  int(np.argmax(history["val_auc"]) + 1),
        "total_epochs":len(history["train_loss"]),
        "elapsed_sec": float(elapsed),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(source: DataSource) -> None:
    CHECKPOINTS_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoints root:  {CHECKPOINTS_ROOT}")
    print(f"Seeds to run:      {SEEDS}")
    print(f"Configs per seed:  {CONFIGS_TO_RUN}")
    print(f"Total runs:        {len(SEEDS) * len(CONFIGS_TO_RUN)}")
    print()

    # Load data + teachers once (splits are deterministic; teachers are frozen).
    shared = load_shared_resources(source)

    all_rows = []     # flat across seeds
    total_t0 = time.time()
    run_counter = 0
    total_runs = len(SEEDS) * len(CONFIGS_TO_RUN)

    for seed in SEEDS:
        seed_root = CHECKPOINTS_ROOT / source.value / f"seed{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        print("\n" + "▓" * 70)
        print(f"▓ SEED {seed}   (output: {seed_root})")
        print("▓" * 70)

        # Seed-matched teachers: this seed's students distill from teacher-seed-i.
        teachers = load_teachers(seed, source, shared["waveform_input_dim"])
        seed_shared = {**shared, **teachers}

        seed_rows = []
        for name in CONFIGS_TO_RUN:
            run_counter += 1
            cfg = EXPERIMENT_CONFIGS[name]
            print(f"\n[{run_counter}/{total_runs}] seed={seed}  config={name}")
            row = run_one_config(name, cfg, seed_shared, seed, seed_root)
            seed_rows.append(row)
            all_rows.append(row)

        # Free this seed's teachers before the next seed loads its own.
        del teachers, seed_shared
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Per-seed summary
        seed_summary = {
            "seed":          seed,
            "checkpoints":   str(seed_root),
            "timestamp":     datetime.now().isoformat(),
            "teacher_paths": TEACHER_PATHS,
            "rows":          seed_rows,
        }
        (seed_root / "sweep_summary.json").write_text(
            json.dumps(seed_summary, indent=2, default=float)
        )

        # Per-seed table
        print("\n" + "-" * 110)
        print(f"SEED {seed} SUMMARY  (sorted by test AUC)")
        print("-" * 110)
        for i, r in enumerate(sorted(seed_rows, key=lambda x: -x["test_auc"]), 1):
            ci = f"[{r['ci_lower']:.3f}-{r['ci_upper']:.3f}]"
            print(f"{i:<3} {r['experiment']:<28} α={r['alpha']:<4} τ={r['temperature']:<5} "
                  f"AUC {r['test_auc']:.4f}  {ci:<20} F1 {r['test_f1']:.4f}  "
                  f"best {r['best_epoch']}/{r['total_epochs']}")

    total_elapsed = time.time() - total_t0

    # --- Cross-seed summary: aggregate each config across seeds ---------
    print("\n" + "=" * 110)
    print(f"CROSS-SEED SUMMARY  (across seeds {SEEDS}, mean ± std per config)")
    print("=" * 110)
    header = (f"{'rank':<4} {'experiment':<28} {'α':<6} {'τ':<6} "
              f"{'test_auc (mean±std)':<25} {'test_f1 (mean±std)':<25} {'n_seeds':<10}")
    print(header); print("-" * len(header))

    # Group by experiment name
    per_exp_auc = {}
    per_exp_f1  = {}
    for r in all_rows:
        per_exp_auc.setdefault(r["experiment"], []).append(r["test_auc"])
        per_exp_f1 .setdefault(r["experiment"], []).append(r["test_f1"])

    agg_rows = []
    for name in CONFIGS_TO_RUN:
        aucs = np.array(per_exp_auc[name])
        f1s  = np.array(per_exp_f1 [name])
        cfg  = EXPERIMENT_CONFIGS[name]
        agg_rows.append({
            "experiment":  name,
            "alpha":       cfg["alpha"],
            "temperature": cfg["temperature"],
            "auc_mean":    float(aucs.mean()),
            "auc_std":     float(aucs.std(ddof=1) if len(aucs) > 1 else 0.0),
            "f1_mean":     float(f1s.mean()),
            "f1_std":      float(f1s.std (ddof=1) if len(f1s) > 1 else 0.0),
            "n_seeds":     int(len(aucs)),
        })

    agg_rows.sort(key=lambda r: -r["auc_mean"])
    for i, r in enumerate(agg_rows, 1):
        auc = f"{r['auc_mean']:.4f} ± {r['auc_std']:.4f}"
        f1  = f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}"
        print(f"{i:<4} {r['experiment']:<28} {r['alpha']:<6} {r['temperature']:<6} "
              f"{auc:<25} {f1:<25} {r['n_seeds']:<10}")

    # Top-level JSON with all seeds' rows + aggregated view
    cross_summary_path = CHECKPOINTS_ROOT / source.value / f"sweep_summary_seeds_{'_'.join(str(s) for s in SEEDS)}.json"
    cross_summary_path.write_text(json.dumps({
        "seeds":              SEEDS,
        "configs":            CONFIGS_TO_RUN,
        "teacher_paths":      TEACHER_PATHS,
        "all_rows":           all_rows,
        "aggregated":         agg_rows,
        "total_elapsed_sec":  float(total_elapsed),
        "timestamp":          datetime.now().isoformat(),
    }, indent=2, default=float))
    print(f"\nCross-seed summary: {cross_summary_path}")
    print(f"Total time: {total_elapsed/60:.1f} min  "
          f"({total_elapsed/total_runs/60:.1f} min/run avg)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Forward-KD: distill seed-matched teachers into the student, on MIMIC or eICU.")
    ap.add_argument("--source", required=True, choices=[s.value for s in DataSource],
                    help="which dataset to run on (mimic | eicu)")
    args = ap.parse_args()
    main(DataSource(args.source))
