"""
Create the baseline text teacher (BIX1).
========================================
Trains BioClinicalBERT with pure supervised cross-entropy (no KD) on the
text_teacher cohort. The resulting best-val-AUC checkpoint is the frozen
text teacher that forward-KD later distills from.

Runs on either dataset — choose with `--source mimic|eicu` (REQUIRED, no default):
    python text_teacher/create_baseline_model.py --source mimic
    python text_teacher/create_baseline_model.py --source eicu
The model, hyperparameters and KD procedure are identical across sources; only
the corpus differs (MIMIC clinical notes vs eICU manufactured notes). The one
source-specific seam is `build_text_datasets()`.

Config = the thesis teacher: batch 32, aggregation "last" (MIMIC), 6h buffer.

Outputs (checkpoints/{source}/{EXPERIMENT_TO_RUN}/seed{seed}/):
  - best_model.pt         state_dict at best-val-AUC epoch
  - test_predictions.csv  hadm_id, true_label, pred_prob, pred_label
  - results.json          config + history + test metrics
"""

import argparse
import sys
from pathlib import Path

# ── Wiring: put the forpublication module folders this script needs on sys.path ──
FORPUB = Path(__file__).resolve().parents[1]      # the forpublication/ root
for _sub in ("common", "config", "db", "text_teacher"):
    _p = str(FORPUB / _sub)
    if _p not in sys.path:
        sys.path.append(_p)

import json
import random
from datetime import datetime

import torch
import numpy as np
import pandas as pd

from experiment_config import TEXT_TEACHER, DEVICE, NUM_CLASSES, MAX_LENGTH, SEEDS
from fetch_data import Split, DataSource, get_text_teacher_data
from TextTeacherDataset import MortalityTextDataset
from EicuTextDataset import EicuTextDataset
from TextTeacher import TextClassifier
from TextTeacherTrainer import TextTeacherTrainer
from class_weights import compute_class_weights

# --------------------------------------------------------------------------- #
# Config — from the single source of truth (config/experiment_config.py)
# --------------------------------------------------------------------------- #
config = TEXT_TEACHER
EXPERIMENT_TO_RUN = TEXT_TEACHER["experiment_name"]
CONSTANTS = {"device": DEVICE, "num_classes": NUM_CLASSES, "max_length": MAX_LENGTH}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_text_datasets(source: DataSource):
    """The ONLY source-specific seam. MIMIC aggregates multiple notes per
    admission ('last'); eICU already has one manufactured note per stay. Both
    yield datasets with the identical (input_ids, attention_mask, label, hadm_id)
    contract, so everything downstream is source-agnostic."""
    tr = get_text_teacher_data(Split.TRAIN, source)
    va = get_text_teacher_data(Split.VAL, source)
    te = get_text_teacher_data(Split.TEST, source)
    if source is DataSource.MIMIC:
        agg = config["aggregation"]
        return (MortalityTextDataset(tr.notes, tr.admissions, aggregation=agg),
                MortalityTextDataset(va.notes, va.admissions, aggregation=agg),
                MortalityTextDataset(te.notes, te.admissions, aggregation=agg))
    if source is DataSource.EICU:
        return EicuTextDataset(tr), EicuTextDataset(va), EicuTextDataset(te)
    raise TypeError(f"unhandled DataSource: {source!r}")


def main(seed: int, source: DataSource):
    set_seed(seed)
    ckpt_dir = str(FORPUB / "text_teacher" / "checkpoints" / source.value / EXPERIMENT_TO_RUN / f"seed{seed}")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    # --- Data + datasets (source-specific; everything below is identical) --- #
    train_dataset, val_dataset, test_dataset = build_text_datasets(source)

    # --- DataLoaders ------------------------------------------------------ #
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    # --- Model ------------------------------------------------------------ #
    model = TextClassifier(
        model_name=config["model_name"],
        num_classes=CONSTANTS["num_classes"],
        dropout=config["dropout"],
        freeze_layers=config["freeze_layers"],
    )

    # --- Class weights (inverse-frequency on train labels) ---------------- #
    train_labels = torch.tensor(train_dataset.labels)
    class_weights = compute_class_weights(train_labels)

    # --- Train (BIX1: alpha=0 -> pure supervised CE) ---------------------- #
    trainer = TextTeacherTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        num_epochs=config["num_epochs"],
        warmup_ratio=config["warmup_ratio"],
        patience=None,
        class_weights=class_weights,
        label_smoothing=config.get("label_smoothing", 0.0),
        checkpoint_dir=ckpt_dir,
        device=CONSTANTS["device"],
    )
    history = trainer.train()

    # --- Evaluate on test ------------------------------------------------- #
    trainer.load_checkpoint("best_model.pt")
    test_metrics = trainer.evaluate(test_loader)

    print("=" * 50)
    print(f"TEST RESULTS: {source.value} / {EXPERIMENT_TO_RUN}")
    print("=" * 50)
    print(f"Loss:     {test_metrics['loss']:.4f}")
    print(f"AUC:      {test_metrics['auc']:.4f}")
    print(f"F1:       {test_metrics['f1']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")

    # --- Save test predictions -------------------------------------------- #
    trainer.model.eval()
    all_preds, all_labels, all_hadm_ids = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(CONSTANTS["device"])
            attention_mask = batch["attention_mask"].to(CONSTANTS["device"])
            logits = trainer.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_preds.extend(probs.tolist())
            all_labels.extend(batch["label"].numpy().tolist())
            all_hadm_ids.extend(np.asarray(batch["hadm_id"]).tolist())

    predictions_df = pd.DataFrame({
        "hadm_id": all_hadm_ids,
        "true_label": all_labels,
        "pred_prob": all_preds,
        "pred_label": [1 if p > 0.5 else 0 for p in all_preds],
    })
    predictions_df.to_csv(f"{ckpt_dir}/test_predictions.csv", index=False)
    print(f"Test predictions saved: {len(predictions_df)} samples")

    # --- Save results.json ------------------------------------------------ #
    params = model.count_parameters()
    results = {
        "experiment": EXPERIMENT_TO_RUN,
        "source": source.value,
        "seed": seed,
        "config": config,
        "constants": CONSTANTS,
        "dataset": {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset),
            "class_distribution": torch.bincount(train_labels).tolist(),
            "class_weights": class_weights.tolist(),
        },
        "model": {
            "parameters_total": params["total"],
            "parameters_trainable": params["trainable"],
            "parameters_frozen": params["frozen"],
        },
        "training": {
            "history": history,
            "best_epoch": history["val_auc"].index(max(history["val_auc"])) + 1,
        },
        "test_metrics": test_metrics,
        "timestamp": datetime.now().isoformat(),
    }
    with open(f"{ckpt_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {ckpt_dir}/results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the baseline text teacher on MIMIC or eICU.")
    ap.add_argument("--source", required=True, choices=[s.value for s in DataSource],
                    help="which dataset to train on (mimic | eicu)")
    args = ap.parse_args()
    _source = DataSource(args.source)
    for _seed in SEEDS:
        main(_seed, _source)
