# Text Teacher Trainer
# ====================
# Self-contained trainer for the TEXT teacher (BioClinicalBERT), covering
# ALL variants:
#   - BIX1 (baseline pretraining): alpha=0 -> pure supervised CE.
#   - BIX2-4 (reverse KD):         alpha>0 -> CE + soft labels from the
#                                  multimodal student.
# The alpha knob toggles the KD branch; everything else is identical.
#
# Text-specific choices (baked in, not parameterised):
#   - Scheduler: linear warmup + linear decay (per-batch step).
#   - Forward:   model(input_ids, attention_mask).
#
# KD objective (alpha>0), Hinton (2015):
#     L = (1 - alpha) * CE
#         + alpha * tau^2 * KL(softmax(model_logits/tau) || softmax(student_soft/tau))

import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader


def _hadm_to_int_list(batch_hadm) -> list:
    """Coerce batch['hadm_id'] into a list of ints (handles tensor and ndarray)."""
    if torch.is_tensor(batch_hadm):
        return batch_hadm.tolist()
    return list(batch_hadm)


class TextTeacherTrainer:
    """
    Trains the text teacher (TextClassifier / BioClinicalBERT) for any BIX variant.

    soft_logits : dict[int, torch.Tensor], optional
        Pre-computed multimodal-student logits keyed by hadm_id. Required
        when alpha > 0; ignored when alpha == 0 (baseline).
    alpha : float
        KD weight in [0, 1]. 0 => pure CE (baseline / BIX1).
    warmup_ratio : float
        Fraction of total steps used for linear warmup.
    """

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 *,
                 learning_rate: float,
                 weight_decay: float,
                 num_epochs: int,
                 warmup_ratio: float,
                 patience: Optional[int] = None,
                 class_weights: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0,
                 soft_logits: Optional[Dict[int, torch.Tensor]] = None,
                 alpha: float = 0.0,
                 temperature: float = 1.0,
                 device: str = "mps",
                 checkpoint_dir: Optional[str] = None,
                 grad_clip_max_norm: float = 1.0,
                 epoch_logits_dir: Optional[str] = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.patience = patience
        self.alpha = alpha
        self.temperature = temperature
        self.grad_clip_max_norm = grad_clip_max_norm

        self.soft_logits = soft_logits
        if alpha > 0:
            assert soft_logits is not None, "alpha > 0 requires soft_logits"
        if soft_logits is not None and alpha == 0:
            print("⚠ soft_logits provided but alpha=0 — KD branch will be skipped")

        self.device = device
        # Mixed precision: fp16 matmuls on CUDA tensor cores, fp32 master
        # weights. No-op on mps/cpu — autocast and the scaler both disable
        # cleanly, so those runs stay bit-for-bit plain fp32.
        self.amp_enabled = str(device) == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.model.to(device)

        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion_sup = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=label_smoothing,
        )
        self.criterion_kd = nn.KLDivLoss(reduction="batchmean")

        self.optimizer = AdamW(model.parameters(),
                               lr=learning_rate,
                               weight_decay=weight_decay)

        # Text convention: linear warmup + linear decay, stepped per batch.
        total_steps  = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.epoch_logits_dir = Path(epoch_logits_dir) if epoch_logits_dir else None
        if self.epoch_logits_dir:
            self.epoch_logits_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            "train_loss":     [],
            "train_loss_sup": [],
            "train_loss_kd":  [],
            "val_loss":       [],
            "val_auc":        [],
            "val_f1":         [],
            "val_accuracy":   [],
            "learning_rate":  [],
        }

    def _forward(self, batch) -> torch.Tensor:
        return self.model(batch["input_ids"].to(self.device),
                          batch["attention_mask"].to(self.device))

    # ── KD helpers ─────────────────────────────────────────────────────
    def _kd_loss(self, model_logits: torch.Tensor,
                 student_soft_logits: torch.Tensor) -> torch.Tensor:
        T = self.temperature
        log_p_model = F.log_softmax(model_logits / T, dim=-1)
        p_student   = F.softmax(student_soft_logits / T, dim=-1)
        return self.criterion_kd(log_p_model, p_student) * (T ** 2)

    def _gather_soft(self, hadm_ids) -> torch.Tensor:
        ids = _hadm_to_int_list(hadm_ids)
        return torch.stack([self.soft_logits[int(h)] for h in ids]).to(self.device)

    # ── Training / evaluation loops ────────────────────────────────────
    def train_epoch(self) -> dict:
        self.model.train()
        tot_loss = tot_sup = tot_kd = 0.0

        for batch in self.train_loader:
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                logits   = self._forward(batch)
                loss_sup = self.criterion_sup(logits, labels)

                if self.soft_logits is not None and self.alpha > 0:
                    soft = self._gather_soft(batch["hadm_id"])
                    loss_kd = self._kd_loss(logits, soft)
                    loss    = (1 - self.alpha) * loss_sup + self.alpha * loss_kd
                else:
                    loss_kd = torch.tensor(0.0)
                    loss    = loss_sup

            self.scaler.scale(loss).backward()
            # Unscale before clipping: clip_grad_norm_ must see true gradient
            # magnitudes, not the scaler's inflated ones.
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           max_norm=self.grad_clip_max_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()   # per-batch (warmup)

            tot_loss += loss.item()
            tot_sup  += loss_sup.item()
            tot_kd   += loss_kd.item()

        n = len(self.train_loader)
        return {
            "loss":     tot_loss / n,
            "loss_sup": tot_sup  / n,
            "loss_kd":  tot_kd   / n,
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, return_logits: bool = False) -> dict:
        self.model.eval()
        tot_loss = 0.0
        all_logits, all_labels, all_hadm = [], [], []

        for batch in loader:
            labels = batch["label"].to(self.device)
            logits = self._forward(batch)
            loss   = self.criterion_sup(logits, labels)
            tot_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_hadm.extend(_hadm_to_int_list(batch["hadm_id"]))

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        probs = torch.softmax(all_logits, dim=-1)[:, 1].numpy()
        preds = all_logits.argmax(dim=-1).numpy()
        labels_np = all_labels.numpy()

        out = {
            "loss":     tot_loss / len(loader),
            "auc":      float(roc_auc_score(labels_np, probs)),
            "f1":       float(f1_score(labels_np, preds, zero_division=0)),
            "accuracy": float(accuracy_score(labels_np, preds)),
        }
        if return_logits:
            out["logits"]   = all_logits
            out["labels"]   = all_labels
            out["hadm_ids"] = all_hadm
        return out

    @torch.no_grad()
    def get_predictions(self, loader: DataLoader) -> dict:
        self.model.eval()
        out_hadm, out_lbl, out_prob = [], [], []
        for batch in loader:
            logits = self._forward(batch)
            probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            out_hadm.extend(_hadm_to_int_list(batch["hadm_id"]))
            out_lbl.extend(batch["label"].tolist())
            out_prob.extend(probs.tolist())
        out_pred = [1 if p >= 0.5 else 0 for p in out_prob]
        return {
            "hadm_id":    out_hadm,
            "true_label": out_lbl,
            "pred_prob":  out_prob,
            "pred_label": out_pred,
        }

    # ── Full training driver ───────────────────────────────────────────
    def train(self) -> dict:
        kd_str = (f"alpha={self.alpha} tau={self.temperature}"
                  if self.alpha > 0 else "no-KD (baseline)")
        print(f"Training text teacher on {self.device}  |  scheduler=linear_warmup  |  {kd_str}")
        print(f"Train batches: {len(self.train_loader)}  Val batches: {len(self.val_loader)}")
        if self.patience is not None:
            print(f"Early stopping patience: {self.patience}")
        print("-" * 60)

        best_auc = 0.0
        no_improve = 0

        for epoch in range(self.num_epochs):
            t0 = time.time()
            train_metrics = self.train_epoch()
            want_logits = self.epoch_logits_dir is not None
            val = self.evaluate(self.val_loader, return_logits=want_logits)
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"]    .append(train_metrics["loss"])
            self.history["train_loss_sup"].append(train_metrics["loss_sup"])
            self.history["train_loss_kd"] .append(train_metrics["loss_kd"])
            self.history["val_loss"]      .append(val["loss"])
            self.history["val_auc"]       .append(val["auc"])
            self.history["val_f1"]        .append(val["f1"])
            self.history["val_accuracy"]  .append(val["accuracy"])
            self.history["learning_rate"] .append(current_lr)

            print(f"Epoch {epoch+1}/{self.num_epochs} ({time.time()-t0:.1f}s) "
                  f"[lr={current_lr:.2e}]")
            if self.alpha > 0:
                print(f"  Train loss: {train_metrics['loss']:.4f} "
                      f"(sup {train_metrics['loss_sup']:.4f}, "
                      f"kd {train_metrics['loss_kd']:.4f})")
            else:
                print(f"  Train loss: {train_metrics['loss']:.4f}")
            print(f"  Val   loss: {val['loss']:.4f}  AUC {val['auc']:.4f}  "
                  f"F1 {val['f1']:.4f}  Acc {val['accuracy']:.4f}")

            if want_logits:
                torch.save({
                    "epoch":    epoch + 1,
                    "logits":   val["logits"],
                    "labels":   val["labels"],
                    "hadm_ids": val["hadm_ids"],
                }, self.epoch_logits_dir / f"val_epoch_{epoch+1:02d}.pt")

            if val["auc"] > best_auc:
                best_auc = val["auc"]
                no_improve = 0
                if self.checkpoint_dir:
                    self.save_checkpoint("best_model.pt")
                print("  ↑ New best AUC")
            else:
                no_improve += 1
                if self.patience is not None and no_improve >= self.patience:
                    print(f"\n  Early stopping after {epoch+1} epochs "
                          f"(no improvement for {self.patience} epochs)")
                    break

        print("-" * 60)
        print(f"Best val AUC: {best_auc:.4f}")
        return self.history

    # ── Checkpointing ──────────────────────────────────────────────────
    def save_checkpoint(self, filename: str):
        if self.checkpoint_dir is None:
            return
        torch.save(self.model.state_dict(), self.checkpoint_dir / filename)

    def load_checkpoint(self, filename: str):
        path = self.checkpoint_dir / filename
        self.model.load_state_dict(torch.load(path, map_location=self.device))
