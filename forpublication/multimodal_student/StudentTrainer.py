# Knowledge Distillation Trainer
# ===============================
# Training loop for the multimodal student model with the joint
# supervised-distillation objective from Section 3.3.3:
#
#   L = (1 - alpha) * L_sup + alpha * (L_KD_text + L_KD_wave) / 2
#
# where:
#   L_sup   = cross-entropy with ground-truth labels
#   L_KD_*  = KL divergence between student and teacher soft logits
#   alpha   = balance between supervised and distillation learning
#   tau     = temperature for softening logit distributions
#
# During each training step:
#   1. Student receives paired (text, waveform) input → produces logits
#   2. Text teacher (frozen) receives text input → produces soft logits
#   3. Waveform teacher (frozen) receives waveform input → produces soft logits
#   4. Combined loss is computed and backpropagated through student only
#
# When alpha=0, this reduces to pure supervised training (the baseline).
# This allows the same trainer to produce both the baseline and KD results.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from typing import Optional
import time
from pathlib import Path


class DistillationTrainer:
    """
    Trainer for the multimodal student with knowledge distillation.

    Supports three modes via the alpha parameter:
        alpha = 0.0  →  Pure supervised (multimodal baseline, no KD)
        alpha = 1.0  →  Pure distillation (no ground-truth supervision)
        0 < alpha < 1 →  Joint supervised-distillation (the FL-KD method)

    Teachers are frozen and used only for inference during training.
    """

    def __init__(self,
                 student: nn.Module,
                 text_teacher: Optional[nn.Module],
                 waveform_teacher: Optional[nn.Module],
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 alpha: float = 0.5,
                 temperature: float = 3.0,
                 learning_rate: float = 2e-5,
                 weight_decay: float = 0.01,
                 waveform_learning_rate: Optional[float] = None,
                 waveform_weight_decay:  Optional[float] = None,
                 fusion_learning_rate:   Optional[float] = None,
                 fusion_weight_decay:    Optional[float] = None,
                 num_epochs: int = 8,
                 warmup_ratio: float = 0.1,
                 patience: Optional[int] = None,
                 device: Optional[str] = None,
                 checkpoint_dir: Optional[str] = None,
                 class_weights: Optional[torch.Tensor] = None,
                 text_teacher_temperature: Optional[float] = None,
                 waveform_teacher_temperature: Optional[float] = None,
                text_kd_weight = 0.5,
                 waveform_kd_weight = 0.5):

        """
        Parameters
        ----------
        student : nn.Module
            The multimodal student model (MultimodalStudent).

        text_teacher : nn.Module, optional
            Frozen text teacher (TextClassifier). Required if alpha > 0.

        waveform_teacher : nn.Module, optional
            Frozen waveform teacher (WaveformClassifier). Required if alpha > 0.

        alpha : float
            Distillation weight in [0, 1].
            Loss = (1 - alpha) * L_sup + alpha * (L_KD_text + L_KD_wave) / 2

        temperature : float
            Softmax temperature for KD. Higher values produce softer
            probability distributions that encode more inter-class
            information. Hinton et al. (2015) suggest tau in [3, 20].

        learning_rate : float
            Learning rate for AdamW. Default 2e-5 follows the BERT
            fine-tuning convention since the student contains BERT.

        weight_decay : float
            L2 regularization weight for AdamW.

        num_epochs : int
            Maximum number of training epochs.

        warmup_ratio : float
            Fraction of total steps for linear LR warmup.

        patience : int, optional
            Early stopping patience (epochs without AUC improvement).
            If None, train for all num_epochs.

        device : str, optional
            Device to train on. Auto-detected if not specified.

        checkpoint_dir : str, optional
            Directory to save best model checkpoint.

        class_weights : torch.Tensor, optional
            Class weights for the supervised cross-entropy loss.
        """
        self.student = student
        self.text_teacher = text_teacher
        self.waveform_teacher = waveform_teacher
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.alpha = alpha
        self.temperature = temperature
        self.num_epochs = num_epochs
        self.patience = patience
        # Per-teacher calibrated temperatures (from temperature scaling)
        # If provided, teacher logits are pre-scaled by these before KD.
        # If None, raw teacher logits are used (original behavior).
        self.text_teacher_T = text_teacher_temperature
        self.waveform_teacher_T = waveform_teacher_temperature
        self.text_kd_weight = text_kd_weight
        self.waveform_kd_weight = waveform_kd_weight

        # Validate alpha and teacher availability
        if alpha > 0:
            assert text_teacher is not None, \
                "text_teacher required when alpha > 0"
            assert waveform_teacher is not None, \
                "waveform_teacher required when alpha > 0"

        # Device setup
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        # Mixed precision: fp16 matmuls on CUDA tensor cores, fp32 master
        # weights. No-op on mps/cpu. autocast keeps softmax/log_softmax/KLDiv
        # in fp32 by policy, so the temperature-scaled KD losses below are
        # computed at full precision even when enabled.
        self.amp_enabled = str(device) == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        # Move models to device
        self.student.to(self.device)
        if self.text_teacher is not None:
            self.text_teacher.to(self.device)
            self.text_teacher.eval()  # Freeze in eval mode permanently
        if self.waveform_teacher is not None:
            self.waveform_teacher.to(self.device)
            self.waveform_teacher.eval()

        # Supervised loss
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion_sup = nn.CrossEntropyLoss(weight=class_weights)

        # KL divergence loss for distillation
        # reduction='batchmean' gives the proper KL divergence estimate
        self.criterion_kd = nn.KLDivLoss(reduction='batchmean')

        # Optimizer — per-branch parameter groups, each branch inherits its
        # corresponding teacher's training hyperparameters where possible.
        #
        #   Text branch (BERT + text_projection)   : `learning_rate`, `weight_decay`
        #                                            → BERT fine-tuning convention,
        #                                              matches text teacher training.
        #   Waveform encoder                       : `waveform_learning_rate`,
        #                                            `waveform_weight_decay`
        #                                            → matches waveform teacher training.
        #   Fusion head (no teacher analog)        : `fusion_learning_rate`,
        #                                            `fusion_weight_decay`
        #                                            → defaults to the waveform
        #                                              regime (small MLP head).
        #
        # If any per-branch override is None, it falls back to the text-branch
        # defaults — so passing only `learning_rate` and `weight_decay` preserves
        # the original unified-LR behavior.
        wave_lr = waveform_learning_rate if waveform_learning_rate is not None else learning_rate
        wave_wd = waveform_weight_decay  if waveform_weight_decay  is not None else weight_decay
        fuse_lr = fusion_learning_rate   if fusion_learning_rate   is not None else wave_lr
        fuse_wd = fusion_weight_decay    if fusion_weight_decay    is not None else wave_wd

        self.optimizer = AdamW([
            {   # Text branch: BERT encoder
                "params":       self.student.text_encoder.parameters(),
                "lr":           learning_rate,
                "weight_decay": weight_decay,
            },
            {   # Text branch: projection head (trained in BERT regime)
                "params":       self.student.text_projection.parameters(),
                "lr":           learning_rate,
                "weight_decay": weight_decay,
            },
            {   # Waveform encoder: inherits waveform teacher's regime
                "params":       self.student.waveform_encoder.parameters(),
                "lr":           wave_lr,
                "weight_decay": wave_wd,
            },
            {   # Fusion head: no teacher analog; waveform-MLP regime by default
                "params":       self.student.fusion.parameters(),
                "lr":           fuse_lr,
                "weight_decay": fuse_wd,
            },
        ])

        # Stash for logging / audit
        self._effective_lrs = {
            "text_encoder":    learning_rate,
            "text_projection": learning_rate,
            "waveform_encoder": wave_lr,
            "fusion":           fuse_lr,
        }
        self._effective_wds = {
            "text_encoder":    weight_decay,
            "text_projection": weight_decay,
            "waveform_encoder": wave_wd,
            "fusion":           fuse_wd,
        }

        # LR scheduler with linear warmup (follows text teacher convention)
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Checkpoint directory
        self.checkpoint_dir = None
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training history — track all loss components for thesis analysis
        self.history = {
            'train_loss': [],
            'train_loss_sup': [],
            'train_loss_kd_text': [],
            'train_loss_kd_wave': [],
            'train_loss_kd_combined': [],
            'train_auc': [],
            'train_f1': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_auc': [],
            'val_f1': [],
            'val_accuracy': [],
            'learning_rate': [],
            # Branch diagnostics — see train() for how these are populated.
            # `val_auc_text_only`: val AUC when the waveform branch is zeroed.
            # `val_auc_wave_only`: val AUC when the text branch is zeroed.
            # Useful to tell which branch is doing the work at each epoch
            # and to spot per-branch overfitting (a branch whose ablated
            # AUC climbs then falls is likely overfitting).
            'val_auc_text_only': [],
            'val_auc_wave_only': [],
            # Per-param-group gradient norms (L2, averaged over train steps
            # in the epoch). Large disparity → unequal effective LR pressure.
            'train_grad_norm_bert':      [],
            'train_grad_norm_text_proj': [],
            'train_grad_norm_wave':      [],
            'train_grad_norm_fusion':    [],
        }

        # Stash param-group iterables on self for concise grad-norm loops
        self._param_groups_for_grads = {
            'bert':      list(self.student.text_encoder.parameters()),
            'text_proj': list(self.student.text_projection.parameters()),
            'wave':      list(self.student.waveform_encoder.parameters()),
            'fusion':    list(self.student.fusion.parameters()),
        }

    def _compute_kd_loss(self,
                         student_logits: torch.Tensor,
                         teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute the distillation loss between student and teacher logits.

        Uses KL divergence with temperature scaling as per Hinton et al. (2015).
        The loss is scaled by T^2 to ensure gradients are comparable
        across different temperature values.

        Parameters
        ----------
        student_logits : torch.Tensor, shape (batch_size, num_classes)
            Raw logits from the student model.

        teacher_logits : torch.Tensor, shape (batch_size, num_classes)
            Raw logits from the (frozen) teacher model.

        Returns
        -------
        kd_loss : torch.Tensor
            Scalar KL divergence loss, scaled by T^2.
        """
        T = self.temperature

        # Soft student distribution (log-probabilities for KLDivLoss input)
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)

        # Soft teacher distribution (probabilities as KLDivLoss target)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)

        # KL(teacher || student), scaled by T^2 per Hinton et al.
        kd_loss = self.criterion_kd(student_log_probs, teacher_probs) * (T ** 2)

        return kd_loss

    @staticmethod
    def _group_grad_norm(params) -> float:
        """L2 norm of concatenated gradients for a list of parameters."""
        grads = [p.grad for p in params if p.grad is not None]
        if not grads:
            return 0.0
        return float(torch.norm(torch.stack([g.norm() for g in grads])).item())

    def train_epoch(self) -> dict:
        """
        Train for one epoch. Returns dict of average loss components and
        per-param-group gradient norms averaged across steps.
        """
        self.student.train()

        # Teachers stay in eval mode — but we still need no_grad for them
        total_loss = 0.0
        total_sup = 0.0
        total_kd_text = 0.0
        total_kd_wave = 0.0

        # Per-group grad-norm accumulators (summed over steps, divided at end)
        grad_norm_sums = {k: 0.0 for k in self._param_groups_for_grads}
        grad_norm_n = 0

        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            waveform_features = batch['waveform_features'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                # ── Student forward pass ─────────────────────────────────
                student_logits = self.student(
                    input_ids, attention_mask, waveform_features
                )

                # ── Supervised loss ──────────────────────────────────────
                loss_sup = self.criterion_sup(student_logits, labels)

                # ── Distillation losses ──────────────────────────────────
                if self.alpha > 0:
                    with torch.no_grad():
                        # Text teacher sees only the text
                        text_teacher_logits = self.text_teacher(
                            input_ids, attention_mask
                        )
                        # Waveform teacher sees only the waveform features
                        waveform_teacher_logits = self.waveform_teacher(
                            waveform_features
                        )

                    # Pre-scale teacher logits with calibrated temperatures
                    # This corrects teacher miscalibration before softening
                    if self.text_teacher_T is not None:
                        text_teacher_logits = text_teacher_logits / self.text_teacher_T
                    if self.waveform_teacher_T is not None:
                        waveform_teacher_logits = waveform_teacher_logits / self.waveform_teacher_T

                    loss_kd_text = self._compute_kd_loss(
                        student_logits, text_teacher_logits
                    )
                    loss_kd_wave = self._compute_kd_loss(
                        student_logits, waveform_teacher_logits
                    )

                    # Weight the teachers based on provided weights
                    loss_kd_combined = (
                            self.text_kd_weight * loss_kd_text +
                            self.waveform_kd_weight * loss_kd_wave
                    )

                    # Combined objective: Section 3.3.3
                    loss = (1 - self.alpha) * loss_sup + self.alpha * loss_kd_combined
                else:
                    # Pure supervised (alpha=0), the multimodal baseline
                    loss_kd_text = torch.tensor(0.0)
                    loss_kd_wave = torch.tensor(0.0)
                    loss = loss_sup

            self.scaler.scale(loss).backward()
            # Unscale here so the per-group grad norms below are TRUE gradient
            # magnitudes, not the scaler's inflated ones — keeps the diagnostic
            # comparable to the fp32 runs.
            self.scaler.unscale_(self.optimizer)

            # Per-group grad norms *before* clipping and *before* optimizer
            # step, so we see the raw gradient signal. clip_grad_norm_ will
            # rescale after this measurement if any group's total grads
            # exceed max_norm.
            for name, params in self._param_groups_for_grads.items():
                grad_norm_sums[name] += self._group_grad_norm(params)
            grad_norm_n += 1

            torch.nn.utils.clip_grad_norm_(
                self.student.parameters(), max_norm=1.0
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            total_sup += loss_sup.item()
            total_kd_text += loss_kd_text.item()
            total_kd_wave += loss_kd_wave.item()

        n = len(self.train_loader)
        gn = max(grad_norm_n, 1)
        return {
            'loss': total_loss / n,
            'loss_sup': total_sup / n,
            'loss_kd_text': total_kd_text / n,
            'loss_kd_wave': total_kd_wave / n,
            'loss_kd_combined': (self.text_kd_weight * total_kd_text + self.waveform_kd_weight * total_kd_wave) / n,
            'grad_norm_bert':      grad_norm_sums['bert']      / gn,
            'grad_norm_text_proj': grad_norm_sums['text_proj'] / gn,
            'grad_norm_wave':      grad_norm_sums['wave']      / gn,
            'grad_norm_fusion':    grad_norm_sums['fusion']    / gn,
        }

    @torch.no_grad()
    def evaluate_ablated(self, data_loader: DataLoader,
                         use_text: bool, use_wave: bool) -> dict:
        """
        Evaluate the student with one branch zeroed out — branch-attribution
        diagnostic, not a full eval. Returns just {auc, f1, accuracy}; loss
        is omitted because the zeroed-branch outputs aren't calibrated for
        the supervised loss.

        Used to separate "which branch is doing the work" from "is the fused
        model overfitting." If val_auc_text_only climbs then falls while
        val_auc_wave_only plateaus, it's BERT that's overfitting, not the
        waveform branch.
        """
        assert use_text or use_wave, "Must keep at least one branch on."
        self.student.eval()

        all_logits, all_labels = [], []
        for batch in data_loader:
            input_ids         = batch['input_ids'].to(self.device)
            attention_mask    = batch['attention_mask'].to(self.device)
            waveform_features = batch['waveform_features'].to(self.device)
            labels            = batch['label'].to(self.device)

            logits = self.student.forward_ablated(
                input_ids, attention_mask, waveform_features,
                use_text=use_text, use_wave=use_wave,
            )
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        probs = torch.softmax(all_logits, dim=-1)[:, 1].numpy()
        preds = all_logits.argmax(dim=-1).numpy()
        labels_np = all_labels.numpy()

        return {
            'auc':      roc_auc_score(labels_np, probs),
            'f1':       f1_score(labels_np, preds),
            'accuracy': accuracy_score(labels_np, preds),
        }

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> dict:
        """
        Evaluate the student model on a given data loader.
        Uses only supervised loss and standard metrics for evaluation
        (KD loss is a training signal, not an evaluation metric).
        """
        self.student.eval()

        total_loss = 0.0
        all_logits = []
        all_labels = []

        for batch in data_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            waveform_features = batch['waveform_features'].to(self.device)
            labels = batch['label'].to(self.device)

            logits = self.student(
                input_ids, attention_mask, waveform_features
            )
            loss = self.criterion_sup(logits, labels)

            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        probs = torch.softmax(all_logits, dim=-1)[:, 1].numpy()
        preds = all_logits.argmax(dim=-1).numpy()
        labels_np = all_labels.numpy()

        return {
            'loss': total_loss / len(data_loader),
            'auc': roc_auc_score(labels_np, probs),
            'f1': f1_score(labels_np, preds),
            'accuracy': accuracy_score(labels_np, preds)
        }

    def train(self) -> dict:
        """
        Full training loop with optional early stopping.
        """
        mode_str = (
            "pure supervised (alpha=0, no KD)"
            if self.alpha == 0
            else f"KD (alpha={self.alpha}, tau={self.temperature})"
        )

        print(f"Training on {self.device}")
        print(f"Mode: {mode_str}")
        print(f"Train batches: {len(self.train_loader)}, "
              f"Val batches: {len(self.val_loader)}")
        if self.patience is not None:
            print(f"Early stopping patience: {self.patience}")
        print("Per-branch LRs:  "
              f"text_encoder={self._effective_lrs['text_encoder']:.0e}, "
              f"text_proj={self._effective_lrs['text_projection']:.0e}, "
              f"wave={self._effective_lrs['waveform_encoder']:.0e}, "
              f"fusion={self._effective_lrs['fusion']:.0e}")
        print("Per-branch WDs:  "
              f"text_encoder={self._effective_wds['text_encoder']:.0e}, "
              f"text_proj={self._effective_wds['text_projection']:.0e}, "
              f"wave={self._effective_wds['waveform_encoder']:.0e}, "
              f"fusion={self._effective_wds['fusion']:.0e}")
        print("-" * 60)

        best_auc = 0.0
        epochs_without_improvement = 0

        for epoch in range(self.num_epochs):
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Evaluate on both splits
            train_eval_metrics = self.evaluate(self.train_loader)
            val_metrics = self.evaluate(self.val_loader)

            # Branch-attribution ablations (val only, cheap)
            val_text_only = self.evaluate_ablated(
                self.val_loader, use_text=True, use_wave=False,
            )
            val_wave_only = self.evaluate_ablated(
                self.val_loader, use_text=False, use_wave=True,
            )

            epoch_time = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]['lr']

            # Record history
            self.history['train_auc'].append(train_eval_metrics['auc'])
            self.history['train_f1'].append(train_eval_metrics['f1'])
            self.history['train_accuracy'].append(train_eval_metrics['accuracy'])
            self.history['learning_rate'].append(current_lr)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_loss_sup'].append(train_metrics['loss_sup'])
            self.history['train_loss_kd_text'].append(
                train_metrics['loss_kd_text']
            )
            self.history['train_loss_kd_wave'].append(
                train_metrics['loss_kd_wave']
            )
            self.history['train_loss_kd_combined'].append(
                train_metrics['loss_kd_combined']
            )
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])

            # Branch-attribution and grad-norm diagnostics
            self.history['val_auc_text_only'].append(val_text_only['auc'])
            self.history['val_auc_wave_only'].append(val_wave_only['auc'])
            self.history['train_grad_norm_bert'     ].append(train_metrics['grad_norm_bert'])
            self.history['train_grad_norm_text_proj'].append(train_metrics['grad_norm_text_proj'])
            self.history['train_grad_norm_wave'     ].append(train_metrics['grad_norm_wave'])
            self.history['train_grad_norm_fusion'   ].append(train_metrics['grad_norm_fusion'])

            # Print progress
            print(f"Epoch {epoch + 1}/{self.num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_metrics['loss']:.4f} "
                  f"(sup: {train_metrics['loss_sup']:.4f}"
                  f", kd_text: {train_metrics['loss_kd_text']:.4f}"
                  f", kd_wave: {train_metrics['loss_kd_wave']:.4f})"
                  if self.alpha > 0
                  else f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | "
                  f"AUC: {val_metrics['auc']:.4f} | "
                  f"F1: {val_metrics['f1']:.4f} | "
                  f"Acc: {val_metrics['accuracy']:.4f}")
            print(f"  Branch AUC:  text-only {val_text_only['auc']:.4f}  "
                  f"wave-only {val_wave_only['auc']:.4f}  "
                  f"(fused-text Δ={val_metrics['auc'] - val_text_only['auc']:+.4f}, "
                  f"fused-wave Δ={val_metrics['auc'] - val_wave_only['auc']:+.4f})")
            print(f"  Grad norms:  bert={train_metrics['grad_norm_bert']:.3f}  "
                  f"text_proj={train_metrics['grad_norm_text_proj']:.3f}  "
                  f"wave={train_metrics['grad_norm_wave']:.3f}  "
                  f"fusion={train_metrics['grad_norm_fusion']:.3f}")

            # Checkpoint on best AUC
            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                epochs_without_improvement = 0
                print(f"  ↑ New best AUC!")

                if self.checkpoint_dir:
                    self.save_checkpoint('best_model.pt')
            else:
                epochs_without_improvement += 1
                if (self.patience is not None and
                        epochs_without_improvement >= self.patience):
                    print(f"\n  Early stopping after {epoch + 1} epochs "
                          f"(no improvement for {self.patience} epochs)")
                    break

        print("-" * 60)
        print(f"Training complete. Best AUC: {best_auc:.4f}")

        return self.history

    @torch.no_grad()
    def get_test_predictions(self, data_loader: DataLoader) -> dict:
        """
        Generate detailed predictions for the test set.
        Returns hadm_ids, true labels, predicted probabilities,
        and predicted labels for saving to CSV.
        """
        self.student.eval()

        all_hadm_ids = []
        all_labels = []
        all_probs = []

        for batch in data_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            waveform_features = batch['waveform_features'].to(self.device)

            logits = self.student(
                input_ids, attention_mask, waveform_features
            )
            probs = torch.softmax(logits, dim=-1)[:, 1]

            all_hadm_ids.extend(batch['hadm_id'].tolist())
            all_labels.extend(batch['label'].tolist())
            all_probs.extend(probs.cpu().tolist())

        all_preds = [1 if p >= 0.5 else 0 for p in all_probs]

        return {
            'hadm_id': all_hadm_ids,
            'true_label': all_labels,
            'pred_prob': all_probs,
            'pred_label': all_preds
        }

    def save_checkpoint(self, filename: str):
        if self.checkpoint_dir is None:
            return
        path = self.checkpoint_dir / filename
        torch.save(self.student.state_dict(), path)
        print(f"  Saved checkpoint: {path}")

    def load_checkpoint(self, filename: str):
        path = self.checkpoint_dir / filename
        self.student.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        print(f"Loaded checkpoint: {path}")