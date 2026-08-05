# Multimodal Student Model
# ========================
# Two-branch fusion architecture for knowledge distillation.
#
# Text branch:   BioClinicalBERT encoder → projection → text representation
# Waveform branch: MLP encoder → projection → waveform representation
# Fusion:        Concatenation → fusion MLP → mortality logits
#
# During training, the student receives paired (text, waveform) inputs
# and is supervised by:
#   1. Ground-truth mortality labels (cross-entropy)
#   2. Text teacher soft logits (KL divergence)
#   3. Waveform teacher soft logits (KL divergence)
#
# The combined loss is controlled by alpha (distillation weight) and
# temperature (softmax temperature for KD).

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional

class MultimodalStudent(nn.Module):
    """
    Two-branch multimodal fusion model for mortality prediction.

    Architecture:
        text_input → BioClinicalBERT → [768] → projection → [text_proj_dim]  ─┐
                                                                                ├─ concat → fusion MLP → [2] logits
        waveform_input → MLP backbone → [wave_proj_dim]                       ─┘

    The text branch reuses the same BioClinicalBERT encoder as the text teacher.
    The waveform branch is a small MLP that maps the 317-dim feature vector
    to a learned representation.

    Both branches project to a common dimensionality before concatenation,
    giving the fusion layer balanced input from each modality.
    """

    def __init__(self,
                 # Text branch config
                 text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
                 text_proj_dim: int = 256,
                 text_dropout: float = 0.1,
                 freeze_bert: bool = False,
                 freeze_bert_layers: Optional[int] = None,

                 # Waveform branch config
                 waveform_input_dim: int = 317,
                 waveform_hidden_dims: Optional[list[int]] = None,
                 wave_proj_dim: int = 128,
                 waveform_dropout: float = 0.1,
                 waveform_batch_norm: bool = True,

                 # Fusion config
                 fusion_hidden_dim: int = 128,
                 fusion_dropout: float = 0.2,
                 num_classes: int = 2):
        """
        Parameters
        ----------
        text_model_name : str
            HuggingFace model identifier for the text encoder.

        text_proj_dim : int
            Dimension of the text branch projection. The 768-dim CLS output
            is projected down to this size before fusion.

        text_dropout : float
            Dropout applied to the CLS output before projection.

        freeze_bert : bool
            If True, freeze the entire BERT encoder. Only the projection
            and fusion layers will be trained.

        freeze_bert_layers : int, optional
            If set, freeze the embedding layer and the first N transformer
            layers. Allows fine-tuning of upper layers only.

        waveform_input_dim : int
            Dimension of the waveform feature vector (317 for V2 features).

        waveform_hidden_dims : list[int], optional
            Hidden layer sizes for the waveform branch MLP. Default: [256, 128].

        wave_proj_dim : int
            Output dimension of the waveform branch (before fusion).

        waveform_dropout : float
            Dropout in the waveform branch MLP.

        waveform_batch_norm : bool
            Whether to use batch normalisation in the waveform branch.

        fusion_hidden_dim : int
            Hidden dimension of the fusion MLP.

        fusion_dropout : float
            Dropout in the fusion layer.

        num_classes : int
            Number of output classes (2 for binary mortality prediction).
        """
        super().__init__()

        # ── Text Branch ──────────────────────────────────────────────
        self.text_config = AutoConfig.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        bert_hidden_size = self.text_config.hidden_size  # 768

        self.text_dropout = nn.Dropout(text_dropout)
        self.text_projection = nn.Sequential(nn.Linear(bert_hidden_size, text_proj_dim),
                                             nn.ReLU(),
                                             nn.Dropout(text_dropout))

        # Handle freezing
        if freeze_bert:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        elif freeze_bert_layers is not None:
            self._freeze_bert_layers(freeze_bert_layers)

        # ── Waveform Branch ──────────────────────────────────────────
        if waveform_hidden_dims is None:
            waveform_hidden_dims = [256, 128]

        wave_layers = []
        prev_dim = waveform_input_dim

        for h_dim in waveform_hidden_dims:
            wave_layers.append(nn.Linear(prev_dim, h_dim))
            if waveform_batch_norm:
                wave_layers.append(nn.BatchNorm1d(h_dim))
            wave_layers.append(nn.ReLU())
            wave_layers.append(nn.Dropout(waveform_dropout))
            prev_dim = h_dim

        # Final projection to wave_proj_dim (if last hidden != wave_proj_dim)
        if prev_dim != wave_proj_dim:
            wave_layers.append(nn.Linear(prev_dim, wave_proj_dim))
            wave_layers.append(nn.ReLU())
            wave_layers.append(nn.Dropout(waveform_dropout))

        self.waveform_encoder = nn.Sequential(*wave_layers)

        # ── Fusion Head ──────────────────────────────────────────────
        fusion_input_dim = text_proj_dim + wave_proj_dim

        self.fusion = nn.Sequential(nn.Linear(fusion_input_dim, fusion_hidden_dim),
                                    nn.ReLU(),
                                    nn.Dropout(fusion_dropout),
                                    nn.Linear(fusion_hidden_dim, num_classes))

        # Store dims for reference
        self.text_proj_dim = text_proj_dim
        self.wave_proj_dim = wave_proj_dim

    def _freeze_bert_layers(self, n: int):
        """Freeze embedding layer and first n transformer layers."""
        for param in self.text_encoder.embeddings.parameters():
            param.requires_grad = False

        for i, layer in enumerate(self.text_encoder.encoder.layer):
            if i < n:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                waveform_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through both branches and fusion.

        Parameters
        ----------
        input_ids : torch.Tensor, shape (batch_size, seq_len)
            Tokenized clinical note input IDs.

        attention_mask : torch.Tensor, shape (batch_size, seq_len)
            Attention mask for the tokenized text.

        waveform_features : torch.Tensor, shape (batch_size, waveform_input_dim)
            Pre-computed waveform feature vector.

        Returns
        -------
        logits : torch.Tensor, shape (batch_size, num_classes)
            Mortality prediction logits.
        """
        # Text branch: BERT → CLS token → projection
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = text_outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        cls_output = self.text_dropout(cls_output)
        text_repr = self.text_projection(cls_output)  # [batch, text_proj_dim]

        # Waveform branch: features → MLP → representation
        wave_repr = self.waveform_encoder(waveform_features)  # [batch, wave_proj_dim]

        # Fusion: concatenate → MLP → logits
        fused = torch.cat([text_repr, wave_repr], dim=-1)  # [batch, text_proj_dim + wave_proj_dim]
        logits = self.fusion(fused)  # [batch, num_classes]

        return logits

    def forward_ablated(self,
                        input_ids: torch.Tensor,
                        attention_mask: torch.Tensor,
                        waveform_features: torch.Tensor,
                        use_text: bool = True,
                        use_wave: bool = True) -> torch.Tensor:
        """
        Forward pass with one branch optionally zeroed out, for branch-attribution eval.

        When ``use_text=False``, the text branch's contribution to fusion is
        set to a zero vector; BERT is skipped entirely (cheap). Similarly
        for ``use_wave=False``. The fusion head still runs — we're asking
        "what does the fused prediction look like if this branch contributed
        nothing?", which is exactly what a per-branch ablation eval answers.

        Intended ONLY for diagnostic evaluation, not training: the model was
        trained seeing both branches, so zeroed-branch predictions are
        out-of-distribution. They're meaningful as a *relative* signal
        (e.g. "val AUC drops by 0.05 when text is removed") more than as
        absolute predictions.
        """
        batch_size = waveform_features.shape[0]
        device = waveform_features.device

        if use_text:
            text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = text_outputs.last_hidden_state[:, 0, :]
            cls_output = self.text_dropout(cls_output)
            text_repr = self.text_projection(cls_output)
        else:
            text_repr = torch.zeros(batch_size, self.text_proj_dim, device=device)

        if use_wave:
            wave_repr = self.waveform_encoder(waveform_features)
        else:
            wave_repr = torch.zeros(batch_size, self.wave_proj_dim, device=device)

        fused = torch.cat([text_repr, wave_repr], dim=-1)
        return self.fusion(fused)

    def predict_proba(self,
                      input_ids: torch.Tensor,
                      attention_mask: torch.Tensor,
                      waveform_features: torch.Tensor,
                      temperature: float = 1.0) -> torch.Tensor:
        logits = self.forward(input_ids, attention_mask, waveform_features)
        return torch.softmax(logits / temperature, dim=-1)

    def get_branch_representations(self,
                                   input_ids: torch.Tensor,
                                   attention_mask: torch.Tensor,
                                   waveform_features: torch.Tensor) -> dict:

        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = text_outputs.last_hidden_state[:, 0, :]
        cls_output = self.text_dropout(cls_output)
        text_repr = self.text_projection(cls_output)

        wave_repr = self.waveform_encoder(waveform_features)

        fused = torch.cat([text_repr, wave_repr], dim=-1)

        return {
            'text_repr': text_repr,
            'wave_repr': wave_repr,
            'fused_repr': fused
        }

    # Count total | trainable | frozen -> parameters by component
    def count_parameters(self) -> dict:
        def _count(module):
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total, trainable

        text_total, text_train = _count(self.text_encoder)
        text_proj_total, text_proj_train = _count(self.text_projection)
        wave_total, wave_train = _count(self.waveform_encoder)
        fusion_total, fusion_train = _count(self.fusion)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
            'trainable_pct': trainable / total * 100,
            'breakdown': {
                'text_encoder': {'total': text_total, 'trainable': text_train},
                'text_projection': {'total': text_proj_total, 'trainable': text_proj_train},
                'waveform_encoder': {'total': wave_total, 'trainable': wave_train},
                'fusion': {'total': fusion_total, 'trainable': fusion_train},
            }
        }

#Load a multimodal student from a checkpoint
def load_student(checkpoint_path: str,
                 device: str = "cpu",
                 **model_kwargs) -> MultimodalStudent:
    model = MultimodalStudent(**model_kwargs)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model