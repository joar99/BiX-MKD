# Waveform Classifier (MLP)
#
# Feedforward classifier for the waveform teacher model.
# Takes an x-dim feature vector and produces mortality logits

import torch
import torch.nn as nn
from waveform_features_v2 import FEATURE_DIM

class WaveformClassifier(nn.Module):

    def __init__(self,
                 input_dim: int = FEATURE_DIM,
                 hidden_dims: list[int] | None = None,
                 num_classes: int = 2,
                 dropout: float = 0.3,
                 batch_norm: bool = True):

        # Params:
        #
        # input_dim : int
        # Dimension of input feature vector
        #
        # hidden_dims : list[int], optional
        # Hidden layer sizes. Default: [128, 64].
        #
        # num_classes : int
        # Number of output classes (we use 2 for binary classification dead or not).
        #
        # dropout : float
        # Dropout rate applied after each hidden layer.
        #
        # batch_norm : bool
        # Whether to apply batch normalisation after each hidden layer.

        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:

        # Params:
        #
        # features : torch.Tensor of shape (batch_size, input_dim)
        #
        # Returns:
        #
        # logits : torch.Tensor of shape (batch_size, num_classes)

        h = self.backbone(features)
        return self.classifier(h)

    def predict_proba(self,
                      features: torch.Tensor,
                      temperature: float = 1.0) -> torch.Tensor:
        # Produce soft probability distribution
        logits = self.forward(features)
        return torch.softmax(logits / temperature, dim=-1)

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen,
            'trainable_pct': trainable / total * 100
        }