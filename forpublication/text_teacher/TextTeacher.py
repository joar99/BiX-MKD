import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional


class TextClassifier(nn.Module):

    def __init__(self,
                 model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
                 num_classes: int = 2,
                 dropout: float = 0.1,
                 freeze_encoder: bool = False,
                 freeze_layers: Optional[int] = None):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        elif freeze_layers is not None:
            self._freeze_n_layers(freeze_layers)

    def _freeze_n_layers(self, n: int):
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False

        for i, layer in enumerate(self.encoder.encoder.layer):
            if i < n:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        return logits

    def predict_proba(self,
                      input_ids: torch.Tensor,
                      attention_mask: torch.Tensor,
                      temperature: float = 1.0) -> torch.Tensor:

        logits = self.forward(input_ids, attention_mask)
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


def load_classifier(checkpoint_path: str,
                    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
                    device: str = "cpu") -> TextClassifier:

    model = TextClassifier(model_name=model_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    return model