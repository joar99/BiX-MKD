"""
eICU Multimodal Student Dataset
===============================
The eICU counterpart of `MultimodalStudentDataset`.

Simpler than the MIMIC version: the eICU student fetch is ALREADY paired (one
manufactured note + a 236-dim vitals vector per stay), so there is no note
aggregation and no text/waveform inner join to do here — both happened upstream
(the paired-only cut in fed_split, the note assembly at build time). This class
tokenizes the note and standardizes the vitals.

Input: the DataFrame from
    fetch_data.get_multimodal_student_data(split, DataSource.EICU)
with columns [patientunitstayid, mortality, note, <feature_columns>].

Standardization mirrors the teachers: fit the scaler on TRAIN, reuse on VAL/TEST.

Output dict matches the MIMIC student dataset EXACTLY, so the shared
`MultimodalStudent` model and `DistillationTrainer` run unchanged:
    input_ids, attention_mask, waveform_features, label, hadm_id
`hadm_id` carries `patientunitstayid`.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional


class EicuStudentDataset(Dataset):

    def __init__(self,
                 df: pd.DataFrame,
                 feature_columns: list[str],
                 tokenizer_name: str = "emilyalsentzer/Bio_ClinicalBERT",
                 max_length: int = 512,
                 standardize: bool = False,
                 scaler: Optional[dict] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.feature_columns = feature_columns

        self.stay_ids = df["patientunitstayid"].values        # -> the 'hadm_id' key
        self.texts = df["note"].values
        self.labels = df["mortality"].values.astype(int)
        self.waveform_features = df[feature_columns].values.astype(np.float32)

        self._scaler = None
        if scaler is not None:
            self._scaler = scaler
            self._apply_standardization()
        elif standardize:
            self._scaler = {
                'mean': np.mean(self.waveform_features, axis=0),
                'std': np.std(self.waveform_features, axis=0),
            }
            self._apply_standardization()

        print(f"EicuStudentDataset: {len(self)} paired stays | "
              f"mortality {self.labels.mean():.3f} | "
              f"vitals dim {self.waveform_features.shape[1]}"
              + ("  standardized" if self._scaler is not None else ""))

    def _apply_standardization(self):
        mean = self._scaler['mean']
        std = self._scaler['std'].copy()
        std[std < 1e-8] = 1.0
        self.waveform_features = (self.waveform_features - mean) / std

    def get_scaler(self) -> Optional[dict]:
        return self._scaler

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(self.texts[idx],
                                  truncation=True,
                                  padding='max_length',
                                  max_length=self.max_length,
                                  return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'waveform_features': torch.tensor(
                self.waveform_features[idx], dtype=torch.float32),
            'label': torch.tensor(int(self.labels[idx]), dtype=torch.long),
            'hadm_id': self.stay_ids[idx],
        }
