"""
eICU Vitals Dataset
===================
The eICU counterpart of `MortalityWaveformDataset` (WaveformTeacherDataset.py).

Identical in spirit — z-score standardization with a scaler fit on TRAIN and
reused on VAL/TEST — but keyed on `patientunitstayid` instead of `hadm_id`, and
the feature vector is the 236-dim eICU vitals recipe (3 signals, 2 interaction
pairs) rather than MIMIC's 317-dim waveform recipe.

Input: the DataFrame from
    fetch_data.get_waveform_teacher_data(split, DataSource.EICU)
with columns [patientunitstayid, mortality, <feature_columns>].

Usage mirrors the MIMIC dataset:
    train_ds = EicuWaveformDataset(train_df, feature_columns=cols, standardize=True)
    val_ds   = EicuWaveformDataset(val_df,   feature_columns=cols, scaler=train_ds.get_scaler())

Output dict matches the MIMIC dataset EXACTLY (features, label, hadm_id), so the
shared `WaveformTeacher` model and trainer run unchanged. `hadm_id` carries
`patientunitstayid`.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional


class EicuWaveformDataset(Dataset):

    def __init__(self,
                 df: pd.DataFrame,
                 feature_columns: list[str],
                 standardize: bool = False,
                 scaler: Optional[dict] = None):
        self.feature_columns = feature_columns
        self.stay_ids = df["patientunitstayid"].values        # -> the 'hadm_id' key
        self.labels = df["mortality"].values.astype(int)
        self.features = df[feature_columns].values.astype(np.float32)

        self._scaler = None
        if scaler is not None:                # val/test: reuse the train scaler
            self._scaler = scaler
            self._apply_standardization()
        elif standardize:                     # train: fit the scaler here
            self._scaler = {
                'mean': np.mean(self.features, axis=0),
                'std': np.std(self.features, axis=0),
            }
            self._apply_standardization()

        print(f"EicuWaveformDataset: {len(self)} stays | "
              f"pos {self.labels.sum()} / neg {(1 - self.labels).sum()} | "
              f"feature dim {self.features.shape[1]}"
              + ("  standardized" if self._scaler is not None else ""))

    def _apply_standardization(self):
        mean = self._scaler['mean']
        std = self._scaler['std'].copy()
        std[std < 1e-8] = 1.0                 # constant-feature guard
        self.features = (self.features - mean) / std

    def get_scaler(self) -> Optional[dict]:
        return self._scaler

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'label': torch.tensor(int(self.labels[idx]), dtype=torch.long),
            'hadm_id': self.stay_ids[idx],
        }
