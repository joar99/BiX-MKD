"""
Waveform Dataset — V2
=====================
PyTorch Dataset wrapping pre-computed waveform feature vectors.
Extends V1 with optional feature standardization (z-score normalisation).

Usage:
    # Training set: compute stats and standardize
    train_dataset = MortalityWaveformDataset(train_features, feature_columns=cols, standardize=True)

    # Val/test sets: reuse training stats
    val_dataset = MortalityWaveformDataset(val_features, feature_columns=cols, scaler=train_dataset.get_scaler())
    test_dataset = MortalityWaveformDataset(test_features, feature_columns=cols, scaler=train_dataset.get_scaler())
"""

# Waveform Dataset V2
# Pytorch dataset wrapping pre-computed waveform feature vectors
# Includes optional feature standardization

# Usage:
# train_dataset = MortalityWaveformDataset(train_features, feature_columns=cols, standardize=True)
# val_dataset = MortalityWaveformDataset(val_features, feature_columns=cols, scaler=train_dataset.get_scaler())
# test_dataset = MortalityWaveformDataset(test_features, feature_columns=cols, scaler=train_dataset.get_scaler())

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Optional

class MortalityWaveformDataset(Dataset):

    def __init__(self,
                 features_df: pd.DataFrame,
                 feature_columns: list[str],
                 standardize: bool = False,
                 scaler: Optional[dict] = None):
        # Params:
        # features_df : pd.DataFrame
        # Must contain 'hadm_id', 'mortality', and the feature columns.
        #
        # feature_columns : list[str]
        # Column names for the features
        #
        # standardize : bool
        # If True, compute mean/std from this data and apply z-score
        # normalisation. Use for the training set.
        #
        # scaler: dict, optional
        # Pre-computed {'mean': np.ndarray, 'std': np.ndarray} from the
        # training set. Use this for val/test sets to avoid data leakage.
        # If provided, standardize is ignored.

        self.feature_columns = feature_columns
        self.hadm_ids = features_df['hadm_id'].values
        self.labels = features_df['mortality'].values.astype(int)
        self.features = features_df[feature_columns].values.astype(np.float32)

        # Store scaler parameters
        self._scaler = None

        if scaler is not None:
            # Apply pre-computed scaler (val/test sets)
            self._scaler = scaler
            self._apply_standardization()
        elif standardize:
            # Compute scaler from this data (training set)
            self._scaler = {
                'mean': np.mean(self.features, axis=0),
                'std': np.std(self.features, axis=0),
            }
            self._apply_standardization()

        print(f"Dataset initialized: {len(self)} admissions")
        print(f"  Positive (deceased): {self.labels.sum()}")
        print(f"  Negative (survived): {(1 - self.labels).sum()}")
        print(f"  Feature dimension: {self.features.shape[1]}")
        if self._scaler is not None:
            print(f"  Standardized: Yes")

    def _apply_standardization(self):
        # Apply z-score normalisation using stored scaler params
        mean = self._scaler['mean']
        std = self._scaler['std']

        # Avoid division by zero for constant features
        std_safe = std.copy()
        std_safe[std_safe < 1e-8] = 1.0

        self.features = (self.features - mean) / std_safe

    def get_scaler(self) -> Optional[dict]:
        # Return the scaler params for use with val/test sets
        return self._scaler

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'hadm_id': self.hadm_ids[idx]
        }