# Multimodal Student Dataset
# ==========================
# PyTorch Dataset that serves paired (text, waveform) inputs for
# the multimodal student model. Each sample contains:
#   - Tokenized clinical note (input_ids, attention_mask)
#   - Pre-computed 317-dim waveform feature vector
#   - Ground-truth mortality label
#   - Admission ID (hadm_id) for tracking
#
# Text handling follows the text teacher pipeline (dataset.py):
#   last-note aggregation, BioClinicalBERT tokenization
#
# Waveform handling follows the waveform teacher pipeline (waveform_dataset_v2.py):
#   pre-computed features from parquet, z-score standardization
#
# Only admissions present in BOTH the notes and waveform features
# are included (inner join on hadm_id).

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from typing import Optional

class MultimodalStudentDataset(Dataset):
    """
    Paired (text, waveform) dataset for the multimodal student cohort.

    Each __getitem__ returns a dict with keys:
        - input_ids:         (seq_len,)     tokenized text
        - attention_mask:    (seq_len,)     attention mask
        - waveform_features: (317,)         standardized feature vector
        - label:             scalar         mortality label (0 or 1)
        - hadm_id:           scalar         admission ID
    """

    def __init__(self,
                 notes_df: pd.DataFrame,
                 admissions_df: pd.DataFrame,
                 waveform_features_df: pd.DataFrame,
                 feature_columns: list[str],
                 tokenizer_name: str = "emilyalsentzer/Bio_ClinicalBERT",
                 max_length: int = 512,
                 aggregation: str = "last",
                 standardize: bool = False,
                 scaler: Optional[dict] = None):
        """
        Parameters
        ----------
        notes_df : pd.DataFrame
            Clinical notes with columns: hadm_id, text, hours_before_discharge.
            Should already be temporally filtered (6h buffer applied).

        admissions_df : pd.DataFrame
            Admissions with columns: hadm_id, mortality (at minimum).

        waveform_features_df : pd.DataFrame
            Pre-computed waveform features with columns: hadm_id, mortality,
            and all feature columns. Typically loaded from parquet.

        feature_columns : list[str]
            Column names for the 317 waveform features.

        tokenizer_name : str
            HuggingFace tokenizer to use for clinical notes.

        max_length : int
            Maximum token length for text truncation/padding.

        aggregation : str
            Note aggregation strategy: 'first', 'last', or 'concat'.
            Default 'last' matches the text teacher's best configuration.

        standardize : bool
            If True, compute z-score stats from this data (use for training set).

        scaler : dict, optional
            Pre-computed {'mean': ndarray, 'std': ndarray} from training set.
            Use for val/test to prevent leakage. Overrides standardize if provided.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.feature_columns = feature_columns

        # ── Step 1: Aggregate notes (same logic as text teacher) ─────
        agg_notes = self._aggregate_notes(notes_df, aggregation)

        # ── Step 2: Join text with admissions to get labels ──────────
        text_samples = agg_notes.merge(
            admissions_df[['hadm_id', 'mortality']],
            on='hadm_id',
            how='inner'
        )

        # ── Step 3: Inner join with waveform features on hadm_id ─────
        # Only keep admissions that have BOTH text and waveform data
        waveform_subset = waveform_features_df[
            ['hadm_id'] + feature_columns
        ].copy()

        paired = text_samples.merge(
            waveform_subset,
            on='hadm_id',
            how='inner'
        )

        # Track what we lost in the join
        text_only = len(text_samples) - len(paired)
        wave_only = len(waveform_features_df) - len(paired)

        # ── Step 4: Store arrays ─────────────────────────────────────
        self.hadm_ids = paired['hadm_id'].values
        self.texts = paired['text'].values
        self.labels = paired['mortality'].values.astype(int)
        self.waveform_features = paired[feature_columns].values.astype(np.float32)

        # ── Step 5: Waveform standardization ─────────────────────────
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

        # ── Report ───────────────────────────────────────────────────
        print(f"MultimodalStudentDataset initialized:")
        print(f"  Paired admissions: {len(self)}")
        print(f"  Positive (deceased): {self.labels.sum()}")
        print(f"  Negative (survived): {(1 - self.labels).sum()}")
        print(f"  Mortality rate: {self.labels.mean():.3f}")
        print(f"  Waveform feature dim: {self.waveform_features.shape[1]}")
        if text_only > 0:
            print(f"  ⚠ Dropped {text_only} admissions (text only, no waveform)")
        if wave_only > 0:
            print(f"  ⚠ Dropped {wave_only} admissions (waveform only, no text)")
        if self._scaler is not None:
            print(f"  Standardized: Yes")

    def _aggregate_notes(self, notes_df: pd.DataFrame, aggregation: str) -> pd.DataFrame:
        """Aggregate multiple notes per admission into a single text."""
        if aggregation == "first":
            agg = (notes_df
                   .sort_values('hours_before_discharge', ascending=False)
                   .groupby('hadm_id')
                   .agg({'text': 'first'})
                   .reset_index())

        elif aggregation == "last":
            agg = (notes_df
                   .sort_values('hours_before_discharge', ascending=True)
                   .groupby('hadm_id')
                   .agg({'text': 'first'})
                   .reset_index())

        elif aggregation == "concat":
            agg = (notes_df
                   .sort_values('hours_before_discharge', ascending=False)
                   .groupby('hadm_id')
                   .agg({'text': lambda x: ' [SEP] '.join(x)})
                   .reset_index())

        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        return agg

    def _apply_standardization(self):
        """Apply z-score normalization using stored scaler parameters."""
        mean = self._scaler['mean']
        std = self._scaler['std']

        # Avoid division by zero for constant features
        std_safe = std.copy()
        std_safe[std_safe < 1e-8] = 1.0

        self.waveform_features = (self.waveform_features - mean) / std_safe

    def get_scaler(self) -> Optional[dict]:
        """Return scaler parameters for use with val/test sets."""
        return self._scaler

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        # Tokenize the clinical note
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'waveform_features': torch.tensor(
                self.waveform_features[idx], dtype=torch.float32
            ),
            'label': torch.tensor(int(self.labels[idx]), dtype=torch.long),
            'hadm_id': self.hadm_ids[idx]
        }