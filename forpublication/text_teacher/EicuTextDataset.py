"""
eICU Text Dataset
=================
The eICU counterpart of `MortalityTextDataset` (TextTeacherDataset.py).

Simpler than the MIMIC version on purpose: eICU has ONE manufactured note per
ICU stay (built by preprocessing/eicu/build_text_notes.py), so there is no
multi-note aggregation and no leakage filter here — both were already applied at
build time. This class just tokenizes.

Input: the DataFrame from
    fetch_data.get_text_teacher_data(split, DataSource.EICU)
with columns [patientunitstayid, mortality, note].

Output dict per item matches the MIMIC text dataset EXACTLY, so the shared
`TextTeacher` model and `TextTeacherTrainer` run unchanged:
    input_ids, attention_mask, label, hadm_id

`hadm_id` is the trainer's generic sample-id key (used to align KD soft labels);
for eICU it carries `patientunitstayid`.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer


class EicuTextDataset(Dataset):

    def __init__(self,
                 df: pd.DataFrame,
                 tokenizer_name: str = "emilyalsentzer/Bio_ClinicalBERT",
                 max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        self.stay_ids = df["patientunitstayid"].values      # -> the 'hadm_id' key
        self.texts = df["note"].values
        self.labels = df["mortality"].values.astype(int)

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
            'label': torch.tensor(int(self.labels[idx]), dtype=torch.long),
            'hadm_id': self.stay_ids[idx],
        }
