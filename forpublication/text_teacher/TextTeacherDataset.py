import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer

class MortalityTextDataset(Dataset):

    def __init__(self,
                 notes_df: pd.DataFrame,
                 admissions_df: pd.DataFrame,
                 tokenizer_name: str = "emilyalsentzer/Bio_ClinicalBERT",
                 max_length: int = 512,
                 aggregation: str = "first"):

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.aggregation = aggregation

        self.samples = self._aggregate_notes(notes_df, admissions_df)
        # Exposed so callers can read labels without tokenizing every note.
        self.labels = self.samples['mortality'].values.astype(int)

    def _aggregate_notes(self,
                         notes_df: pd.DataFrame,
                         admissions_df: pd.DataFrame) -> pd.DataFrame:

        if self.aggregation == "first":
            agg_notes = (notes_df
                         .sort_values('hours_before_discharge', ascending=False)
                         .groupby('hadm_id')
                         .agg({'text': 'first'})
                         .reset_index())

        elif self.aggregation == "concat":
            agg_notes = (notes_df
                         .sort_values('hours_before_discharge', ascending=False)
                         .groupby('hadm_id')
                         .agg({'text': lambda x: ' [SEP] '.join(x)})
                         .reset_index())

        elif self.aggregation == "last":
            agg_notes = (notes_df
                         .sort_values('hours_before_discharge', ascending=True)
                         .groupby('hadm_id')
                         .agg({'text': 'first'})
                         .reset_index())

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        samples = agg_notes.merge(admissions_df[['hadm_id', 'mortality']],
                                  on='hadm_id',
                                  how='inner')

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        row = self.samples.iloc[idx]

        encoding = self.tokenizer(row['text'],
                                  truncation=True,
                                  padding='max_length',
                                  max_length=self.max_length,
                                  return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(int(row['mortality']), dtype=torch.long),
            'hadm_id': row['hadm_id']
        }