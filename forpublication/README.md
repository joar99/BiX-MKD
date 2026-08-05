# forpublication — code to reproduce the paper

Four components. Each folder = the minimal script set to train/run that model.
Copied from the thesis repo (originals under `Models/`); exploration/sweep/superseded
scripts were intentionally left out.

## Components

- **text_teacher/** — BioClinicalBERT teacher. Also the shared **text + SQL/data-access
  layer** (the modules below are reused by the student and reverse-KD).
  - `train_text_teacher.ipynb` — training entry point (there is no `.py` trainer for the text teacher)
  - `experiments.py` (holds `DB_CONFIG` + text configs), `get_sql_data.py`, `filters.py`,
    `trainer.py`, `dataset.py`, `text_encoder.py`
- **waveform_teacher/** — the thesis waveform teacher (`strong_drop4_v2`, [256,128], dropout 0.4).
  - `train_strong_drop4_v2.py` — entry point
  - `extract_waveform_features_v2.py` — regenerates the feature parquet from raw waveforms
  - `waveform_experiments_v2.py`, `waveform_features_v2.py`, `waveform_dataset_v2.py`,
    `waveform_encoder.py`, `waveform_trainer.py`
- **student_forward_kd/** — multimodal student, forward distillation.
  - `run_kd_sweep.py` — entry point; `student_experiments.py` — configs + teacher paths
  - `MultimodalStudent.py`, `MultimodalStudentDataset.py`, `StudentTrainer.py`
  - helpers: `build_thesis_data.py` (data prep), `aggregate_seeds.py` (result aggregation)
- **reverse_kd/** — student → teachers.
  - `run_reverse_kd_sweep.py` — entry point; `reverse_experiments.py` — configs
  - `reverse_trainer.py`

## ⚠️ Cross-folder imports (must rewire after you reorganize)

Scripts currently use flat imports (`from get_sql_data import ...`) resolved at runtime via
`sys.path.append(...)` to sibling dirs. After moving to these folders, those append paths are
stale. The dependency edges to preserve:

- **waveform_teacher** imports `get_sql_data` from **text_teacher**.
- **student_forward_kd** imports from **text_teacher** (`get_sql_data`, `filters`, `trainer`,
  `text_encoder`) AND **waveform_teacher** (`waveform_encoder`, `waveform_features_v2`).
- **reverse_kd** imports from **text_teacher**, **waveform_teacher**, AND **student_forward_kd**
  (`MultimodalStudent`, `MultimodalStudentDataset`).

Fix by updating each entry point's `sys.path.append` targets (or make this a package).

## External inputs NOT copied (needed to actually run)

- **`waveform_features_v2.parquet`** — extracted waveform features (was `Models/WaveformModels/data/`).
  Regenerate with `extract_waveform_features_v2.py` from the raw waveform files, or copy the parquet.
- **Postgres DB `mimic_thesis`** — text notes + cohort splits come from SQL (`get_sql_data.py`,
  `DB_CONFIG` in `experiments.py`).
- **Trained teacher checkpoints** — the student loads frozen teacher `.pt` files
  (`student_experiments.py` → `TEACHER_PATHS`). Produced by the teacher training scripts.

## Planned changes for the journal (not yet applied)

- Forward-KD: switch from a single fixed teacher (seed 99999) to **per-seed matched** teachers
  (align with reverse-KD).
- Forward-KD: **drop the teacher temperature calibration** (`calibrate_teachers()` in `run_kd_sweep.py`).
- Full clean rerun into fresh output dirs (do not overwrite thesis artifacts).
