# Preprocessing pipeline — from raw MIMIC-III to the 4 experiment files

This documents **exactly** how the raw data was turned into the files the
experiments read, so the same procedure can be reproduced (or ported to another
source, e.g. eICU). The original scripts (`populate_*`, waveform fetch/clean)
were deleted after the DB was built; this README + `fed_split.py` +
`waveform_data_preprocessing/` are the surviving record.

## Final outputs (what the experiments actually consume)

Four files, produced by this pipeline:

| file | rows | columns | produced by |
|---|---|---|---|
| `patients.parquet`   | ~10k | `subject_id, cohort, split` | DB → `dump_database.py` |
| `admissions.parquet` | ~16k | `hadm_id, subject_id, admittime, dischtime, mortality, excluded` | DB → `dump_database.py` |
| `notes.parquet`      | ~431k | `note_id, hadm_id, note_time, category, text` | DB → `dump_database.py` |
| `waveform_features_v2.parquet` | ~4.5k | `hadm_id` + 317 feature cols | `waveform_data_preprocessing/extract_waveform_features_v2.py` |

Relationships (preserved as key columns, no DB engine needed):
`notes.hadm_id → admissions.hadm_id → patients.subject_id`.

## Database schema (Postgres `mimic_thesis`)

```
patients    (subject_id INT PRIMARY KEY,
             cohort     TEXT,          -- set by fed_split: text_teacher | waveform_teacher | student | surplus
             split      TEXT)          -- set by fed_split: train | val | test | excluded

admissions  (hadm_id    INT PRIMARY KEY,
             subject_id INT REFERENCES patients,
             admittime  TIMESTAMP,
             dischtime  TIMESTAMP,
             mortality  BOOLEAN,        -- MIMIC HOSPITAL_EXPIRE_FLAG == 1 (in-hospital death)
             excluded   BOOLEAN)        -- set by fed_split (default FALSE)

notes       (note_id    SERIAL PRIMARY KEY,
             hadm_id    INT REFERENCES admissions,
             note_time  TIMESTAMP,      -- CHARTTIME, falling back to CHARTDATE
             category   TEXT,
             text       TEXT)
```

## Raw source data required

- **`RECORDS-numerics`** — the PhysioNet index of the *MIMIC-III Waveform DB
  Matched Subset* (`mimic3wdb-matched/1.0`). Lists numeric records as paths like
  `p00/p000033/p000033-2116-12-24-12-35n`. **This file defines the cohort**: the
  study population is exactly the subjects that have numeric (vital-sign) waveform
  records.
- **MIMIC-III clinical CSVs** (credentialed, from PhysioNet): `PATIENTS.csv`,
  `ADMISSIONS.csv`, `NOTEEVENTS.csv` (and `ICUSTAYS.csv` was used only in ad-hoc
  sanity checks, not the pipeline).
- **Waveform files** downloaded from `physionet.org/files/mimic3wdb-matched/1.0/`
  (the `*n.dat` / `*n.hea` numeric files per subject).

## Pipeline, in order

### 1. `populate_subjects` → seeds the `patients` table
- Parse `subject_id`s out of `RECORDS-numerics` (the waveform index).
- Sanity-check every such subject exists in `PATIENTS.csv`.
- `INSERT INTO patients (subject_id) … ON CONFLICT DO NOTHING`.
- **Cohort root**: population = subjects with waveform numeric records.

### 2. `populate_admissions` → fills `admissions`
- Read `ADMISSIONS.csv`, keep rows whose `SUBJECT_ID` is in `patients`.
- Drop admissions that: lack `ADMITTIME`/`DISCHTIME`; have `DISCHTIME < ADMITTIME`;
  have `HOSPITAL_EXPIRE_FLAG` not in {0,1}.
- Delete any patient left with **zero** eligible admissions.
- `INSERT` `hadm_id, subject_id, admittime, dischtime, mortality`
  where **`mortality = (HOSPITAL_EXPIRE_FLAG == '1')`** (in-hospital death).

### 3. `populate_notes` → fills `notes`
- Read `NOTEEVENTS.csv`, keep notes whose `HADM_ID` is in `admissions`.
- Drop notes with `ISERROR == '1'`.
- Drop admissions with **0 notes**, then drop subjects with 0 remaining
  admissions (cascade delete notes → admissions → patients).
- `INSERT` `hadm_id, note_time, category, text` where
  **`note_time = CHARTTIME or CHARTDATE`** (fallback when charttime missing),
  `category = CATEGORY`, `text = TEXT`.

### 4. Waveform fetch (`get_patient_folders` + `fetch_waveform_data_script`)
- For each `subject_id`, map to the PhysioNet path (`p{zero-padded[:2]}/p{zero-padded}`)
  and scrape its folder listing.
- Download every numeric file (`*n.dat`, `*n.hea`) → `waveformfiles/{subject_id}/`.
- Resumable (batch of 100, `download_progress.json`), polite delay 1s/request.

### 5. `clean_subjects_by_numerics` → drop failed downloads
- A hardcoded `FAILED_SUBJECTS` list (subjects whose download failed) is deleted
  from `notes` → `admissions` → `patients`.

### 6. Waveform feature extraction (`waveform_data_preprocessing/`)
- `extract_waveform_features_v2.py` (uses `waveform_features_v2.py`) reads the raw
  `*n.hea`/`*n.dat`, builds a **317-dim V2 feature vector per admission**, writes
  `waveform_features_v2.parquet`. Feature spec (see `waveform_features_v2.py`):
  4 canonical signals (HR, RESP, SpO2, PULSE), 6×4h segment stats + full-window
  stats + cross-signal interactions + first→last delta features. 24h observation
  window; recordings assigned to the best-overlapping admission.

### 7. `fed_split.py` → cohorts + train/val/test split (the crux)
Deterministic (`RANDOM_STATE = 42`). Reads admissions from the DB and the raw
`.hea` files (for durations), then:
1. Waveform duration per admission (recording→admission by max time-overlap; keep
   only recordings ≥ **`MIN_WAVEFORM_MINUTES = 10`**).
2. Classify subjects: **waveform-eligible** (any admission has waveform coverage)
   vs **text-only**. `mortality` per subject = max over their admissions.
3. **2:1 admission-balanced split** of waveform-eligible subjects into
   `waveform_teacher` (2/3) vs `student` (1/3), via greedy bin-packing on waveform
   admission count, stratified by mortality.
4. **Text-teacher selection**: greedily draw text-only subjects (per mortality
   stratum, shuffled with `RANDOM_STATE`) until their admission count matches the
   waveform teacher's usable-admission count. Leftovers → `surplus` (excluded).
5. **70/15/15 train/val/test** per cohort, subject-level, stratified on mortality,
   with staggered seeds: text_teacher=`42`, waveform_teacher=`43`, student=`44`.
6. **Excluded admissions** = (a) belong to a surplus subject, or (b) belong to a
   waveform/student subject but that specific admission has no waveform coverage.
7. Persist: `UPDATE patients SET cohort, split`; `UPDATE admissions SET excluded`.

Result (usable = not excluded), for reference:

| cohort | train | val | test |
|---|---|---|---|
| text_teacher | 1941 | 416 | 416 |
| waveform_teacher | 2436 | 522 | 523 |
| student | 1349 | 289 | 290 |

### 8. `dump_database.py` → the parquet files
- Auto-discovers every table and dumps to `db_dump/*.parquet`. This replaces the
  live Postgres dependency for all downstream experiments.

## Porting to eICU (or any other source)

Keep the **structure**, swap the **source specifics**:

- **Cohort definition** (step 1): here it's "subjects with matched waveform
  records." eICU has its own vital-sign tables (`vitalPeriodic`/`vitalAperiodic`) —
  define the cohort as patients with usable vitals coverage.
- **Schema**: the 3 tables generalize directly. eICU keys are
  `patientunitstayid` (≈ `hadm_id`) and `uniquepid`/`patienthealthsystemstayid`
  (≈ `subject_id`). Mortality: eICU `unitdischargestatus`/`hospitaldischargestatus`
  = "Expired" (≈ `HOSPITAL_EXPIRE_FLAG`).
- **Notes**: eICU has limited free-text; the text-teacher branch may need a
  different note source or may not port cleanly.
- **Waveforms**: eICU vitals are already tabular time series (no PhysioNet download
  / `.hea` parsing) — steps 4–6 become "pull the vitals table + compute the same
  317-dim feature spec over the observation window."
- **Split** (`fed_split.py`): the logic is source-agnostic — the 2:1 balance,
  text-match, 70/15/15, seeds, and exclusion rules all carry over unchanged.
- **Sanity**: label-leakage filters live at *training* time (drop notes within 6h
  of discharge, exclude discharge summaries) — see the teacher training code, not
  here.

## Notes
- Every original script hardcoded the Postgres creds (`mimic_thesis` / `thesis`).
  Once you're reading the parquet dumps, there is no DB and no password anywhere.
- `fed_split.py` needs the raw `.hea` files to *run*; post-nuke it is documentation
  (the split it produced is already baked into `patients.parquet` / `admissions.parquet`).
