# Script to extract 317 dim segment based feature vectors from waveform recordings
# Run once then use produced parquet file
#
# Produces 6 x 4 hour segment features (temporal dynamics)
# Cross-signal interaction features (correlations)
# Delta features (trajectory from first to last segment)

import pandas as pd
import psycopg
import time
import json
from pathlib import Path

# Waveform source path + DB connection (inlined; formerly waveform_experiments_v2.py).
# NOTE: set the DB password via env/secrets before releasing publicly.
CONSTANTS = {
    "waveform_dir": "../../../Preprocessing/populate_numerics/datamined/waveformfiles",
}

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "mimic_thesis",
    "user": "thesis_user",
    "password": "thesis",
}

from waveform_features_v2 import (process_admission, get_feature_columns, FEATURE_DIM)

WAVEFORM_DIR = Path(CONSTANTS["waveform_dir"])
OUTPUT_PATH = Path("../data/waveform_features_v2.parquet")
PROGRESS_PATH = Path("../data/waveform_extraction_v2_progress.json")

def load_progress() -> dict:
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {"completed": {}, "skipped": [], "failed": []}


def save_progress(progress: dict):
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f)

def main():
    print("=" * 60)
    print("Waveform Feature Extraction — V2 (Segment-Based)")
    print("=" * 60)
    print(f"Waveform directory: {WAVEFORM_DIR}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Feature dimensionality: {FEATURE_DIM}")
    print()

    if not WAVEFORM_DIR.exists():
        print(f"ERROR: Waveform directory not found: {WAVEFORM_DIR}")
        print("Update CONSTANTS['waveform_dir'] at the top of this file")
        return

    # Load admissions for both waveform_teacher and student cohorts
    admissions_query = """
    SELECT
        a.hadm_id,
        a.subject_id,
        a.admittime,
        a.dischtime,
        a.mortality
    FROM admissions a
    JOIN patients p ON a.subject_id = p.subject_id
    WHERE p.cohort IN ('waveform_teacher', 'student')
    AND NOT a.excluded
    """

    print("Loading admissions from database...")
    with psycopg.connect(**DB_CONFIG) as conn:
        admissions_df = pd.read_sql(admissions_query, conn)

    total = len(admissions_df)
    print(f"Total admissions to process: {total}")

    feature_columns = get_feature_columns()

    progress = load_progress()
    completed = progress["completed"]
    skipped = set(progress["skipped"])
    failed = set(progress["failed"])

    already_done = len(completed) + len(skipped) + len(failed)
    if already_done > 0:
        print(f"Resuming: {already_done} already processed "
              f"({len(completed)} retained, {len(skipped)} skipped, {len(failed)} failed)")
    print()

    start_time = time.time()
    new_processed = 0

    for idx, row in admissions_df.iterrows():
        hadm_id = int(row["hadm_id"])
        hadm_str = str(hadm_id)

        if hadm_str in completed or hadm_id in skipped or hadm_id in failed:
            continue

        subject_id = int(row["subject_id"])
        outcome_time = pd.Timestamp(row["dischtime"])

        try:
            features = process_admission(subject_id=subject_id,
                                         outcome_time=outcome_time,
                                         waveform_dir=WAVEFORM_DIR)

            if features is not None:
                feature_dict = {
                    "hadm_id": hadm_id,
                    "mortality": bool(row["mortality"]),
                }
                for i, col_name in enumerate(feature_columns):
                    feature_dict[col_name] = float(features[i])

                completed[hadm_str] = feature_dict
            else:
                skipped.add(hadm_id)

        except Exception as e:
            failed.add(hadm_id)
            print(f"  ERROR hadm_id={hadm_id}: {e}")

        new_processed += 1

        if new_processed % 100 == 0:
            total_done = len(completed) + len(skipped) + len(failed)
            elapsed = time.time() - start_time
            rate = new_processed / elapsed
            remaining = (total - total_done) / rate if rate > 0 else 0

            print(f"  [{total_done}/{total}] "
                  f"{len(completed)} retained, {len(skipped)} skipped, {len(failed)} failed "
                  f"({rate:.1f} adm/sec, ~{remaining / 60:.0f} min remaining)")

            progress["completed"] = completed
            progress["skipped"] = list(skipped)
            progress["failed"] = list(failed)
            save_progress(progress)

    progress["completed"] = completed
    progress["skipped"] = list(skipped)
    progress["failed"] = list(failed)
    save_progress(progress)

    elapsed = time.time() - start_time
    print()
    print("-" * 60)
    print(f"Extraction complete in {elapsed / 60:.1f} minutes")
    print(f"  Retained (≥24h window): {len(completed)}")
    print(f"  Skipped (insufficient data): {len(skipped)}")
    print(f"  Failed (errors): {len(failed)}")

    if completed:
        mort_count = sum(1 for v in completed.values() if v["mortality"])
        print(f"  Mortality rate: {mort_count}/{len(completed)} "
              f"({mort_count / len(completed) * 100:.1f}%)")

    features_df = pd.DataFrame(list(completed.values()))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved features to {OUTPUT_PATH}")
    print(f"Shape: {features_df.shape}")

    print(f"\nProgress file retained at {PROGRESS_PATH} for reference")

if __name__ == "__main__":
    main()