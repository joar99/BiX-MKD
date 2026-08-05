# This is the script that splits the cohorts into 40/40/20

import psycopg
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "mimic_thesis",
    "user": "thesis_user",
    "password": "thesis",
}

WAVEFORM_ROOT = Path("populate_numerics/datamined/waveformfiles")
MIN_WAVEFORM_MINUTES = 10

RANDOM_STATE = 42

def get_connection():
    return psycopg.connect(**DB_CONFIG)


def fetch_dataframe(query):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]
    cur.close()
    conn.close()
    return pd.DataFrame(rows, columns=cols)


def parse_waveform_duration_minutes(hea_path: Path) -> float | None:
    """
    Reads .hea file to estimate duration:
    duration = num_samples / sampling_rate
    """
    try:
        with open(hea_path, "r") as f:
            first_line = f.readline().strip()
            parts = first_line.split()
            if len(parts) < 4:
                return None
            num_samples = int(parts[3])
            sampling_rate = float(parts[2])
            return num_samples / sampling_rate / 60.0
    except Exception:
        return None


# ============================================================================
# STEP 1: COMPUTE WAVEFORM DURATION PER ADMISSION
# Reuses the overlap logic from the validation script to assign waveform
# recordings to admissions, but accumulates total duration per admission
# rather than just counting recordings.
# ============================================================================

def compute_waveform_duration_per_admission(admissions_df) -> dict[int, float]:
    """
    Returns: {hadm_id: total_waveform_minutes} for all admissions that have
    at least one waveform recording meeting MIN_WAVEFORM_MINUTES.
    """
    import re
    from datetime import datetime, timedelta

    # Build subject -> admissions lookup
    admissions_by_subject = defaultdict(list)
    for _, row in admissions_df.iterrows():
        admissions_by_subject[row["subject_id"]].append({
            "hadm_id": row["hadm_id"],
            "admittime": row["admittime"],
            "dischtime": row["dischtime"],
        })

    admission_durations = defaultdict(float)  # hadm_id -> total minutes

    for subject_dir in sorted(WAVEFORM_ROOT.iterdir()):
        if not subject_dir.is_dir():
            continue
        try:
            subject_id = int(subject_dir.name)
        except ValueError:
            continue

        admissions = admissions_by_subject.get(subject_id, [])
        if not admissions:
            continue

        for hea_file in subject_dir.glob("*.hea"):
            # Parse start time from filename (e.g. p000052-2191-01-10-12-15n.hea)
            match = re.search(r"-(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})", hea_file.name)
            if not match:
                continue

            year, month, day, hour, minute = map(int, match.groups())
            start_time = datetime(year, month, day, hour, minute)

            duration_min = parse_waveform_duration_minutes(hea_file)
            if not duration_min or duration_min < MIN_WAVEFORM_MINUTES:
                continue

            end_time = start_time + timedelta(minutes=duration_min)

            # Find best-overlapping admission
            best_match = None
            best_overlap = 0.0

            for adm in admissions:
                latest_start = max(start_time, adm["admittime"])
                earliest_end = min(end_time, adm["dischtime"])
                overlap = max((earliest_end - latest_start).total_seconds() / 60.0, 0.0)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = adm

            if best_match and best_overlap > 0:
                admission_durations[best_match["hadm_id"]] += best_overlap

    return dict(admission_durations)


# ============================================================================
# STEP 2: CLASSIFY SUBJECTS INTO WAVEFORM-ELIGIBLE AND TEXT-ONLY POOLS
# A subject is waveform-eligible if ANY of their admissions have waveform
# coverage. Their waveform_admission_count is the number of admissions with
# waveform coverage — this is what Step 3 balances on.
# ============================================================================

def classify_subjects(admissions_df, waveform_durations) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        waveform_subjects_df: one row per waveform-eligible subject with columns
            [subject_id, mortality, waveform_admission_count]
        text_only_subjects_df: one row per text-only subject with columns
            [subject_id, mortality]

    mortality is aggregated as max across admissions (same as original fed_split).
    """
    admissions_df = admissions_df.copy()
    admissions_df["has_waveform"] = admissions_df["hadm_id"].isin(waveform_durations)

    # Aggregate at subject level
    subject_agg = (
        admissions_df.groupby("subject_id")
        .agg(
            mortality=("mortality", "max"),
            waveform_admission_count=("has_waveform", "sum"),
            any_waveform=("has_waveform", "any"),
        )
        .reset_index()
    )

    waveform_subjects = (
        subject_agg[subject_agg["any_waveform"]]
        .drop(columns=["any_waveform"])
        .reset_index(drop=True)
    )
    text_only_subjects = (
        subject_agg[~subject_agg["any_waveform"]][["subject_id", "mortality"]]
        .reset_index(drop=True)
    )

    return waveform_subjects, text_only_subjects


# ============================================================================
# STEP 3: ADMISSION-COUNT-BALANCED 2:1 SPLIT OF WAVEFORM-ELIGIBLE SUBJECTS
# Splits into waveform_teacher (2/3) and student (1/3) by waveform admission
# count, stratified on mortality. Uses greedy bin-packing within each
# mortality stratum. This ensures the 40/40/20 ratio holds at the usable
# admission level after exclusions, since both cohorts lose admissions to
# exclusion at similar rates (~38%).
# ============================================================================

def admission_balanced_split(waveform_subjects_df) -> tuple[list, list]:
    """
    Returns:
        wf_teacher_subjects: list of subject_ids assigned to waveform teacher
        student_subjects: list of subject_ids assigned to student

    The split targets a 2:1 ratio of waveform admission count (not duration).
    Stratification on mortality is enforced by splitting within each stratum
    independently.
    """
    wf_teacher_subjects = []
    student_subjects = []

    for mortality_val in [0, 1]:
        stratum = waveform_subjects_df[
            waveform_subjects_df["mortality"] == mortality_val
        ].sort_values("waveform_admission_count", ascending=False).reset_index(drop=True)

        total_admissions = stratum["waveform_admission_count"].sum()
        wf_target = total_admissions * (2 / 3)
        student_target = total_admissions - wf_target

        wf_accumulated = 0
        student_accumulated = 0
        wf_stratum = []
        student_stratum = []

        # Greedy bin-packing: assign each subject (most admissions first) to
        # whichever cohort is currently furthest below its target
        for _, row in stratum.iterrows():
            subj_admissions = int(row["waveform_admission_count"])

            wf_deficit = wf_target - wf_accumulated
            student_deficit = student_target - student_accumulated

            if wf_deficit >= student_deficit:
                wf_stratum.append(int(row["subject_id"]))
                wf_accumulated += subj_admissions
            else:
                student_stratum.append(int(row["subject_id"]))
                student_accumulated += subj_admissions

        wf_teacher_subjects.extend(wf_stratum)
        student_subjects.extend(student_stratum)

        print(f"  Mortality={mortality_val}: wf_teacher={len(wf_stratum)} subjects "
              f"({wf_accumulated} waveform admissions), student={len(student_stratum)} subjects "
              f"({student_accumulated} waveform admissions), "
              f"achieved ratio={wf_accumulated/total_admissions*100:.1f}% / "
              f"{student_accumulated/total_admissions*100:.1f}%")

    return wf_teacher_subjects, student_subjects


# ============================================================================
# STEP 4: SELECT TEXT TEACHER SUBJECTS FROM TEXT-ONLY POOL
# Draws text-only subjects until their cumulative usable admission count
# matches the waveform teacher's usable admission count, stratified on
# mortality. This ensures the 40/40/20 ratio holds at the admission level,
# not just the subject level. Remaining text-only subjects are surplus.
# ============================================================================

def compute_wf_teacher_usable_admissions(
    admissions_df, wf_teacher_subjects: list, waveform_durations: dict
) -> dict[int, int]:
    """
    Returns: {mortality_label: usable_admission_count} for the waveform teacher.
    A wf_teacher admission is usable iff it appears in waveform_durations.
    """
    wf_set = set(wf_teacher_subjects)
    counts = {0: 0, 1: 0}

    for _, row in admissions_df.iterrows():
        if row["subject_id"] in wf_set and row["hadm_id"] in waveform_durations:
            counts[int(row["mortality"])] += 1

    return counts


def select_text_teacher_subjects(
    text_only_subjects_df, admissions_df, wf_teacher_subjects, waveform_durations
) -> tuple[list, list]:
    """
    Greedily selects text-only subjects (stratified by mortality) until their
    cumulative admission count matches the waveform teacher's usable admission
    count per stratum.

    Returns:
        text_teacher_subjects: list of subject_ids selected for text teacher
        surplus_subjects: list of subject_ids not selected (admissions will be excluded)
    """
    # Target: match wf_teacher usable admissions per mortality stratum
    wf_targets = compute_wf_teacher_usable_admissions(
        admissions_df, wf_teacher_subjects, waveform_durations
    )
    print(f"  Wf teacher usable admissions by mortality: {wf_targets}")

    # Build per-subject admission counts for text-only subjects
    text_subject_set = set(text_only_subjects_df["subject_id"].values)
    admission_counts = (
        admissions_df[admissions_df["subject_id"].isin(text_subject_set)]
        .groupby("subject_id")
        .size()
        .to_dict()
    )  # {subject_id: admission_count}

    text_teacher_subjects = []
    surplus_subjects = []

    # Greedy selection within each mortality stratum independently
    for mortality_val in [0, 1]:
        stratum = text_only_subjects_df[
            text_only_subjects_df["mortality"] == mortality_val
        ].copy()

        # Shuffle deterministically so greedy selection isn't biased by DB ordering
        stratum = stratum.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        target_admissions = wf_targets[mortality_val]
        accumulated = 0
        selected = []
        not_selected = []

        for _, row in stratum.iterrows():
            subj_id = int(row["subject_id"])
            subj_admissions = admission_counts.get(subj_id, 0)

            if accumulated < target_admissions:
                selected.append(subj_id)
                accumulated += subj_admissions
            else:
                not_selected.append(subj_id)

        text_teacher_subjects.extend(selected)
        surplus_subjects.extend(not_selected)

        print(f"  Mortality={mortality_val}: selected {len(selected)} subjects "
              f"({accumulated} admissions, target was {target_admissions})")

    return text_teacher_subjects, surplus_subjects


# ============================================================================
# STEP 5: TRAIN/VAL/TEST SPLIT WITHIN EACH COHORT
# 70/15/15 subject-level split, stratified on mortality.
# Uses staggered random states per cohort (same as original fed_split).
# ============================================================================

def split_train_val_test(subjects, labels, random_state) -> tuple[list, list, list]:
    """
    Returns: (train_subjects, val_subjects, test_subjects)
    """
    train_subj, temp_subj, train_lab, temp_lab = train_test_split(
        subjects, labels,
        test_size=0.30,
        stratify=labels,
        random_state=random_state,
    )

    val_subj, test_subj, _, _ = train_test_split(
        temp_subj, temp_lab,
        test_size=0.50,
        stratify=temp_lab,  # Fixed: stratify the val/test split too
        random_state=random_state,
    )

    return list(train_subj), list(val_subj), list(test_subj)


# ============================================================================
# STEP 6: DETERMINE EXCLUDED ADMISSIONS
# An admission is excluded if:
#   (a) it belongs to a surplus text-only subject, OR
#   (b) it belongs to a waveform_teacher or student subject but has no
#       waveform coverage itself
# ============================================================================

def compute_excluded_admissions(
    admissions_df,
    surplus_subjects: list,
    wf_teacher_subjects: list,
    student_subjects: list,
    waveform_durations: dict,
) -> set[int]:
    """
    Returns: set of hadm_ids that should be marked excluded=true
    """
    excluded = set()

    surplus_set = set(surplus_subjects)
    waveform_cohort_set = set(wf_teacher_subjects) | set(student_subjects)

    for _, row in admissions_df.iterrows():
        hadm_id = row["hadm_id"]
        subject_id = row["subject_id"]

        # (a) Surplus text-only subject
        if subject_id in surplus_set:
            excluded.add(hadm_id)
            continue

        # (b) Waveform/student cohort but this specific admission has no waveform
        if subject_id in waveform_cohort_set and hadm_id not in waveform_durations:
            excluded.add(hadm_id)

    return excluded


# ============================================================================
# STEP 7: PERSIST TO DATABASE
# ============================================================================

def persist_assignments(
    cohort_assignments: dict[int, tuple[str, str]],  # subject_id -> (cohort, split)
    excluded_admissions: set[int],
):
    """
    Updates patients.cohort and patients.split, and admissions.excluded.
    """
    conn = get_connection()
    cur = conn.cursor()

    # Update patient cohort/split assignments
    patient_updates = [(cohort, split, subj_id) for subj_id, (cohort, split) in cohort_assignments.items()]
    cur.executemany(
        "UPDATE patients SET cohort = %s, split = %s WHERE subject_id = %s",
        patient_updates,
    )

    # Reset all admissions to excluded=false first, then mark the excluded ones
    cur.execute("UPDATE admissions SET excluded = FALSE")
    if excluded_admissions:
        cur.executemany(
            "UPDATE admissions SET excluded = TRUE WHERE hadm_id = %s",
            [(hadm_id,) for hadm_id in excluded_admissions],
        )

    conn.commit()
    cur.close()
    conn.close()
    print(f"Persisted {len(patient_updates)} patient assignments, {len(excluded_admissions)} excluded admissions")


# ============================================================================
# STEP 8: VERIFICATION
# ============================================================================

def verify():
    print("\n" + "=" * 70)
    print("COHORT SUMMARY (subjects)")
    print("=" * 70)

    summary = fetch_dataframe("""
        SELECT
            p.cohort,
            p.split,
            COUNT(DISTINCT p.subject_id) AS subjects,
            COUNT(DISTINCT a.hadm_id) AS total_admissions,
            COUNT(DISTINCT CASE WHEN NOT a.excluded THEN a.hadm_id END) AS usable_admissions,
            ROUND(100.0 * SUM(CASE WHEN NOT a.excluded THEN a.mortality::int ELSE 0 END)
                        / NULLIF(COUNT(CASE WHEN NOT a.excluded THEN 1 END), 0), 1) AS mortality_pct
        FROM patients p
        JOIN admissions a ON p.subject_id = a.subject_id
        GROUP BY p.cohort, p.split
        ORDER BY p.cohort, p.split
    """)
    print(summary.to_string(index=False))

    print("\n" + "=" * 70)
    print("COHORT TOTALS (usable admissions only)")
    print("=" * 70)

    totals = fetch_dataframe("""
        SELECT
            p.cohort,
            COUNT(DISTINCT p.subject_id) AS subjects,
            COUNT(DISTINCT CASE WHEN NOT a.excluded THEN a.hadm_id END) AS usable_admissions,
            ROUND(100.0 * COUNT(DISTINCT CASE WHEN NOT a.excluded THEN a.hadm_id END)
                        / NULLIF(SUM(COUNT(DISTINCT CASE WHEN NOT a.excluded THEN a.hadm_id END)) OVER (), 0), 1) AS pct_of_total
        FROM patients p
        JOIN admissions a ON p.subject_id = a.subject_id
        GROUP BY p.cohort
        ORDER BY p.cohort
    """)
    print(totals.to_string(index=False))

    print("\n" + "=" * 70)
    print("EXCLUSION SUMMARY")
    print("=" * 70)

    exclusion_summary = fetch_dataframe("""
        SELECT
            COUNT(*) AS total_admissions,
            COUNT(CASE WHEN excluded THEN 1 END) AS excluded,
            COUNT(CASE WHEN NOT excluded THEN 1 END) AS usable
        FROM admissions
    """)
    print(exclusion_summary.to_string(index=False))

    # Leakage checks
    print("\n" + "=" * 70)
    print("LEAKAGE CHECKS")
    print("=" * 70)

    cross_cohort = fetch_dataframe("""
        SELECT COUNT(*) AS subjects_in_multiple_cohorts
        FROM patients
        GROUP BY subject_id
        HAVING COUNT(DISTINCT cohort) > 1
    """)
    print(f"  Subjects in multiple cohorts: {cross_cohort['subjects_in_multiple_cohorts'].sum()}")

    unassigned = fetch_dataframe("""
        SELECT COUNT(*) AS unassigned
        FROM patients
        WHERE cohort IS NULL OR split IS NULL
    """)
    print(f"  Unassigned patients:          {unassigned['unassigned'].iloc[0]}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Loading admissions from database...")
    admissions_df = fetch_dataframe(
        "SELECT hadm_id, subject_id, admittime, dischtime, mortality FROM admissions"
    )
    print(f"  Loaded {len(admissions_df)} admissions for {admissions_df['subject_id'].nunique()} subjects\n")

    # --- Step 1: Waveform duration per admission ---
    print("Computing waveform duration per admission...")
    waveform_durations = compute_waveform_duration_per_admission(admissions_df)
    print(f"  {len(waveform_durations)} admissions have waveform coverage\n")

    # --- Step 2: Classify subjects ---
    print("Classifying subjects into waveform-eligible and text-only pools...")
    waveform_subjects_df, text_only_subjects_df = classify_subjects(admissions_df, waveform_durations)
    print(f"  Waveform-eligible subjects: {len(waveform_subjects_df)}")
    print(f"  Text-only subjects:         {len(text_only_subjects_df)}\n")

    # --- Step 3: Admission-count-balanced 2:1 split of waveform subjects ---
    print("Admission-count-balanced split of waveform-eligible subjects (2:1 wf_teacher:student)...")
    wf_teacher_subjects, student_subjects = admission_balanced_split(waveform_subjects_df)
    print(f"  Waveform teacher: {len(wf_teacher_subjects)} subjects")
    print(f"  Student:          {len(student_subjects)} subjects\n")

    # --- Step 4: Text teacher draws from text-only pool to match wf_teacher usable admissions ---
    print("Selecting text teacher subjects from text-only pool (admission-count matched)...")
    text_teacher_subjects, surplus_subjects = select_text_teacher_subjects(
        text_only_subjects_df, admissions_df, wf_teacher_subjects, waveform_durations
    )
    print(f"  Text teacher: {len(text_teacher_subjects)} subjects")
    print(f"  Surplus (will be excluded): {len(surplus_subjects)} subjects\n")

    # --- Step 5: Train/val/test splits within each cohort ---
    print("Splitting each cohort into train/val/test (70/15/15)...")

    # Build label lookup for each cohort
    wf_subjects_set = set(wf_teacher_subjects)
    student_subjects_set = set(student_subjects)
    text_subjects_set = set(text_teacher_subjects)

    # Get mortality labels per subject (max across admissions)
    subject_mortality = (
        admissions_df.groupby("subject_id")["mortality"].max().to_dict()
    )

    cohort_assignments = {}  # subject_id -> (cohort, split)

    cohorts = [
        ("text_teacher",     text_teacher_subjects,  RANDOM_STATE),
        ("waveform_teacher", wf_teacher_subjects,    RANDOM_STATE + 1),
        ("student",          student_subjects,       RANDOM_STATE + 2),
    ]

    for cohort_name, subjects, rs in cohorts:
        subj_array = np.array(subjects)
        label_array = np.array([subject_mortality[s] for s in subjects])

        train, val, test = split_train_val_test(subj_array, label_array, rs)

        for subj in train:
            cohort_assignments[subj] = (cohort_name, "train")
        for subj in val:
            cohort_assignments[subj] = (cohort_name, "val")
        for subj in test:
            cohort_assignments[subj] = (cohort_name, "test")

        print(f"  {cohort_name}: train={len(train)}, val={len(val)}, test={len(test)}")

    # Surplus subjects still need a cohort/split entry so the DB is consistent
    # Mark them with cohort='surplus' so they're clearly identifiable
    for subj in surplus_subjects:
        cohort_assignments[subj] = ("surplus", "excluded")

    print()

    # --- Step 6: Compute excluded admissions ---
    print("Computing excluded admissions...")
    excluded_admissions = compute_excluded_admissions(
        admissions_df, surplus_subjects, wf_teacher_subjects, student_subjects, waveform_durations
    )
    print(f"  {len(excluded_admissions)} admissions marked excluded\n")

    # --- Step 7: Persist ---
    print("Persisting to database...")
    persist_assignments(cohort_assignments, excluded_admissions)

    # --- Step 8: Verify ---
    verify()


if __name__ == "__main__":
    main()