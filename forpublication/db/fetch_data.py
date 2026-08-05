"""
Central data access — MIMIC-III and eICU.
==========================================
All parquet / cohort / split / filtering logic lives here. Callers pick a
model's method, a `Split`, AND a `DataSource` — nothing else touches the parquet
files, and there is no database or password anywhere.

Both `split` and `source` are REQUIRED enums — passing nothing (or a raw string)
raises immediately. No fallbacks, no silent defaults.

MIMIC-III (db/mimic_data/):
    patients.parquet             subject_id, cohort, split
    admissions.parquet           hadm_id, subject_id, admittime, dischtime, mortality, excluded
    notes.parquet                note_id, hadm_id, note_time, category, text
    waveform_features_v2.parquet hadm_id, mortality, <317 feature columns>
  The note leakage-filter (drop notes within 6h of discharge + discharge
  summaries) is applied inside the text-bearing methods, so no call site can
  forget it. Waveform features are already windowed, so they are not filtered.

eICU-CRD v2.0 (db/e-icu-data/) — built by preprocessing/eicu/*.py:
    eicu_fed_split.parquet       patientunitstayid, uniquepid, hospitalid, mortality, role, split
    eicu_text_notes.parquet      patientunitstayid, note, ...     (empties already dropped)
    eicu_vitals_features.parquet patientunitstayid, <236 feature columns>
  Cohort (role) + train/val/test come straight from eicu_fed_split.parquet.
  The leakage filter, the paired-only cut and the manufactured text are all
  already applied by the build scripts, so these methods are thin joins.

The two sources are DELIBERATELY self-contained — MIMIC and eICU differ in nearly
every column, key, label and filter — so each has its own internals. The public
method is a thin gate: `DataSource` just routes the call to the right one.
"""

import sys
from enum import Enum
from pathlib import Path
from typing import NamedTuple

import pandas as pd

# Note leakage-filter thresholds come from the single source of truth.
_FORPUB = Path(__file__).resolve().parents[1]
if str(_FORPUB / "config") not in sys.path:
    sys.path.insert(0, str(_FORPUB / "config"))
from experiment_config import MIN_HOURS_BEFORE_DISCHARGE, EXCLUDE_NOTE_CATEGORIES  # noqa: E402

MIMIC_DIR = Path(__file__).resolve().parent / "mimic_data"
EICU_DIR = Path(__file__).resolve().parent / "e-icu-data"


class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DataSource(Enum):
    MIMIC = "mimic"
    EICU = "eicu"


class TextData(NamedTuple):
    notes: pd.DataFrame
    admissions: pd.DataFrame


class StudentData(NamedTuple):
    notes: pd.DataFrame
    admissions: pd.DataFrame
    features: pd.DataFrame


def _require_split(split) -> Split:
    """Loud failure if split is missing or not a Split enum."""
    if not isinstance(split, Split):
        raise TypeError(
            "split must be a Split enum (Split.TRAIN / Split.VAL / Split.TEST); "
            f"got {type(split).__name__}: {split!r}"
        )
    return split


def _require_source(source) -> DataSource:
    """Loud failure if sourcdce is missing or not a DataSource enum."""
    if not isinstance(source, DataSource):
        raise TypeError(
            "source must be a DataSource enum (DataSource.MIMIC / DataSource.EICU); "
            f"got {type(source).__name__}: {source!r}"
        )
    return source


# ═══════════════════════════════════════════════════════════════════════════
# MIMIC-III internals   (byte-for-byte the thesis pipeline)
# ═══════════════════════════════════════════════════════════════════════════
_cache: dict[str, pd.DataFrame] = {}


def _load(name: str) -> pd.DataFrame:
    """Read (and cache) a parquet table from db/mimic_data/."""
    if name not in _cache:
        path = MIMIC_DIR / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing data file: {path}")
        _cache[name] = pd.read_parquet(path)
    return _cache[name]


def _cohort_split_hadms(cohort: str, split: Split) -> set:
    """Usable (non-excluded) hadm_ids for one cohort's split."""
    patients = _load("patients")
    admissions = _load("admissions")
    subjects = set(
        patients[(patients["cohort"] == cohort) & (patients["split"] == split.value)]["subject_id"]
    )
    adm = admissions[admissions["subject_id"].isin(subjects) & (~admissions["excluded"])]
    return set(adm["hadm_id"])


def _admissions_for(hadms: set) -> pd.DataFrame:
    admissions = _load("admissions")
    return admissions[admissions["hadm_id"].isin(hadms)].reset_index(drop=True)


def _filtered_notes(hadms: set) -> pd.DataFrame:
    """Notes for the given admissions, leakage-filtered (thesis rule)."""
    notes = _load("notes")
    admissions = _load("admissions")

    notes = notes[notes["hadm_id"].isin(hadms)].copy()
    notes = notes.merge(admissions[["hadm_id", "dischtime"]], on="hadm_id", how="left")
    notes["hours_before_discharge"] = (
        (notes["dischtime"] - notes["note_time"]).dt.total_seconds() / 3600.0
    )
    notes = notes.drop(columns=["dischtime"])

    keep = (
        (notes["hours_before_discharge"] >= MIN_HOURS_BEFORE_DISCHARGE)
        & (~notes["category"].isin(EXCLUDE_NOTE_CATEGORIES))
    )
    return notes[keep].reset_index(drop=True)


def _features_for(hadms: set) -> pd.DataFrame:
    """Waveform feature rows (already carry `mortality`) for the given admissions."""
    features = _load("waveform_features_v2")
    return features[features["hadm_id"].isin(hadms)].reset_index(drop=True)


def _mimic_text(split: Split) -> TextData:
    hadms = _cohort_split_hadms("text_teacher", split)
    return TextData(notes=_filtered_notes(hadms), admissions=_admissions_for(hadms))


def _mimic_waveform(split: Split) -> pd.DataFrame:
    hadms = _cohort_split_hadms("waveform_teacher", split)
    return _features_for(hadms)


def _mimic_student(split: Split) -> StudentData:
    """Student cohort restricted to PAIRED admissions — those with BOTH a usable
    (leakage-filtered) note AND a waveform feature vector. The restriction is
    bolted in HERE so student training, forward KD and reverse KD all see the
    exact same paired set; text-only / waveform-only admissions are never returned.
    """
    hadms    = _cohort_split_hadms("student", split)
    notes    = _filtered_notes(hadms)
    features = _features_for(hadms)

    paired = set(notes["hadm_id"]) & set(features["hadm_id"])

    return StudentData(
        notes=notes[notes["hadm_id"].isin(paired)].reset_index(drop=True),
        admissions=_admissions_for(paired),
        features=features[features["hadm_id"].isin(paired)].reset_index(drop=True),
    )


# ═══════════════════════════════════════════════════════════════════════════
# eICU-CRD internals   (cohort/split/leakage/pairing/text already in the parquets)
# ═══════════════════════════════════════════════════════════════════════════
_eicu_cache: dict[str, pd.DataFrame] = {}


def _load_eicu(name: str) -> pd.DataFrame:
    """Read (and cache) a parquet table from db/e-icu-data/."""
    if name not in _eicu_cache:
        path = EICU_DIR / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing eICU file: {path} — run preprocessing/eicu/*.py first"
            )
        _eicu_cache[name] = pd.read_parquet(path)
    return _eicu_cache[name]


def _eicu_cohort(role: str, split: Split) -> pd.DataFrame:
    """The (patientunitstayid, mortality) rows for one role's split, straight
    from the federated-split manifest. Split.TRAIN -> the 'TRAIN' rows, etc."""
    fs = _load_eicu("eicu_fed_split")
    sel = fs[(fs["role"] == role) & (fs["split"] == split.name)]
    return sel[["patientunitstayid", "mortality"]].reset_index(drop=True)


def _eicu_text(split: Split) -> pd.DataFrame:
    """Text-teacher cohort: one assembled note per stay + label."""
    cohort = _eicu_cohort("text_teacher", split)
    notes = _load_eicu("eicu_text_notes")[["patientunitstayid", "note"]]
    out = cohort.merge(notes, on="patientunitstayid", how="left")
    if out["note"].isna().any():
        raise ValueError(
            f"{out['note'].isna().sum()} eICU text stays have no note — "
            "fed_split and eicu_text_notes are out of sync"
        )
    return out


def _eicu_waveform(split: Split) -> pd.DataFrame:
    """Waveform (vitals) teacher cohort: 236 feature columns + label."""
    cohort = _eicu_cohort("waveform_teacher", split)
    feats = _load_eicu("eicu_vitals_features")
    out = cohort.merge(feats, on="patientunitstayid", how="left")
    _feat_cols = [c for c in feats.columns if c != "patientunitstayid"]
    if out[_feat_cols].isna().any().any():
        raise ValueError(
            "eICU waveform cohort has stays with no features — "
            "fed_split and eicu_vitals_features are out of sync"
        )
    return out


def _eicu_student(split: Split) -> pd.DataFrame:
    """Student cohort: note + 236 features + label, one row per stay. fed_split is
    already paired-only, so the join is complete by construction — but we assert
    it, since a silent unpaired row would corrupt the multimodal training."""
    cohort = _eicu_cohort("student", split)
    notes = _load_eicu("eicu_text_notes")[["patientunitstayid", "note"]]
    feats = _load_eicu("eicu_vitals_features")
    out = (cohort.merge(notes, on="patientunitstayid", how="left")
                 .merge(feats, on="patientunitstayid", how="left"))
    _feat_cols = [c for c in feats.columns if c != "patientunitstayid"]
    if out["note"].isna().any() or out[_feat_cols].isna().any().any():
        raise ValueError(
            "eICU student cohort is not fully paired — fed_split and the built "
            "parquets are out of sync"
        )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Public API — one method per model; Split AND DataSource both required.
# The method is a thin gate that dispatches to the source's self-contained code.
# ═══════════════════════════════════════════════════════════════════════════
def get_text_teacher_data(split: Split, source: DataSource):
    """Text-teacher cohort, one split. Return shape depends on source:
    MIMIC -> TextData(notes, admissions);  eICU -> DataFrame[patientunitstayid, mortality, note]."""
    _require_split(split)
    _require_source(source)
    if source is DataSource.MIMIC:
        return _mimic_text(split)
    if source is DataSource.EICU:
        return _eicu_text(split)
    raise TypeError(f"unhandled DataSource: {source!r}")


def get_waveform_teacher_data(split: Split, source: DataSource):
    """Waveform/vitals-teacher cohort, one split. MIMIC -> features DataFrame with
    `mortality` (+317 cols); eICU -> DataFrame[patientunitstayid, mortality, +236 cols]."""
    _require_split(split)
    _require_source(source)
    if source is DataSource.MIMIC:
        return _mimic_waveform(split)
    if source is DataSource.EICU:
        return _eicu_waveform(split)
    raise TypeError(f"unhandled DataSource: {source!r}")


def get_multimodal_student_data(split: Split, source: DataSource):
    """Student cohort (PAIRED only), one split. MIMIC -> StudentData(notes,
    admissions, features);  eICU -> DataFrame[patientunitstayid, mortality, note, +236 cols]."""
    _require_split(split)
    _require_source(source)
    if source is DataSource.MIMIC:
        return _mimic_student(split)
    if source is DataSource.EICU:
        return _eicu_student(split)
    raise TypeError(f"unhandled DataSource: {source!r}")
