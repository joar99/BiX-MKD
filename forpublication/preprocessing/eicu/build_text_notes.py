"""
Build the eICU text-teacher corpus  (eICU-CRD v2.0 -> parquet).
==============================================================

eICU contains NO clinical prose (`note.notetext` maxes out at 101 characters and
the narrative sections were removed during de-identification). So the text the
teacher reads is *manufactured* from four structured tables.

Ratified design choices (see db/eicu-data-plan.md):
  1. `pastHistory` IS included — comorbidity burden predicts mortality, and it
     only costs ~64 chars. It repeats across a patient's stays; that is fine
     because the cohort split keeps a patient inside one cohort.
  2. FULL taxonomy path, pipes converted to commas. Leaf-only is unsafe:
     "cardiovascular|chest pain / ASHD|coronary artery disease|known" has the
     leaf "known", which is meaningless. Commas are in BioClinicalBERT's
     pretraining distribution; "|" is not.
  3. DEDUPE within a stay. diagnosis is 74% repeat rows, treatment 70%.
     Repetition tells BERT nothing and burns the 512-token budget.
  4. SECTION HEADERS, because real clinical notes have them, and because
     "Past history: dementia" != "Active problems: dementia".
  5. ORDER = admission reason -> active problems -> treatments -> past history.
     Ordering controls TRUNCATION, not emphasis (BERT keeps the first 512
     tokens). Past history goes last because it is the most generic and the
     most repeated, so it is the right thing to lose first.
  6. Empty stays are KEPT, with an empty note. Excluding them is a cohort
     decision and belongs in the fetch layer, not here.

Leakage filter: an item is dropped if it was recorded within
LEAKAGE_BUFFER_MINUTES of the patient leaving the ICU. Anchored on
`unitdischargeoffset` because the label is now `unitdischargestatus` (ICU
death). Applied at BUILD time; rebuilding is cheap, so change the constant and
re-run rather than post-filtering.

Outputs (into db/e-icu-data/):
  eicu_text_notes.parquet   one row per ICU stay: the assembled note + QA columns
  eicu_patients.parquet     the 5 agreed `patient` columns

Run:  python forpublication/preprocessing/eicu/build_text_notes.py
"""

from pathlib import Path

import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG — every rule that shapes the corpus is visible here
# ═══════════════════════════════════════════════════════════════════════════
_FORPUB = Path(__file__).resolve().parents[2]
EICU_DIR = _FORPUB / "db" / "e-icu-data" / "physionet.org" / "files" / "eicu-crd" / "2.0"
OUT_DIR = _FORPUB / "db" / "e-icu-data"

NOTES_OUT = OUT_DIR / "eicu_text_notes.parquet"
PATIENTS_OUT = OUT_DIR / "eicu_patients.parquet"

# Drop anything recorded within this many minutes of ICU discharge.
# MIMIC used 6 hours before discharge; same buffer, ICU anchor.
LEAKAGE_BUFFER_MINUTES = 360

# BioClinicalBERT's cap. Only used for QA reporting here — the tokenizer does
# the actual truncation at training time.
BERT_MAX_TOKENS = 512

# Section headers, in the order they are written into the note.
SECTIONS = [
    ("admissionDx", "Reason for admission"),
    ("diagnosis",   "Active problems"),
    ("treatment",   "Treatments"),
    ("pastHistory", "Past history"),
]

# `admissionDx` mixes real diagnoses with administrative yes/no answers.
# Keep only the clinically meaningful branches of admitdxpath.
ADMISSIONDX_KEEP_BRANCHES = (
    "All Diagnosis",
    "Non-operative Organ Systems",
    "Operative Organ Systems",
    "Elective",
)

# `pastHistory` is 26% administrative exhaust ("Performed" alone appears
# 278,271 times) living under this branch of the path.
PASTHISTORY_DROP_BRANCH = "Past History Obtain Options"

# Path segments that are UI navigation rather than clinical content. eICU stores
# each item as the trail of clicks used to reach the form, e.g.
#   notes / Progress Notes / Past History / Organ Systems / Endocrine (R) / ...
# The leading steps are identical on every entry for every patient, so they
# carry no discriminative signal and spent ~25% of every note's token budget.
#
# NOT stripped, deliberately:
#   "Elective", "Non-operative Organ Systems", "Operative Organ Systems"
#     — these are clinically meaningful (elective vs emergency, surgical vs
#       medical). Stripping "Elective" would leave a bare "Yes" with nothing
#       to attach it to.
#   "Diagnosis" fragments and the "(R)" organ markers are left in place — they
#     are clinical vocabulary rather than navigation.
#
# `diagnosis` and `treatment` need no equivalent: their taxonomies are medicine
# all the way up ("cardiovascular -> chest pain / ASHD -> coronary artery
# disease"), so nothing is removed from them.
NAVIGATION_SEGMENTS = {
    "admission diagnosis",   # admitdxpath root
    "All Diagnosis",         # admitdxpath branch label
    "Organ System",          # admitdxpath filler
    "notes",                 # pasthistorypath root
    "Progress Notes",        # pasthistorypath filler
    "Past History",          # pasthistorypath filler
    "Organ Systems",         # pasthistorypath filler
}

# The 5 agreed columns of `patient`.
PATIENT_COLUMNS = [
    "patientunitstayid",
    "uniquepid",
    "hospitalid",
    "unitdischargestatus",
    "unitdischargeoffset",
]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def _require(path: Path) -> Path:
    """Loud failure — no silent skipping of a missing source table."""
    if not path.exists():
        raise FileNotFoundError(f"Missing eICU source file: {path}")
    return path


def path_to_phrase(raw: str, sep: str) -> str:
    """
    Turn a taxonomy path into something prose-shaped, dropping the navigation
    steps that are identical on every entry.

        "cardiovascular|chest pain / ASHD|coronary artery disease|known"
     -> "cardiovascular, chest pain / ASHD, coronary artery disease, known"

        "notes/Progress Notes/Past History/Organ Systems/Endocrine (R)/NIDDM"
     -> "Endocrine (R), NIDDM"

    The clinical part of the path is kept in full on purpose (see choice 2 in
    the docstring) — leaf-only is unsafe, because some leaves are meaningless
    on their own ("known", "Yes").
    """
    parts = [p.strip() for p in raw.split(sep) if p.strip()]
    parts = [p for p in parts if p not in NAVIGATION_SEGMENTS]
    return ", ".join(parts)


def dedupe_keep_order(values) -> list:
    """Dedupe preserving first-appearance order, so the note stays chronological."""
    seen, out = set(), []
    for v in values:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def apply_leakage_filter(df: pd.DataFrame, offset_col: str,
                         discharge: pd.Series) -> pd.DataFrame:
    """Drop items recorded within LEAKAGE_BUFFER_MINUTES of ICU discharge."""
    cutoff = df.patientunitstayid.map(discharge) - LEAKAGE_BUFFER_MINUTES
    keep = df[offset_col].le(cutoff) | cutoff.isna()
    dropped = (~keep).sum()
    print(f"      leakage filter dropped {dropped:,} of {len(df):,} rows "
          f"({dropped / max(len(df), 1):.2%})")
    return df[keep]


def load_source(fname: str, text_from: str, offset_col: str,
                discharge: pd.Series, sep: str,
                path_col: str = None,
                keep_branches: tuple = None,
                drop_branch: str = None) -> pd.Series:
    """
    Load one source table and reduce it to  patientunitstayid -> "a, b, c".

    text_from   column whose value becomes the phrase (a taxonomy path)
    path_col    column used for branch filtering (defaults to text_from)
    """
    src = _require(EICU_DIR / fname)
    usecols = {"patientunitstayid", text_from, offset_col}
    if path_col:
        usecols.add(path_col)
    print(f"   reading {fname} ...")
    df = pd.read_csv(src, usecols=list(usecols), dtype=str, keep_default_na=False)

    df["patientunitstayid"] = pd.to_numeric(df.patientunitstayid, errors="coerce")
    df[offset_col] = pd.to_numeric(df[offset_col], errors="coerce")
    df = df.dropna(subset=["patientunitstayid", offset_col])
    df["patientunitstayid"] = df.patientunitstayid.astype("int64")

    branch_col = path_col or text_from
    if keep_branches:
        before = len(df)
        pat = "|".join(b.replace("|", r"\|") for b in keep_branches)
        df = df[df[branch_col].str.contains(pat, regex=True, na=False)]
        print(f"      branch filter kept {len(df):,} of {before:,} rows")
    if drop_branch:
        before = len(df)
        df = df[~df[branch_col].str.contains(drop_branch, regex=False, na=False)]
        print(f"      dropped admin branch: {before - len(df):,} rows")

    df = apply_leakage_filter(df, offset_col, discharge)

    # chronological, then dedupe by first appearance
    df = df.sort_values(["patientunitstayid", offset_col])
    df["phrase"] = df[text_from].map(lambda s: path_to_phrase(s, sep))
    df = df[df.phrase.str.len() > 0]

    return df.groupby("patientunitstayid").phrase.apply(
        lambda s: "; ".join(dedupe_keep_order(s))
    )


# ═══════════════════════════════════════════════════════════════════════════
# Build
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── patient: the spine + the leakage anchor ──────────────────────────
    print("\n[1/3] patient")
    pat = pd.read_csv(_require(EICU_DIR / "patient.csv.gz"),
                      usecols=PATIENT_COLUMNS, dtype=str, keep_default_na=False)
    pat["patientunitstayid"] = pat.patientunitstayid.astype("int64")
    pat["unitdischargeoffset"] = pd.to_numeric(pat.unitdischargeoffset, errors="coerce")
    pat["hospitalid"] = pd.to_numeric(pat.hospitalid, errors="coerce").astype("Int64")
    discharge = pat.set_index("patientunitstayid").unitdischargeoffset
    print(f"   {len(pat):,} ICU stays")
    if pat.patientunitstayid.duplicated().any():
        raise ValueError("patientunitstayid is not unique — the grain assumption is broken")

    # ── the four text sources ────────────────────────────────────────────
    print("\n[2/3] text sources")
    parts = {}
    parts["admissionDx"] = load_source(
        "admissionDx.csv.gz", text_from="admitdxpath",
        offset_col="admitdxenteredoffset", discharge=discharge, sep="|",
        keep_branches=ADMISSIONDX_KEEP_BRANCHES)
    parts["diagnosis"] = load_source(
        "diagnosis.csv.gz", text_from="diagnosisstring",
        offset_col="diagnosisoffset", discharge=discharge, sep="|")
    parts["treatment"] = load_source(
        "treatment.csv.gz", text_from="treatmentstring",
        offset_col="treatmentoffset", discharge=discharge, sep="|")
    parts["pastHistory"] = load_source(
        "pastHistory.csv.gz", text_from="pasthistorypath",
        offset_col="pasthistoryenteredoffset", discharge=discharge, sep="/",
        drop_branch=PASTHISTORY_DROP_BRANCH)

    # ── assemble ─────────────────────────────────────────────────────────
    print("\n[3/3] assembling notes")
    notes = pd.DataFrame({"patientunitstayid": pat.patientunitstayid})
    for key, _ in SECTIONS:
        notes[key] = notes.patientunitstayid.map(parts[key]).fillna("")

    def assemble(row) -> str:
        chunks = [f"{header}: {row[key]}."
                  for key, header in SECTIONS if row[key]]
        return " ".join(chunks)

    notes["note"] = notes.apply(assemble, axis=1)
    notes["n_chars"] = notes.note.str.len()
    # 3.86 chars/token, calibrated against the real BioClinicalBERT tokenizer
    # on a 3,000-note sample of this corpus (chars/4 was the earlier guess).
    notes["est_tokens"] = (notes.n_chars / 3.86).round().astype(int)
    for key, _ in SECTIONS:
        notes[f"has_{key}"] = notes[key].str.len().gt(0)
    notes = notes[["patientunitstayid", "note", "n_chars", "est_tokens"]
                  + [f"has_{k}" for k, _ in SECTIONS]]

    # Drop stays whose note is empty — same effect as MIMIC's inner join
    # (an admission with no surviving note is absent from the dataset), done
    # here at build time for symmetry with the vitals features parquet.
    n_before = len(notes)
    empty = notes.n_chars.eq(0)
    notes = notes[~empty].reset_index(drop=True)
    print(f"\ndropped {empty.sum():,} empty-note stays of {n_before:,} "
          f"({empty.mean():.1%}) — matches MIMIC's no-note exclusion")

    notes.to_parquet(NOTES_OUT, index=False)
    pat.to_parquet(PATIENTS_OUT, index=False)

    # ── QA ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"wrote {NOTES_OUT}")
    print(f"wrote {PATIENTS_OUT}")
    print("=" * 68)
    print(f"stays with a note    : {len(notes):,}  (empties already dropped)")
    print("\nnote length (chars):")
    print(notes["n_chars"]
          .describe(percentiles=[.1, .5, .9, .99]).to_string())
    over = notes.est_tokens.gt(BERT_MAX_TOKENS)
    print(f"\nest. tokens > {BERT_MAX_TOKENS}: {over.sum():,} ({over.mean():.1%}) "
          f"— these get truncated, losing past history first")
    print("\nsection coverage:")
    for key, header in SECTIONS:
        print(f"   {header:<22} {notes[f'has_{key}'].mean():6.1%}")

    print("\n--- example note ---")
    sample = notes[notes.n_chars.between(400, 900)].head(1)
    if len(sample):
        print(sample.iloc[0].note[:1000])


if __name__ == "__main__":
    main()
