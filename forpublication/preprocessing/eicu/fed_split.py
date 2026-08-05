"""
eICU federated split — assign whole hospitals to model roles AND to train/val/test.
===================================================================================

The genuine federated setup: each role is served by a DISJOINT set of hospitals,
and — because eICU is genuinely multi-site — train/val/test is ALSO split by
whole hospitals. A role's model is therefore evaluated on hospitals it never
trained on (true cross-institution generalisation), not just new patients from
the same hospitals. MIMIC could not do this (single site); eICU can.

Two nested applications of the SAME mortality-stratified greedy bin-packer:
  ROLES:   text_teacher / waveform_teacher / student   at 40 / 40 / 20 of stays
  SPLITS:  within each role, TRAIN / VAL / TEST         at 70 / 15 / 15 of stays

Both are stratified by hospital mortality rate, so every role — and every split
within a role — gets a balanced slice of each mortality band. That keeps stay
count AND mortality comparable across all cells, so nothing downstream is
confounded by between-site case-mix drift.

Pipeline:
  1. CUT to paired (stays in BOTH build_*.py outputs).
  2. PATIENT DISJOINTNESS — drop patients spanning >1 hospital (no reliable
     cross-hospital time order in eICU). Run FAILS LOUDLY if any patient or
     hospital ends up spanning two roles or two splits.
  3. ROLE assignment, then SPLIT assignment within each role.

Because whole hospitals move as a unit, ratios land close but not exact — the QA
block reports how close, on all three dimensions the split must balance:
stay %, mortality rate, and hospital count. The student cohort (fewest hospitals)
is the one to watch.

Output (into db/e-icu-data/):
  eicu_fed_split.parquet   one row per assigned ICU stay:
     patientunitstayid, uniquepid, hospitalid, mortality, role, split

Run:  python forpublication/preprocessing/eicu/fed_split.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
_FORPUB = Path(__file__).resolve().parents[2]
DATA = _FORPUB / "db" / "e-icu-data"
TEXT_PARQUET = DATA / "eicu_text_notes.parquet"
VITALS_PARQUET = DATA / "eicu_vitals_features.parquet"
PATIENTS_PARQUET = DATA / "eicu_patients.parquet"
OUT = DATA / "eicu_fed_split.parquet"

ROLE_TARGETS = {"text_teacher": 0.40, "waveform_teacher": 0.40, "student": 0.20}
SPLIT_TARGETS = {"TRAIN": 0.70, "VAL": 0.15, "TEST": 0.15}

# "drop" — exclude patients with stays at >1 hospital (clean disjointness)
# "majority" — bind them to the hospital holding most of their stays
MULTI_HOSPITAL_PATIENTS = "drop"

# Mortality bands used to balance each assignment. Roles have many hospitals so
# 3 is comfortable; the within-role split of the small student cohort has fewer,
# so it uses SPLIT_STRATA (kept lower to avoid empty bands).
N_MORTALITY_STRATA = 3
SPLIT_STRATA = 2

SEED = 42


# ═══════════════════════════════════════════════════════════════════════════
def _require(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"Missing input — run the build scripts first: {p}")
    return p


def stratified_assign(hosp: pd.DataFrame, targets: dict, n_strata: int) -> dict:
    """
    Assign whole hospitals to the keys of `targets` (fractions summing to 1),
    hitting the target share of STAYS while balancing mortality.

    hosp: rows indexed arbitrarily, columns [hospitalid, n_stays, n_deaths, rate].
    Returns {hospitalid: key}. Deterministic (stable sorts, no RNG).
    """
    hosp = hosp.reset_index(drop=True)
    n_strata = max(1, min(n_strata, len(hosp)))
    # stratify by mortality rank (robust to the pile of 0%-mortality small sites)
    order = hosp.sort_values(["rate", "n_stays"], kind="mergesort").index.to_numpy()
    stratum = np.empty(len(hosp), dtype=int)
    stratum[order] = np.arange(len(hosp)) * n_strata // len(hosp)

    keys = list(targets)
    out = {}
    for s in range(n_strata):
        sub = hosp[stratum == s].sort_values("n_stays", ascending=False)
        tot = sub.n_stays.sum()
        target = {k: targets[k] * tot for k in keys}
        got = {k: 0 for k in keys}
        for hid, n in zip(sub.hospitalid, sub.n_stays):
            # give this hospital to the key with the largest remaining deficit
            k = max(keys, key=lambda k: (target[k] - got[k]) / target[k]
                    if target[k] > 0 else -1)
            out[hid] = k
            got[k] += n
    return out


def _hosp_stats(df: pd.DataFrame) -> pd.DataFrame:
    h = (df.groupby("hospitalid")
         .agg(n_stays=("patientunitstayid", "size"),
              n_deaths=("mortality", "sum")).reset_index())
    h["rate"] = h.n_deaths / h.n_stays
    return h


def main() -> None:
    # ── 1. cut to paired ─────────────────────────────────────────────────
    text_ids = set(pd.read_parquet(_require(TEXT_PARQUET),
                                   columns=["patientunitstayid"]).patientunitstayid)
    vit_ids = set(pd.read_parquet(_require(VITALS_PARQUET),
                                  columns=["patientunitstayid"]).patientunitstayid)
    paired = text_ids & vit_ids
    print(f"text stays {len(text_ids):,} | vitals stays {len(vit_ids):,} "
          f"| PAIRED {len(paired):,}")

    pat = pd.read_parquet(_require(PATIENTS_PARQUET))
    pat = pat[pat.patientunitstayid.isin(paired)].copy()

    usable = pat.unitdischargestatus.isin(["Alive", "Expired"])
    if (~usable).any():
        print(f"dropping {(~usable).sum():,} paired stays with no usable label")
    pat = pat[usable].copy()
    pat["mortality"] = (pat.unitdischargestatus == "Expired").astype(int)

    # ── 2. patient disjointness ──────────────────────────────────────────
    hosp_per_pat = pat.groupby("uniquepid").hospitalid.nunique()
    spanning = set(hosp_per_pat[hosp_per_pat > 1].index)
    print(f"\npatients spanning >1 hospital: {len(spanning):,} "
          f"({len(spanning)/pat.uniquepid.nunique():.2%})")
    if MULTI_HOSPITAL_PATIENTS == "drop":
        before = len(pat)
        pat = pat[~pat.uniquepid.isin(spanning)].copy()
        print(f"  dropped -> {before - len(pat):,} stays removed, {len(pat):,} remain")
    elif MULTI_HOSPITAL_PATIENTS == "majority":
        maj = (pat[pat.uniquepid.isin(spanning)]
               .groupby(["uniquepid", "hospitalid"]).size().reset_index(name="n")
               .sort_values(["uniquepid", "n"], ascending=[True, False])
               .drop_duplicates("uniquepid").set_index("uniquepid").hospitalid)
        keep = pat.uniquepid.map(maj)
        drop_mask = pat.uniquepid.isin(spanning) & (pat.hospitalid != keep)
        pat = pat[~drop_mask].copy()
        print(f"  bound to majority hospital -> {drop_mask.sum():,} off-hospital stays removed")
    else:
        raise ValueError(f"Unknown MULTI_HOSPITAL_PATIENTS: {MULTI_HOSPITAL_PATIENTS}")

    # ── 3. roles, then splits within each role ───────────────────────────
    role_of = stratified_assign(_hosp_stats(pat), ROLE_TARGETS, N_MORTALITY_STRATA)
    pat["role"] = pat.hospitalid.map(role_of)

    pat["split"] = pd.NA
    for role in ROLE_TARGETS:
        sub = pat[pat.role == role]
        split_of = stratified_assign(_hosp_stats(sub), SPLIT_TARGETS, SPLIT_STRATA)
        pat.loc[sub.index, "split"] = pat.loc[sub.index, "hospitalid"].map(split_of)

    # ── disjointness guarantees (loud) ───────────────────────────────────
    for col, label in [("role", "role"), ("split", "role/split")]:
        key = "role" if col == "role" else ["role", "split"]
        if col == "split":
            # a hospital must map to exactly one (role, split) cell
            hcells = pat.groupby("hospitalid")[["role", "split"]].nunique()
            if (hcells > 1).any().any():
                raise AssertionError("a hospital spans >1 role/split cell — broken")
        spans = pat.groupby("uniquepid")[col].nunique() if col == "role" else None
        if spans is not None and (spans > 1).any():
            raise AssertionError(f"{(spans > 1).sum()} patients span >1 {label} — disjointness broken")

    out = pat[["patientunitstayid", "uniquepid", "hospitalid", "mortality", "role", "split"]]
    out.to_parquet(OUT, index=False)

    # ── QA ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print(f"wrote {OUT}")
    print("=" * 74)
    N = len(out)
    print(f"assigned stays {N:,} | patients {out.uniquepid.nunique():,} "
          f"| hospitals {out.hospitalid.nunique()} | overall mortality {out.mortality.mean():.2%}\n")

    print(f"{'role':<17}{'split':<7}{'stays':>9}{'stay%':>8}{'of-target':>11}"
          f"{'mortality':>11}{'hosp':>6}")
    for role in ROLE_TARGETS:
        r = out[out.role == role]
        for split in SPLIT_TARGETS:
            g = r[r.split == split]
            share = len(g) / len(r) if len(r) else 0
            print(f"{role:<17}{split:<7}{len(g):>9,}{share:>7.1%}"
                  f"{share - SPLIT_TARGETS[split]:>+10.1%}"
                  f"{g.mortality.mean():>11.2%}{g.hospitalid.nunique():>6}")
        print(f"{'':17}{'ALL':<7}{len(r):>9,}{len(r)/N:>7.1%}"
              f"{len(r)/N - ROLE_TARGETS[role]:>+10.1%}"
              f"{r.mortality.mean():>11.2%}{r.hospitalid.nunique():>6}")
        print()


if __name__ == "__main__":
    main()
