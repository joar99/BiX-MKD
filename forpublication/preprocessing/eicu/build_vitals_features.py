"""
Build the eICU vitals-teacher feature matrix  (eICU-CRD v2.0 -> parquet).
========================================================================

A faithful port of MIMIC's `waveform_features_v2.py` recipe onto eICU's
`vitalPeriodic` table. The statistics, quality filters, segment layout, delta
construction and column naming are copied deliberately — only the *source* of
the time series and two ratified parameters differ.

What is IDENTICAL to MIMIC:
  - the 10 statistics per signal per segment, including the special cases for
    0 and 1 valid samples, ddof=1 std, linregress slope, bias-corrected skew
  - quality filters: zeros -> NaN, RESP > 80 -> NaN
  - 6 segments, plus a full-window block, plus per-segment and full-window
    Pearson interactions, plus last-minus-first-segment deltas
  - binning by nanmean, window ending at (outcome - EXCLUSION_BUFFER_HOURS)
  - column naming: seg{i}_{SIG}_{stat} / full_{SIG}_{stat} /
    seg{i}_corr_{a}_{b} / full_corr_{a}_{b} / delta_{SIG}_{stat}

What DIFFERS, and why (both ratified — see db/eicu-data-plan.md):
  - D5  PULSE does not exist in eICU. 3 signals, not 4. The ("HR","PULSE")
        interaction pair dies with it — 2 pairs, not 3.
  - D9  OBSERVATION_WINDOW_HOURS = 12, not 24. MIMIC's 24h window plus the 6h
        buffer needs 30h of stay, which only 56.9% of eICU ICU stays have, and
        it would discard 42% of all deaths. 12h keeps 78% of stays and 70% of
        deaths while still giving 24 samples per segment.
  - eICU samples every ~5 minutes, so the grid is 5-minute rather than
    per-minute. 12h -> 144 slots -> 24 per segment.

  => FEATURE_DIM = 236, not 317.

Outputs (into db/e-icu-data/):
  eicu_vitals_features.parquet   one row per ELIGIBLE ICU stay: 236 features

Run:  python forpublication/preprocessing/eicu/build_vitals_features.py
      (expect ~10-15 min: a 146M-row scan, then per-stay statistics)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
_FORPUB = Path(__file__).resolve().parents[2]
EICU_DIR = _FORPUB / "db" / "e-icu-data" / "physionet.org" / "files" / "eicu-crd" / "2.0"
OUT = _FORPUB / "db" / "e-icu-data" / "eicu_vitals_features.parquet"

# --- ratified deviations from MIMIC --------------------------------------
CANONICAL_SIGNALS = ["HR", "RESP", "SpO2"]          # D5: PULSE does not exist
INTERACTION_PAIRS = [("HR", "SpO2"), ("SpO2", "RESP")]   # D5: HR-PULSE died
OBSERVATION_WINDOW_HOURS = 12                       # D9: not MIMIC's 24
# -------------------------------------------------------------------------

EXCLUSION_BUFFER_HOURS = 6                          # unchanged from MIMIC
N_SEGMENTS = 6                                      # unchanged from MIMIC
SAMPLE_MINUTES = 5                                  # eICU vitalPeriodic cadence

# eICU column -> MIMIC canonical signal name
SIGNAL_SOURCE = {"HR": "heartrate", "SpO2": "sao2", "RESP": "respiration"}

STAT_NAMES = ["mean", "median", "std", "iqr",
              "min", "max", "range", "slope",
              "skewness", "completeness"]
DELTA_STATS = ["mean", "std", "min", "max"]
RESP_MAX_VALID = 80                                 # unchanged from MIMIC

N_STATS = len(STAT_NAMES)
N_SIGNALS = len(CANONICAL_SIGNALS)
N_INTERACTIONS = len(INTERACTION_PAIRS)

SAMPLES_PER_HOUR = 60 // SAMPLE_MINUTES                      # 12
WINDOW_SAMPLES = OBSERVATION_WINDOW_HOURS * SAMPLES_PER_HOUR  # 144
SEGMENT_SAMPLES = WINDOW_SAMPLES // N_SEGMENTS                # 24
BUFFER_MINUTES = EXCLUSION_BUFFER_HOURS * 60                  # 360
WINDOW_MINUTES = OBSERVATION_WINDOW_HOURS * 60                # 720

FEATURE_DIM = (N_SEGMENTS * N_STATS * N_SIGNALS      # 180
               + N_STATS * N_SIGNALS                 #  30
               + (N_SEGMENTS + 1) * N_INTERACTIONS   #  14
               + len(DELTA_STATS) * N_SIGNALS)       #  12   => 236


# ═══════════════════════════════════════════════════════════════════════════
# Statistics — copied from MIMIC's waveform_features_v2.py, unchanged
# ═══════════════════════════════════════════════════════════════════════════
def apply_quality_filters(values: np.ndarray, signal_name: str) -> np.ndarray:
    cleaned = values.copy().astype(float)
    cleaned[cleaned == 0.0] = np.nan
    if signal_name == "RESP":
        cleaned[cleaned > RESP_MAX_VALID] = np.nan
    return cleaned


def compute_signal_stats(values: np.ndarray, expected_samples: int) -> np.ndarray:
    valid = values[~np.isnan(values)]
    n_valid = len(valid)
    completeness = n_valid / expected_samples if expected_samples > 0 else 0.0

    if n_valid == 0:
        features = np.zeros(N_STATS)
        features[9] = 0.0
        return features

    if n_valid == 1:
        val = valid[0]
        return np.array([val, val, 0.0, 0.0, val, val, 0.0, 0.0, 0.0, completeness])

    mean_val = np.mean(valid)
    median_val = np.median(valid)
    std_val = np.std(valid, ddof=1)
    q75, q25 = np.percentile(valid, [75, 25])
    iqr_val = q75 - q25
    min_val = np.min(valid)
    max_val = np.max(valid)
    range_val = max_val - min_val

    valid_indices = np.where(~np.isnan(values))[0]
    slope_val = (scipy_stats.linregress(valid_indices, valid).slope
                 if len(valid_indices) >= 2 else 0.0)
    skew_val = (scipy_stats.skew(valid, bias=False)
                if n_valid >= 3 and std_val > 0 else 0.0)

    return np.array([mean_val, median_val, std_val, iqr_val,
                     min_val, max_val, range_val, slope_val,
                     skew_val, completeness])


def compute_interaction_features(signal_windows: dict, start: int, end: int) -> np.ndarray:
    interactions = np.zeros(N_INTERACTIONS)
    for k, (sig_a, sig_b) in enumerate(INTERACTION_PAIRS):
        if sig_a not in signal_windows or sig_b not in signal_windows:
            continue
        vals_a = signal_windows[sig_a][start:end]
        vals_b = signal_windows[sig_b][start:end]
        both = ~np.isnan(vals_a) & ~np.isnan(vals_b)
        if np.sum(both) < 10:
            continue
        a, b = vals_a[both], vals_b[both]
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            continue
        corr, _ = scipy_stats.pearsonr(a, b)
        interactions[k] = corr if np.isfinite(corr) else 0.0
    return interactions


def compute_stay_features(signal_windows: dict) -> np.ndarray:
    """Same block order as MIMIC: segments, full window, interactions, deltas."""
    fv = np.zeros(FEATURE_DIM)
    offset = 0
    segment_stats = {}

    # 1. segment features
    for sig in CANONICAL_SIGNALS:
        segment_stats[sig] = []
        for seg in range(N_SEGMENTS):
            s = seg * SEGMENT_SAMPLES
            cleaned = apply_quality_filters(signal_windows[sig][s:s + SEGMENT_SAMPLES], sig)
            stats = compute_signal_stats(cleaned, SEGMENT_SAMPLES)
            segment_stats[sig].append(stats)
            fv[offset:offset + N_STATS] = stats
            offset += N_STATS

    # 2. full-window features
    for sig in CANONICAL_SIGNALS:
        cleaned = apply_quality_filters(signal_windows[sig], sig)
        fv[offset:offset + N_STATS] = compute_signal_stats(cleaned, WINDOW_SAMPLES)
        offset += N_STATS

    # 3. interactions — per segment, then full window
    cleaned_windows = {sig: apply_quality_filters(signal_windows[sig], sig)
                       for sig in CANONICAL_SIGNALS}
    for seg in range(N_SEGMENTS):
        s = seg * SEGMENT_SAMPLES
        fv[offset:offset + N_INTERACTIONS] = compute_interaction_features(
            cleaned_windows, s, s + SEGMENT_SAMPLES)
        offset += N_INTERACTIONS
    fv[offset:offset + N_INTERACTIONS] = compute_interaction_features(
        cleaned_windows, 0, WINDOW_SAMPLES)
    offset += N_INTERACTIONS

    # 4. deltas — last segment minus first
    delta_idx = [STAT_NAMES.index(s) for s in DELTA_STATS]
    for sig in CANONICAL_SIGNALS:
        first, last = segment_stats[sig][0], segment_stats[sig][-1]
        for i in delta_idx:
            fv[offset] = last[i] - first[i]
            offset += 1

    assert offset == FEATURE_DIM, f"Expected {FEATURE_DIM} features, got {offset}"
    return fv


def get_feature_columns() -> list:
    cols = []
    for sig in CANONICAL_SIGNALS:
        for seg in range(N_SEGMENTS):
            cols += [f"seg{seg}_{sig}_{s}" for s in STAT_NAMES]
    for sig in CANONICAL_SIGNALS:
        cols += [f"full_{sig}_{s}" for s in STAT_NAMES]
    for seg in range(N_SEGMENTS):
        cols += [f"seg{seg}_corr_{a}_{b}" for a, b in INTERACTION_PAIRS]
    cols += [f"full_corr_{a}_{b}" for a, b in INTERACTION_PAIRS]
    for sig in CANONICAL_SIGNALS:
        cols += [f"delta_{sig}_{s}" for s in DELTA_STATS]
    assert len(cols) == FEATURE_DIM, f"Expected {FEATURE_DIM} columns, got {len(cols)}"
    return cols


# ═══════════════════════════════════════════════════════════════════════════
# Build
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    print(f"window {OBSERVATION_WINDOW_HOURS}h + {EXCLUSION_BUFFER_HOURS}h buffer | "
          f"{WINDOW_SAMPLES} slots of {SAMPLE_MINUTES} min | "
          f"{SEGMENT_SAMPLES}/segment | FEATURE_DIM {FEATURE_DIM}")

    # ── eligible stays ───────────────────────────────────────────────────
    pat = pd.read_csv(EICU_DIR / "patient.csv.gz",
                      usecols=["patientunitstayid", "unitdischargeoffset"])
    need = BUFFER_MINUTES + WINDOW_MINUTES
    elig = pat[pat.unitdischargeoffset >= need].reset_index(drop=True)
    print(f"\neligible stays: {len(elig):,} of {len(pat):,} "
          f"({len(elig)/len(pat):.1%}) — need {need/60:.0f}h of stay")

    stay_ids = elig.patientunitstayid.to_numpy()
    row_of = pd.Series(np.arange(len(elig)), index=stay_ids)
    # window ends at (ICU discharge - buffer), starts 12h before that
    win_end = elig.unitdischargeoffset.to_numpy() - BUFFER_MINUTES
    win_start = win_end - WINDOW_MINUTES

    # ── scatter readings onto the 5-minute grid (nanmean, as MIMIC bins) ──
    n = len(elig)
    sums = {s: np.zeros((n, WINDOW_SAMPLES), dtype=np.float64) for s in CANONICAL_SIGNALS}
    cnts = {s: np.zeros((n, WINDOW_SAMPLES), dtype=np.int32) for s in CANONICAL_SIGNALS}

    scanned = 0
    for ch in pd.read_csv(EICU_DIR / "vitalPeriodic.csv.gz",
                          usecols=["patientunitstayid", "observationoffset"]
                                  + list(SIGNAL_SOURCE.values()),
                          chunksize=4_000_000):
        scanned += len(ch)
        idx = ch.patientunitstayid.map(row_of)
        ch = ch[idx.notna()]
        if not len(ch):
            continue
        idx = idx[idx.notna()].to_numpy().astype(np.int64)
        off = ch.observationoffset.to_numpy()
        rel = off - win_start[idx]
        keep = (rel >= 0) & (off <= win_end[idx])
        if not keep.any():
            continue
        idx, rel = idx[keep], rel[keep]
        slot = np.clip((rel // SAMPLE_MINUTES).astype(np.int64), 0, WINDOW_SAMPLES - 1)
        for sig, col in SIGNAL_SOURCE.items():
            v = pd.to_numeric(ch[col], errors="coerce").to_numpy()[keep]
            ok = ~np.isnan(v)
            if ok.any():
                np.add.at(sums[sig], (idx[ok], slot[ok]), v[ok])
                np.add.at(cnts[sig], (idx[ok], slot[ok]), 1)
        print(f"   scanned {scanned:,} rows ...", end="\r")
    print(f"   scanned {scanned:,} vitalPeriodic rows      ")

    grids = {}
    for sig in CANONICAL_SIGNALS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            g = sums[sig] / cnts[sig]          # nanmean; 0/0 -> nan
        g[cnts[sig] == 0] = np.nan
        grids[sig] = g
    del sums, cnts

    # ── drop stays with NO usable signal at all ──────────────────────────
    # Mirrors MIMIC's process_admission():
    #     if not signal_windows: return None
    # An admission with nothing to compute from is skipped, not emitted as an
    # all-zero row. Stays missing only SOME signals are kept — MIMIC keeps those
    # too, filling the absent signal with zeros and a completeness of 0.
    any_signal = np.zeros(n, dtype=bool)
    for sig in CANONICAL_SIGNALS:
        any_signal |= ~np.isnan(grids[sig]).all(axis=1)
    dropped = int((~any_signal).sum())
    print(f"\nskipping {dropped:,} stays with no usable signal "
          f"({dropped/n:.1%}) — MIMIC excludes these too")
    keep_idx = np.where(any_signal)[0]
    stay_ids = stay_ids[keep_idx]
    grids = {s: grids[s][keep_idx] for s in CANONICAL_SIGNALS}
    n = len(keep_idx)

    # ── per-stay features ────────────────────────────────────────────────
    print(f"\ncomputing features for {n:,} stays (this is the slow part) ...")
    feats = np.zeros((n, FEATURE_DIM))
    for i in range(n):
        feats[i] = compute_stay_features({s: grids[s][i] for s in CANONICAL_SIGNALS})
        if i % 20000 == 0:
            print(f"   {i:,}/{n:,}", end="\r")

    out = pd.DataFrame(feats, columns=get_feature_columns())
    out.insert(0, "patientunitstayid", stay_ids)
    out.to_parquet(OUT, index=False)

    # ── QA ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"wrote {OUT}")
    print("=" * 68)
    print(f"stays: {len(out):,}   features: {FEATURE_DIM}")
    print("\nfull-window completeness by signal:")
    for sig in CANONICAL_SIGNALS:
        c = out[f"full_{sig}_completeness"]
        print(f"   {sig:<5} mean {c.mean():6.1%}   median {c.median():6.1%}   "
              f"zero-coverage stays {c.eq(0).sum():,} ({c.eq(0).mean():.1%})")
    dead = (out[[f"full_{s}_completeness" for s in CANONICAL_SIGNALS]].sum(axis=1) == 0)
    print(f"\nstays with NO vitals at all: {dead.sum():,}  (should be 0 — they are excluded)")
    print(f"non-finite feature values  : {(~np.isfinite(feats)).sum():,}")


if __name__ == "__main__":
    main()
