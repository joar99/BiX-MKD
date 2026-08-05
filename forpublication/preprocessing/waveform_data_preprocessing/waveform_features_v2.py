import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from scipy import stats as scipy_stats
from typing import Optional
import warnings

CANONICAL_SIGNALS = ["HR", "RESP", "SpO2", "PULSE"]

SIGNAL_HARMONISATION = {
    "%SpO2": "SpO2",
}

STAT_NAMES = [
    "mean", "median", "std", "iqr",
    "min", "max", "range", "slope",
    "skewness", "completeness"
]

N_STATS = len(STAT_NAMES)
N_SIGNALS = len(CANONICAL_SIGNALS)

# Temporal parameters
OBSERVATION_WINDOW_HOURS = 24

# How many hours before 0hour to capture
EXCLUSION_BUFFER_HOURS = 6

# After downsampling to 1 sample per minute
SAMPLES_PER_HOUR = 60

# 1440
WINDOW_SAMPLES = OBSERVATION_WINDOW_HOURS * SAMPLES_PER_HOUR

# Divided into 6 segments
N_SEGMENTS = 6

# 4 hours per segment
SEGMENT_HOURS = OBSERVATION_WINDOW_HOURS // N_SEGMENTS

# 240 samples
SEGMENT_SAMPLES = SEGMENT_HOURS * SAMPLES_PER_HOUR

# Data quality thresholds
RESP_MAX_VALID = 80

# Cross-signal interaction pairs
INTERACTION_PAIRS = [

    # Cardiopulmonary coupling
    ("HR", "SpO2"),

    # HR-PULSE divergence (arterial vs electrical)
    ("HR", "PULSE"),

    # Oxygenation-ventilation relationship
    ("SpO2", "RESP"),
]

# 3
N_INTERACTIONS = len(INTERACTION_PAIRS)

# Delta features: last segment minus first segment
# 4 per signal
DELTA_STATS = ["mean", "std", "min", "max"]
N_DELTA_STATS = len(DELTA_STATS)

# Total feature dimensionality

# 240
N_SEGMENT_FEATURES = N_SEGMENTS * N_STATS * N_SIGNALS
# 40
N_FULLWINDOW_FEATURES = N_STATS * N_SIGNALS
# 21
N_INTERACTION_FEATURES = (N_SEGMENTS + 1) * N_INTERACTIONS
# 16
N_DELTA_FEATURES = N_DELTA_STATS * N_SIGNALS
# 317
FEATURE_DIM = (N_SEGMENT_FEATURES + N_FULLWINDOW_FEATURES
               + N_INTERACTION_FEATURES + N_DELTA_FEATURES)

def harmonise_signal_names(sig_names: list[str]) -> list[str]:
    return [SIGNAL_HARMONISATION.get(name, name) for name in sig_names]

def load_recording(record_path: str) -> Optional[dict]:
    try:
        record = wfdb.rdrecord(record_path)
    except Exception:
        return None

    sig_names = harmonise_signal_names(record.sig_name)
    fs = record.fs

    if record.base_datetime is not None:
        start_time = pd.Timestamp(record.base_datetime)
    elif record.base_date is not None:
        start_time = pd.Timestamp(record.base_date)
    else:
        return None

    signals = {}
    for i, name in enumerate(sig_names):
        if name in CANONICAL_SIGNALS:
            signals[name] = record.p_signal[:, i].copy()

    if not signals:
        return None

    return {
        'signals': signals,
        'fs': fs,
        'start_time': start_time,
        'n_samples': record.sig_len,
    }

def downsample_to_per_minute(values: np.ndarray, fs: float) -> np.ndarray:
    if abs(fs - 1.0) < 0.01:
        n_minutes = len(values) // 60
        if n_minutes == 0:
            return np.array([])
        trimmed = values[:n_minutes * 60].reshape(n_minutes, 60)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(trimmed, axis=1)
    elif fs < 0.02:
        return values.copy()
    else:
        samples_per_minute = int(round(fs * 60))
        if samples_per_minute < 1:
            return values.copy()
        n_minutes = len(values) // samples_per_minute
        if n_minutes == 0:
            return np.array([])
        trimmed = values[:n_minutes * samples_per_minute].reshape(n_minutes, samples_per_minute)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(trimmed, axis=1)


def compute_sample_times(start_time: pd.Timestamp,
                         n_samples: int) -> pd.DatetimeIndex:
    return pd.date_range(start=start_time, periods=n_samples, freq='min')

def apply_temporal_filter(signal_values: np.ndarray,
                          sample_times: pd.DatetimeIndex,
                          outcome_time: pd.Timestamp,
                          buffer_hours: float = EXCLUSION_BUFFER_HOURS
                          ) -> tuple[np.ndarray, pd.DatetimeIndex]:
    cutoff_time = outcome_time - pd.Timedelta(hours=buffer_hours)
    mask = sample_times <= cutoff_time
    return signal_values[mask], sample_times[mask]


def extract_observation_window(signal_values: np.ndarray,
                               sample_times: pd.DatetimeIndex,
                               window_hours: int = OBSERVATION_WINDOW_HOURS
                               ) -> tuple[np.ndarray, pd.DatetimeIndex]:
    window_samples = window_hours * SAMPLES_PER_HOUR
    if len(signal_values) < window_samples:
        return np.array([]), pd.DatetimeIndex([])
    return signal_values[-window_samples:], sample_times[-window_samples:]

def apply_quality_filters(values: np.ndarray, signal_name: str) -> np.ndarray:
    cleaned = values.copy().astype(float)
    cleaned[cleaned == 0.0] = np.nan
    if signal_name == "RESP":
        cleaned[cleaned > RESP_MAX_VALID] = np.nan
    return cleaned

# Compute 10 statistical features for a signal segment
def compute_signal_stats(values: np.ndarray,
                         expected_samples: int) -> np.ndarray:
    valid = values[~np.isnan(values)]
    n_valid = len(valid)

    completeness = n_valid / expected_samples if expected_samples > 0 else 0.0

    if n_valid == 0:
        features = np.zeros(N_STATS)
        features[9] = 0.0
        return features

    if n_valid == 1:
        val = valid[0]
        return np.array([
            val, val, 0.0, 0.0,
            val, val, 0.0, 0.0,
            0.0, completeness
        ])

    mean_val = np.mean(valid)
    median_val = np.median(valid)
    std_val = np.std(valid, ddof=1)
    q75, q25 = np.percentile(valid, [75, 25])
    iqr_val = q75 - q25
    min_val = np.min(valid)
    max_val = np.max(valid)
    range_val = max_val - min_val

    valid_indices = np.where(~np.isnan(values))[0]
    if len(valid_indices) >= 2:
        slope_val = scipy_stats.linregress(valid_indices, valid).slope
    else:
        slope_val = 0.0

    if n_valid >= 3 and std_val > 0:
        skew_val = scipy_stats.skew(valid, bias=False)
    else:
        skew_val = 0.0

    return np.array([
        mean_val, median_val, std_val, iqr_val,
        min_val, max_val, range_val, slope_val,
        skew_val, completeness
    ])

# Compute cross-signal interaction features for a given sample range
# For each interaction pair, computes the Pearson correlation between
# the two signals over the specified range. Returns NaN-safe values
# (0.0 if either signal has insufficient valid overlap)
def compute_interaction_features(signal_windows: dict[str, np.ndarray],
                                 start: int, end: int) -> np.ndarray:
    interactions = np.zeros(N_INTERACTIONS)

    for k, (sig_a, sig_b) in enumerate(INTERACTION_PAIRS):
        if sig_a not in signal_windows or sig_b not in signal_windows:
            interactions[k] = 0.0
            continue

        vals_a = signal_windows[sig_a][start:end]
        vals_b = signal_windows[sig_b][start:end]

        # Find positions where both signals are valid
        both_valid = ~np.isnan(vals_a) & ~np.isnan(vals_b)
        n_overlap = np.sum(both_valid)

        if n_overlap < 10:
            interactions[k] = 0.0
            continue

        a_valid = vals_a[both_valid]
        b_valid = vals_b[both_valid]

        # Check for zero variance (constant signal)
        if np.std(a_valid) < 1e-8 or np.std(b_valid) < 1e-8:
            interactions[k] = 0.0
            continue

        corr, _ = scipy_stats.pearsonr(a_valid, b_valid)
        interactions[k] = corr if np.isfinite(corr) else 0.0

    return interactions

# Compute the full 317 dim feature vector for a single admission
#
# Layout (317 features total):
# [0:240]   Segment features: 6 segments × 4 signals × 10 stats
# [240:280] Full-window features: 4 signals × 10 stats
# [280:301] Interaction features: 7 windows × 3 pairs
# [301:317] Delta features: 4 signals × 4 stats
def compute_admission_features_v2(signal_windows: dict[str, np.ndarray]) -> np.ndarray:
    feature_vector = np.zeros(FEATURE_DIM)
    offset = 0

    # ----- 1. Segment features (240) -----
    # Store segment stats for delta computation
    segment_stats = {}  # sig_name -> list of 6 stat arrays

    for sig_name in CANONICAL_SIGNALS:
        segment_stats[sig_name] = []

        for seg_idx in range(N_SEGMENTS):
            seg_start = seg_idx * SEGMENT_SAMPLES
            seg_end = seg_start + SEGMENT_SAMPLES

            if sig_name in signal_windows:
                seg_values = signal_windows[sig_name][seg_start:seg_end]
                cleaned = apply_quality_filters(seg_values, sig_name)
                stats = compute_signal_stats(cleaned, SEGMENT_SAMPLES)
            else:
                stats = np.zeros(N_STATS)

            segment_stats[sig_name].append(stats)
            feature_vector[offset:offset + N_STATS] = stats
            offset += N_STATS

    # ----- 2. Full-window features (40) -----
    for sig_name in CANONICAL_SIGNALS:
        if sig_name in signal_windows:
            cleaned = apply_quality_filters(signal_windows[sig_name], sig_name)
            stats = compute_signal_stats(cleaned, WINDOW_SAMPLES)
        else:
            stats = np.zeros(N_STATS)

        feature_vector[offset:offset + N_STATS] = stats
        offset += N_STATS

    # ----- 3. Cross-signal interactions (21) -----
    # Apply quality filters to all signals once for interaction computation
    cleaned_windows = {}
    for sig_name in CANONICAL_SIGNALS:
        if sig_name in signal_windows:
            cleaned_windows[sig_name] = apply_quality_filters(
                signal_windows[sig_name], sig_name
            )

    # Per-segment interactions
    for seg_idx in range(N_SEGMENTS):
        seg_start = seg_idx * SEGMENT_SAMPLES
        seg_end = seg_start + SEGMENT_SAMPLES
        interactions = compute_interaction_features(
            cleaned_windows, seg_start, seg_end
        )
        feature_vector[offset:offset + N_INTERACTIONS] = interactions
        offset += N_INTERACTIONS

    # Full-window interactions
    interactions = compute_interaction_features(
        cleaned_windows, 0, WINDOW_SAMPLES
    )
    feature_vector[offset:offset + N_INTERACTIONS] = interactions
    offset += N_INTERACTIONS

    # ----- 4. Delta features (16) -----
    # Last segment stats minus first segment stats for key statistics
    delta_stat_indices = [STAT_NAMES.index(s) for s in DELTA_STATS]

    for sig_name in CANONICAL_SIGNALS:
        first_seg = segment_stats[sig_name][0]     # Segment 0 (hours 0-4)
        last_seg = segment_stats[sig_name][-1]      # Segment 5 (hours 20-24)

        for stat_idx in delta_stat_indices:
            delta = last_seg[stat_idx] - first_seg[stat_idx]
            feature_vector[offset] = delta
            offset += 1

    assert offset == FEATURE_DIM, f"Expected {FEATURE_DIM} features, got {offset}"
    return feature_vector

def get_admission_recordings(subject_id: int,
                             waveform_dir: Path) -> list[Path]:
    subject_dir = waveform_dir / str(subject_id)
    if not subject_dir.exists():
        return []
    hea_files = sorted(subject_dir.glob("*n.hea"))
    return [f.with_suffix('') for f in hea_files]

# Process all recordings for a single admission and produce a 317 dim vector
def process_admission(subject_id: int,
                      outcome_time: pd.Timestamp,
                      waveform_dir: Path) -> Optional[np.ndarray]:

    record_paths = get_admission_recordings(subject_id, waveform_dir)
    if not record_paths:
        return None

    signal_segments: dict[str, list[tuple[pd.DatetimeIndex, np.ndarray]]] = {
        sig: [] for sig in CANONICAL_SIGNALS
    }

    for rec_path in record_paths:
        rec_data = load_recording(str(rec_path))
        if rec_data is None:
            continue

        fs = rec_data['fs']
        start_time = rec_data['start_time']

        for sig_name, raw_values in rec_data['signals'].items():
            downsampled = downsample_to_per_minute(raw_values, fs)
            if len(downsampled) == 0:
                continue
            timestamps = compute_sample_times(start_time, len(downsampled))
            signal_segments[sig_name].append((timestamps, downsampled))

    signal_windows: dict[str, np.ndarray] = {}

    for sig_name in CANONICAL_SIGNALS:
        segments = signal_segments[sig_name]
        if not segments:
            continue

        all_times = pd.DatetimeIndex(np.concatenate([ts.values for ts, _ in segments]))
        all_values = np.concatenate([vals for _, vals in segments])

        sort_idx = all_times.argsort()
        all_times = all_times[sort_idx]
        all_values = all_values[sort_idx]

        filtered_values, filtered_times = apply_temporal_filter(all_values, all_times, outcome_time)
        if len(filtered_values) == 0:
            continue

        window_values, window_times = extract_observation_window(filtered_values, filtered_times)
        if len(window_values) == 0:
            continue

        signal_windows[sig_name] = window_values

    if not signal_windows:
        return None

    has_full_window = any(len(v) == WINDOW_SAMPLES for v in signal_windows.values())
    if not has_full_window:
        return None

    return compute_admission_features_v2(signal_windows)

def process_cohort(admissions_df: pd.DataFrame,
                   waveform_dir: str | Path,
                   verbose: bool = True) -> pd.DataFrame:
    waveform_dir = Path(waveform_dir)
    results = []
    skipped = 0
    total = len(admissions_df)

    for idx, row in admissions_df.iterrows():
        hadm_id = row['hadm_id']
        subject_id = row['subject_id']
        outcome_time = pd.Timestamp(row['dischtime'])
        mortality = row['mortality']

        features = process_admission(subject_id=subject_id,
                                     outcome_time=outcome_time,
                                     waveform_dir=waveform_dir)

        if features is not None:
            row_dict = {
                'hadm_id': hadm_id,
                'mortality': mortality,
            }
            for i, col_name in enumerate(get_feature_columns()):
                row_dict[col_name] = features[i]
            results.append(row_dict)
        else:
            skipped += 1

        if verbose and (len(results) + skipped) % 100 == 0:
            processed = len(results) + skipped
            print(f"  Processed {processed}/{total} admissions "
                  f"({len(results)} retained, {skipped} skipped)")

    if verbose:
        print(f"\nCohort processing complete:")
        print(f"  Total admissions: {total}")
        print(f"  Retained (≥24h window): {len(results)}")
        print(f"  Skipped (insufficient data): {skipped}")
        if results:
            mort_count = sum(1 for r in results if r['mortality'])
            print(f"  Mortality rate: {mort_count}/{len(results)} "
                  f"({mort_count / len(results) * 100:.1f}%)")

    return pd.DataFrame(results)

# Returns the ordered list of 317 dim feature column names
def get_feature_columns() -> list[str]:
    columns = []

    # 1. Segment features: seg{0-5}_{SIGNAL}_{stat}
    for sig in CANONICAL_SIGNALS:
        for seg_idx in range(N_SEGMENTS):
            for stat in STAT_NAMES:
                columns.append(f"seg{seg_idx}_{sig}_{stat}")

    # 2. Full-window features: full_{SIGNAL}_{stat}
    for sig in CANONICAL_SIGNALS:
        for stat in STAT_NAMES:
            columns.append(f"full_{sig}_{stat}")

    # 3. Interaction features: seg{0-5}_corr_{sigA}_{sigB}, full_corr_{sigA}_{sigB}
    for seg_idx in range(N_SEGMENTS):
        for sig_a, sig_b in INTERACTION_PAIRS:
            columns.append(f"seg{seg_idx}_corr_{sig_a}_{sig_b}")
    for sig_a, sig_b in INTERACTION_PAIRS:
        columns.append(f"full_corr_{sig_a}_{sig_b}")

    # 4. Delta features: delta_{SIGNAL}_{stat}
    for sig in CANONICAL_SIGNALS:
        for stat in DELTA_STATS:
            columns.append(f"delta_{sig}_{stat}")

    assert len(columns) == FEATURE_DIM, \
        f"Expected {FEATURE_DIM} columns, got {len(columns)}"

    return columns


def get_feature_matrix(features_df: pd.DataFrame) -> np.ndarray:
    """Extract the 317-dimensional feature matrix from a features DataFrame."""
    return features_df[get_feature_columns()].values