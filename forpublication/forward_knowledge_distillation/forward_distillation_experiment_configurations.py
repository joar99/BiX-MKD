# Forward-KD experiment configurations.
# ======================================
# Every hyperparameter/architecture value comes from the single source of truth
# (config/experiment_config.py). This file only defines the forward-KD
# *experiment grid* (alpha × tau) and the teacher checkpoint paths.

import sys
from pathlib import Path

_FORPUB = Path(__file__).resolve().parents[1]
if str(_FORPUB / "config") not in sys.path:
    sys.path.insert(0, str(_FORPUB / "config"))

from experiment_config import (                                    # noqa: E402
    STUDENT,
    TEACHER_MODEL_CONFIGS as TEACHER_CONFIGS,
    TEXT_TEACHER, WAVEFORM_TEACHER,
    DEVICE, NUM_CLASSES, MAX_LENGTH, WAVEFORM_FEATURE_DIM,
    SEEDS, N_BOOTSTRAP,                                             # re-exported for the orchestrator
)

# Student config = the single-source STUDENT (architecture + per-branch training).
_BASE = STUDENT


def _config(**overrides):
    """One forward-KD experiment = STUDENT + (alpha, temperature)."""
    return {**_BASE, **overrides}


# ── Experiment grid: alpha × tau (the only forward-KD-specific knobs) ──────
EXPERIMENT_CONFIGS = {
    "multimodal_baseline": _config(alpha=0.0, temperature=1.0),   # no KD
    # alpha sweep (tau=3)
    "kd_alpha_03_tau_3":   _config(alpha=0.3, temperature=3.0),   # thesis config
    "kd_alpha_05_tau_3":   _config(alpha=0.5, temperature=3.0),
    "kd_alpha_07_tau_3":   _config(alpha=0.7, temperature=3.0),
    # temperature sweep (alpha=0.5)
    "kd_alpha_05_tau_1":   _config(alpha=0.5, temperature=1.0),
    "kd_alpha_05_tau_5":   _config(alpha=0.5, temperature=5.0),
    "kd_alpha_05_tau_10":  _config(alpha=0.5, temperature=10.0),
}

# Which experiments actually run (a subset of EXPERIMENT_CONFIGS), in order.
# The reported forward-KD set: no-KD baseline + the reported KD config. Both are
# consumed downstream as reverse-KD soft-label sources. Sweep points stay defined
# in EXPERIMENT_CONFIGS above but are not run — they were never reported.
CONFIGS_TO_RUN = [
    "multimodal_baseline",
    "kd_alpha_03_tau_3",
]

# ── Frozen teacher checkpoints — per-seed templates ("{seed}" is filled by the
# orchestrator so student-seed-i distills from teacher-seed-i).
# "{source}" and "{seed}" are filled by the orchestrator so the load path matches
# create_baseline_model.py's write path: checkpoints/{source}/{experiment}/seed{N}/.
TEACHER_PATHS = {
    "text_teacher": str(_FORPUB / "text_teacher" / "checkpoints" / "{source}"
                        / TEXT_TEACHER["experiment_name"] / "seed{seed}" / "best_model.pt"),
    "waveform_teacher": str(_FORPUB / "waveform_teacher" / "checkpoints" / "{source}"
                            / WAVEFORM_TEACHER["experiment_name"] / "seed{seed}" / "best_model.pt"),
}

# ── Constants passed to the orchestrator (all from the single source) ──────
CONSTANTS = {
    "device":               DEVICE,
    "num_classes":          NUM_CLASSES,
    "max_length":           MAX_LENGTH,
    "waveform_feature_dim": WAVEFORM_FEATURE_DIM,
}
