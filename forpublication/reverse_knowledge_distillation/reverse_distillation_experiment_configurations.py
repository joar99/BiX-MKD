# Reverse-KD experiment configurations.
# =====================================
# Flip the arrow: each frozen teacher is re-inited and fine-tuned on the student's
# (train ∪ test) pool, with the multimodal student as the soft-label source.
#
# Every hyperparameter/architecture value comes from the single source of truth
# (config/experiment_config.py). This file only defines the reverse-KD
# *experiment grid* + checkpoint paths.

import sys
from pathlib import Path

_FORPUB = Path(__file__).resolve().parents[1]
if str(_FORPUB / "config") not in sys.path:
    sys.path.insert(0, str(_FORPUB / "config"))

from experiment_config import (                                    # noqa: E402
    TEXT_TEACHER as TEXT_TEACHER_REGIME,          # reverse fine-tune reuses the teacher's own regime
    WAVEFORM_TEACHER as WAVEFORM_TEACHER_REGIME,
    STUDENT_MODEL_CONFIG,                         # arch the student is reloaded with
    SEEDS,
    DEVICE, NUM_CLASSES, MAX_LENGTH, N_BOOTSTRAP,
)

# ── Experiment grid: each entry = one (alpha, temperature, soft-label source) ──
#   soft_label_source = None                → pure CE fine-tune (BIX2 baseline)
#   soft_label_source = "<student config>"  → KD against that student's soft labels
EXPERIMENT_CONFIGS = {
    "baseline": {
        "alpha": 0.0, "temperature": 1.0, "soft_label_source": None,
    },
    "kd_from_baseline_student_alpha_03_tau_3": {
        "alpha": 0.3, "temperature": 3.0, "soft_label_source": "multimodal_baseline",
    },
    "kd_from_kd_student_alpha_03_tau_3": {
        "alpha": 0.3, "temperature": 3.0, "soft_label_source": "kd_alpha_03_tau_3",
    },
}

CONFIGS_TO_RUN = [
    "baseline",
    "kd_from_baseline_student_alpha_03_tau_3",
    "kd_from_kd_student_alpha_03_tau_3",
]

# ── Original (frozen) teacher checkpoints — per-seed templates used as init.
# reverse-seed-i re-trains teacher-seed-i. "{seed}" filled by the orchestrator;
# resolved against PROJECT_ROOT there.
# "{source}" and "{seed}" filled by the orchestrator; matches the teacher
# write path checkpoints/{source}/{experiment}/seed{N}/.
ORIGINAL_TEACHER_PATHS = {
    "text_teacher":     f"text_teacher/checkpoints/{{source}}/"
                        f"{TEXT_TEACHER_REGIME['experiment_name']}/seed{{seed}}/best_model.pt",
    "waveform_teacher": f"waveform_teacher/checkpoints/{{source}}/"
                        f"{WAVEFORM_TEACHER_REGIME['experiment_name']}/seed{{seed}}/best_model.pt",
}

# Waveform scaler must come from the ORIGINAL waveform-teacher train split.
WAVEFORM_SCALER_COHORT = "waveform_teacher"

# Forward-KD multimodal student checkpoints (the soft-label source).
STUDENT_CKPT_TEMPLATE = (
    "forward_knowledge_distillation/checkpoints/{source}/seed{seed}/{config}/best_model.pt"
)

CONSTANTS = {
    "device":      DEVICE,
    "num_classes": NUM_CLASSES,
    "max_length":  MAX_LENGTH,
    "n_bootstrap": N_BOOTSTRAP,
}

TEACHERS = ["text_teacher", "waveform_teacher"]
