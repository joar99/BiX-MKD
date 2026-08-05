"""
SINGLE SOURCE OF TRUTH for every experiment configuration.
=========================================================
Every hyperparameter, architecture dimension, batch size, seed, and checkpoint
name is defined **exactly once** in this file. Everything downstream is either
this object or is *derived* from it — never re-typed. So the truth cannot
diverge between the text teacher / waveform teacher / student, or between the
forward and reverse directions.

Change a value here → it changes everywhere it is used. That is the whole point.

Layout:
  GLOBAL / SEEDS / DATA           — cross-cutting constants
  TEXT_TEACHER / WAVEFORM_TEACHER — the two teachers (BIX1), defined once each
  STUDENT                         — multimodal student arch + training, once
  *_derived*                      — configs built FROM the above (can't drift):
      TEACHER_MODEL_CONFIGS       (frozen-teacher shapes the student loads)
      STUDENT_MODEL_CONFIG        (student arch kwargs, for build + reload)
"""

# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL
# ═══════════════════════════════════════════════════════════════════════════
DEVICE               = "mps"        # ← switch to "cuda" on the gaming PC / cloud
NUM_CLASSES          = 2
MAX_LENGTH           = 512          # BERT token cap
WAVEFORM_FEATURE_DIM = 317
N_BOOTSTRAP          = 10000        # test-AUC bootstrap CI samples

# The 10 thesis seeds. Used for BOTH teachers (BIX1) and students (forward KD)
# and teacher fine-tunes (reverse KD), so every stage is seed-matched:
# student-seed-i distills from teacher-seed-i; reverse-seed-i re-trains
# teacher-seed-i from student-seed-i.  (seed 6666 is intentionally excluded.)
SEEDS = [42, 123, 256, 789, 1024, 1337, 2048, 3141, 4096, 5555]

# ═══════════════════════════════════════════════════════════════════════════
# DATA  (note leakage-filter — consumed by db/fetch_data.py)
# ═══════════════════════════════════════════════════════════════════════════
MIN_HOURS_BEFORE_DISCHARGE = 6.0
EXCLUDE_NOTE_CATEGORIES    = ("Discharge summary",)
AGGREGATION                = "last"   # one note per admission (the last one)

# ═══════════════════════════════════════════════════════════════════════════
# TEXT TEACHER  (BIX1) — checkpoint dir name = "pick_last_note_6_hour_window_buffer"
# ═══════════════════════════════════════════════════════════════════════════
TEXT_TEACHER = {
    "experiment_name":        "pick_last_note_6_hour_window_buffer",
    "model_name":             "emilyalsentzer/Bio_ClinicalBERT",
    "dropout":                0.1,
    "freeze_layers":          0,
    "learning_rate":          2e-5,
    "weight_decay":           0.01,
    "warmup_ratio":           0.1,
    "num_epochs":             8,
    "batch_size":             32,
    "patience":               None,        # 8 epochs flat, best-by-val-AUC
    "label_smoothing":        0.0,
    "scheduler_kind":         "linear_warmup",
    "aggregation":            AGGREGATION,
    "class_weights_strategy": "inverse_frequency_on_train",
}

# ═══════════════════════════════════════════════════════════════════════════
# WAVEFORM TEACHER  (BIX1) — checkpoint dir name = "waveform_strong_drop4_v2"
# ═══════════════════════════════════════════════════════════════════════════
WAVEFORM_TEACHER = {
    "experiment_name":        "waveform_strong_drop4_v2",
    "input_dim":              WAVEFORM_FEATURE_DIM,
    "hidden_dims":            [256, 128],
    "dropout":                0.4,
    "batch_norm":             True,
    "learning_rate":          5e-4,
    "weight_decay":           1e-3,
    "num_epochs":             40,
    "patience":               10,          # early stop on val AUC
    "batch_size":             64,
    "label_smoothing":        0.0,
    "scheduler_kind":         "plateau",   # ReduceLROnPlateau on val AUC
    "class_ratio":            2.0,         # pos/neg CE weight ratio
    "class_weights_strategy": "ratio_2",
}

# ═══════════════════════════════════════════════════════════════════════════
# MULTIMODAL STUDENT  (architecture + per-branch training)
# Values marked  ← teacher  are DERIVED from a teacher config above, so the
# student branch can never drift from the teacher it mirrors.
# ═══════════════════════════════════════════════════════════════════════════
STUDENT = {
    # ── architecture ──────────────────────────────────────────────────────
    "text_model_name":      TEXT_TEACHER["model_name"],       # ← teacher
    "text_proj_dim":        256,
    "text_dropout":         TEXT_TEACHER["dropout"],          # ← teacher
    "freeze_bert":          False,
    "freeze_bert_layers":   None,
    "waveform_input_dim":   WAVEFORM_FEATURE_DIM,
    "waveform_hidden_dims": WAVEFORM_TEACHER["hidden_dims"],  # ← teacher (same MLP)
    "wave_proj_dim":        128,
    "waveform_dropout":     WAVEFORM_TEACHER["dropout"],      # ← teacher
    "waveform_batch_norm":  WAVEFORM_TEACHER["batch_norm"],   # ← teacher
    "fusion_hidden_dim":    128,
    "fusion_dropout":       0.2,
    "num_classes":          NUM_CLASSES,
    # ── training (per branch: each branch inherits its teacher's regime) ───
    "learning_rate":          TEXT_TEACHER["learning_rate"],      # ← teacher (text branch)
    "weight_decay":           TEXT_TEACHER["weight_decay"],       # ← teacher
    "waveform_learning_rate": WAVEFORM_TEACHER["learning_rate"],  # ← teacher
    "waveform_weight_decay":  WAVEFORM_TEACHER["weight_decay"],   # ← teacher
    "fusion_learning_rate":   WAVEFORM_TEACHER["learning_rate"],  # ← teacher (small MLP → wave regime)
    "fusion_weight_decay":    WAVEFORM_TEACHER["weight_decay"],   # ← teacher
    "warmup_ratio":           TEXT_TEACHER["warmup_ratio"],       # ← teacher
    "num_epochs":             8,
    "patience":               None,
    "batch_size":             32,        # BERT memory keeps the student ≤ bs 128
    "aggregation":            AGGREGATION,
}

# ═══════════════════════════════════════════════════════════════════════════
# DERIVED configs — built from the definitions above. Do not edit by hand.
# ═══════════════════════════════════════════════════════════════════════════

# Frozen-teacher model shapes the student loads in forward KD (and reverse KD
# re-inits from). Derived from the teachers' own configs → cannot mismatch the
# checkpoints they produced.
TEACHER_MODEL_CONFIGS = {
    "text_teacher": {
        "model_name":  TEXT_TEACHER["model_name"],
        "num_classes": NUM_CLASSES,
        "dropout":     TEXT_TEACHER["dropout"],
    },
    "waveform_teacher": {
        "input_dim":   WAVEFORM_TEACHER["input_dim"],
        "hidden_dims": WAVEFORM_TEACHER["hidden_dims"],
        "num_classes": NUM_CLASSES,
        "dropout":     WAVEFORM_TEACHER["dropout"],
        "batch_norm":  WAVEFORM_TEACHER["batch_norm"],
    },
}

# Student architecture kwargs (for MultimodalStudent(**...)). Forward trains it,
# reverse reloads it — from the SAME slice of STUDENT, so they can't diverge.
_STUDENT_ARCH_KEYS = (
    "text_model_name", "text_proj_dim", "text_dropout", "freeze_bert",
    "freeze_bert_layers", "waveform_input_dim", "waveform_hidden_dims",
    "wave_proj_dim", "waveform_dropout", "waveform_batch_norm",
    "fusion_hidden_dim", "fusion_dropout", "num_classes",
)
STUDENT_MODEL_CONFIG = {k: STUDENT[k] for k in _STUDENT_ARCH_KEYS}
