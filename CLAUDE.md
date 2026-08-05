# MIMIC Bidirectional Knowledge Distillation — journal repo

> This file is the durable memory of a multi-day cleanup + methodology rewrite.
> The master thesis is **done and got an A**. We are now preparing a clean,
> self-contained repo (`forpublication/`) and a corrected methodology for a
> **Tier-2 journal** submission. Read this before touching anything.

---

## 1. What the research is

**Title theme:** Bidirectional Knowledge Distillation for Privacy-Restricted
(federated) Collaborative Learning.

Three **disjoint institutional cohorts** (no subject appears in more than one —
enforced by `fed_split.py`):

- **Text teacher** — BioClinicalBERT over clinical notes (unimodal).
- **Waveform teacher** — small MLP over 317-dim vital-sign features (unimodal).
- **Multimodal student** — fuses text + waveform. It is the **bridge / means to
  an end**, not the point of the paper.

**Forward KD:** the two frozen teachers → distill into the student.
**Reverse KD:** the student → re-train (fine-tune) the teachers. This is "the
collaboration": can two unimodal institutions teach each other *across a shared
multimodal bridge* without ever sharing raw data?

**Two research questions, two different test sets:**
1. *How much better do the unimodal institutions get by collaborating across the
   bridge?* → evaluate each reverse-fine-tuned teacher on **its own private
   cohort test set** (before vs after reverse). This is the reverse experiment's
   primary reported metric.
2. *Is the multimodal student actually better than the teachers?* (student-vs-
   teacher) → evaluate on the **shared/student test set**. That test must stay
   clean, which is why teachers are never fine-tuned on it.

---

## 2. The models + exact configs

**Single source of truth: `config/experiment_config.py`.** Every hyperparameter,
architecture dim, seed, and checkpoint name is defined **once** there. The
forward/reverse config files, both `create_baseline_model.py`s, and `fetch_data`
all *derive* from it (never re-type values). Change a value there → it changes
everywhere. Do not reintroduce hand-maintained "must match" duplication.

- **Text teacher** (`experiment_name = pick_last_note_6_hour_window_buffer`):
  BioClinicalBERT, dropout 0.1, freeze_layers 0, lr 2e-5, wd 0.01, warmup 0.1,
  8 epochs, batch 32, aggregation "last", inverse-frequency class weights,
  linear-warmup scheduler, no early stop (best-by-val-AUC).
- **Waveform teacher** (`experiment_name = waveform_strong_drop4_v2`):
  MLP hidden `[256, 128]`, dropout 0.4, batchnorm, lr 5e-4, wd 1e-3, 40 epochs,
  patience 10, batch 64, class ratio 2.0, plateau scheduler. Input 317-dim.
- **Multimodal student**: text branch (BERT → proj 256), waveform branch
  (`[256,128]` → proj 128), fusion 128 → 2. Dropouts: text 0.1, wave 0.4,
  fusion 0.2. Per-branch LR/WD inherit the teachers; batch 32, 8 epochs.
- **Forward KD grid**: `multimodal_baseline` (α0,τ1), `kd_alpha_03_tau_3`
  (the thesis config), plus sweep points. `CONFIGS_TO_RUN = [multimodal_baseline,
  kd_alpha_03_tau_3, kd_alpha_05_tau_1]`.
- **Reverse KD variants** (per teacher, per seed): `baseline` (pure CE, no KD),
  `kd_from_baseline_student_alpha_03_tau_3`, `kd_from_kd_student_alpha_03_tau_3`
  (both α0.3/τ3). Soft-label sources = 2 forward students: `multimodal_baseline`
  + `kd_alpha_03_tau_3`.

---

## 3. KEY METHODOLOGY DECISIONS (journal ≠ thesis) — with reasoning

1. **Per-seed MATCHED teachers ("KD-shuffle").**
   Seeds = `[42,123,256,789,1024,1337,2048,3141,4096,5555]` (10; **6666 was a
   bug, killed**). Each teacher is trained across all 10 seeds. In forward,
   **student-seed-i distills from text+waveform teacher-seed-i**. In reverse,
   **reverse-seed-i re-trains teacher-seed-i from student-seed-i's soft labels.**
   Thesis used ONE fixed teacher (seed 99999) frozen across all student seeds —
   **retired.** *Why:* remove the forward/reverse asymmetry so teacher variance
   flows into results the same way student variance does.
   Checkpoints: `{text,waveform}_teacher/checkpoints/{experiment_name}/seed{N}/best_model.pt`.

2. **No teacher calibration.**
   Thesis did Guo temperature scaling in forward (fit T on the *student's* val
   set). **Dropped entirely.** *Why (important):* under the federated premise,
   the **reverse** direction structurally *cannot* Guo-calibrate — it would need
   the teachers' validation data on the student/aggregator side, which breaks
   "teacher data stays locked down." So forward drops it too, keeping both
   directions symmetric. Teachers now use raw logits + Hinton τ only.

3. **Paired-only student cohort (deep-bolted in `fetch_data`).**
   The student cohort has **1,928 admissions but only 1,481 have BOTH modalities**
   (438 text-only, 4 waveform-only, 5 neither). **Patients may have MULTIPLE
   admissions** — `fed_split` assigns whole *patients* to a cohort (so no subject
   ever spans cohorts) but keeps every usable admission they have: measured
   mean 1.18 admissions/subject, max 23, 11.2% of subjects have >1.
   `get_multimodal_student_mimic_data(split)` **always returns only the paired
   1,481** — enforced at the single fetch gate so no consumer (student training,
   forward, reverse) can ever see unpaired data. The old paired-filter that lived
   inline in the reverse orchestrator was removed; `MultimodalStudentDataset`'s
   inner join stays (it's the *assembly* mechanism, now a guaranteed no-op).

4. **Reverse fine-tunes on student TRAIN ∪ VAL; TEST held out.**
   *Why:* the student test is the clean ground for the student-vs-teacher
   comparison, so teachers must never fine-tune on it. VAL is fair game now that
   calibration (which used to touch it) is gone. Teachers early-stop on their
   OWN cohort val and report on their OWN cohort test.

5. **DB-free.** The Postgres `mimic_thesis` DB was dumped to parquet in
   `db/mimic_data/` (`patients`, `admissions`, `notes`, `waveform_features_v2`).
   All experiment code reads via `db/fetch_data.py`. No `psycopg`/passwords in
   the pipeline — only in `preprocessing/` scripts that build/dump the DB.

---

## 4. Repo layout (`forpublication/`)

```
config/experiment_config.py     ← THE single source of truth
db/
  fetch_data.py                 ← Split enum + 3 get_*_mimic_data() methods (the data gate)
  mimic_data/*.parquet          ← patient data (GITIGNORE — never publish)
common/class_weights.py
text_teacher/       TextTeacher.py, TextTeacherDataset.py, TextTeacherTrainer.py, create_baseline_model.py
waveform_teacher/   WaveformTeacher.py, WaveformTeacherDataset.py, WaveformTeacherTrainer.py, create_baseline_model.py
multimodal_student/ MultimodalStudent.py, MultimodalStudentDataset.py, StudentTrainer.py (DistillationTrainer)
forward_knowledge_distillation/   *_configurations.py (grid+paths) + *_orchestrator.py (runner)
reverse_knowledge_distillation/   *_configurations.py + *_orchestrator.py
preprocessing/      fed_split.py (cohort+split logic, doc-only), dump_database.py,
                    waveform_data_preprocessing/ (feature extraction), README.md (full pipeline doc)
```

Conventions: **one trainer per model, in the model's folder** (self-contained,
duplication OK — see §7). Experiment folders hold only config + orchestrator.

---

## 5. Data flow

- `Split` enum (TRAIN/VAL/TEST) is **required, no default** — passing nothing / a
  raw string raises `TypeError` (no silent failures).
- Note leakage-filter (drop notes within 6h of discharge + "Discharge summary")
  is baked into the text-bearing fetch methods. Thresholds live in the config.
- Cohorts/splits come from `patients.cohort` / `patients.split` /
  `admissions.excluded`, all produced by `fed_split.py`.
- **Verified split counts** (usable admissions, train/val/test):
  text_teacher `2688/602/567`; waveform_teacher `2082/474/492`.

---

## 6. How to run (order matters — rerun from scratch)

1. `python text_teacher/create_baseline_model.py`  (loops 10 seeds → 10 ckpts)
   `python waveform_teacher/create_baseline_model.py`  (loops 10 seeds → 10 ckpts)
2. `python forward_knowledge_distillation/forward_distillation_experiment_orchestrator.py`
   (loads teacher-seed-i, trains 10 seeds × CONFIGS_TO_RUN students)
3. `python reverse_knowledge_distillation/reverse_distillation_experiment_orchestrator.py`
   (re-inits teacher-seed-i, fine-tunes on student train∪val)

**Model tally:** 20 baseline teachers (10 text + 10 wave) → 30 students
(10 seeds × 3 forward configs) → 60 re-trained teachers (2 teachers × 10 seeds ×
3 reverse variants).

---

## 7. Working conventions / gotchas

- **Imports:** modules use runtime `sys.path.insert(...)` to reach sibling
  folders. PyCharm therefore shows false "Unresolved reference" warnings — the
  code runs fine (verified). To silence: mark each module folder as *Sources
  Root* in PyCharm.
- **User preferences (respect these):** self-contained per-model code over shared
  abstractions (duplication is fine for human readability); **loud failures over
  silent** ones; config in exactly one place; keep replies concise; put questions
  in chat text, not bundled with commands the UI hides; write scripts and hand
  back the path — the user runs them.
- **Hardware:** `DEVICE = "mps"` (M3 Max dev box, 96GB). For the gaming PC
  (14700F / RTX 5070 Ti / 32GB) or cloud: set `DEVICE = "cuda"` in the config and
  add mixed precision (`torch.autocast` / `GradScaler`) to the trainers. Blackwell
  needs recent PyTorch + CUDA 12.8+. The datasets are small (few-thousand
  admissions) — CUDA+AMP will be faster than MPS; not memory-limited.
- **Licensing:** all patient data (parquet, notes, waveforms) is MIMIC/PhysioNet
  **credentialed** — gitignore `db/mimic_data/`, never publish. Code is
  publishable; data is regenerated from PhysioNet via `preprocessing/`.

---

## 8. Status / next step

Everything above is **implemented and verified at the import + path-consistency
level** (all modules import; teacher write-paths == orchestrator load-paths; no
stale refs; single-source config flows everywhere). It has **not** been run
end-to-end yet.

**Next:** a GPU smoke test — train a couple of teacher seeds → one forward seed →
one reverse cell — to confirm the seed-matched loading works against real
checkpoints. Then the full rerun.
