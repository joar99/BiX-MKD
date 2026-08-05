# eICU-CRD v2.0 — data plan & decision log

> Living record of the port from MIMIC-III to eICU, **with reasoning**.
> Two sections that must not be confused: **Decisions** (agreed by Jonas) and
> **Open** (not agreed — do not build on these). Append; don't silently rewrite.
>
> **Standing rule:** no change to the experiments without asking
> *"Jonas, are you ok with this change in the experiment?"* and receiving
> *"yes, i am ok with this change in the experiment"* verbatim. Until then the
> MIMIC design stands.

**Dataset:** eICU Collaborative Research Database v2.0 (PhysioNet,
DOI 10.13026/C2WM1R), credentialed. 31 `.csv.gz`, 5.1 GB compressed
(~35 GB if fully decompressed — leave the big ones gzipped).
Local: `forpublication/db/e-icu-data/physionet.org/files/eicu-crd/2.0/`
Schema: [`eicu_schema.dbml`](eicu_schema.dbml)

---

## 1. Verified facts (measured from the files — not inferred)

### Scale and grain
| quantity | value |
|---|---|
| ICU stays (`patientunitstayid`) — **unique per row** | 200,859 |
| hospital visits (`patienthealthsystemstayid`) | 166,355 |
| unique humans (`uniquepid`) | 139,367 |
| hospitals (`hospitalid`) | 208 |
| stays per hospital | median 557, mean 966, min 3, max 7,059 |
| ICU stays per visit | mean 1.21, max 17 |
| visits per patient | mean 1.19, max 24; 13.5% have >1 |
| patients spanning >1 hospital | 4,070 (2.92%) |

**`patient` is misnamed — it is one row per ICU stay, not per patient.**
This caused a real error early on; the table's name is not its grain.

### The two labels
- `hospitaldischargestatus` — describes the **hospital visit**. Denormalised onto
  every ICU-stay row of that visit (verified: 0 visits have a differing value
  across their stays). Mortality **9.04%**; 1,751 blank.
- `unitdischargestatus` — describes the **ICU stay**. Mortality **5.43%**;
  only 34 blank.
- 2,890 patients have visits with *different* outcomes, so a patient who dies in
  December is still correctly "Alive" for their January visit.
- Counting rows under the visit-level label overstated deaths by **1.23×**
  (18,004 Expired rows vs 14,609 Expired patients), because sicker patients
  accumulate more ICU stays.
- Data-quality wart: 14,623 Expired visits vs 14,609 Expired patients → 14
  patients "die twice". `uniquepid` linkage is imperfect.

### Text availability
- **eICU contains no clinical prose.** `note.notetext` is median 9 chars, p99 59,
  **max 101** — and the longest entries are boilerplate. Zero rows over 500
  chars. The eICU docs confirm narrative sections were *removed* during
  de-identification. `note` is UI exhaust ("Problem View", "Copies, 1").
- If text is assembled from the four structured tables, per-stay volume is:

| source | median chars/stay |
|---|---|
| `treatment` | 283 |
| `diagnosis` | 157 |
| `pastHistory` | 64 |
| `admissionDx` | 49 |
| **combined** | **534 (~134 tokens)**, p90 ~399 tokens |

  BERT's cap is 512 tokens, so it fits. 180,760 stays produce something;
  **9.1% come in under 50 tokens**.
- **`pastHistory` is person-level, not stay-level** (measured, not assumed):
  for patient `013-23495` it returned the identical 6 values on two different
  ICU stays. It repeats across a patient's stays.
- `admissionDx`, `treatment` and `diagnosis` **are** genuinely stay-specific —
  same patient showed "carotid endarterectomy" on one stay and "respiratory
  arrest / atropine" on the fatal one.
- Repetition inside a stay: `diagnosis` 74% repeat rows, `treatment` 70%.

### Vitals availability
- MIMIC's teacher uses `CANONICAL_SIGNALS = ["HR", "RESP", "SpO2", "PULSE"]`,
  10 stats per signal, `OBSERVATION_WINDOW_HOURS = 24`,
  `EXCLUSION_BUFFER_HOURS = 6`, 6 segments → `FEATURE_DIM = 317`.
  It reads real PhysioNet **waveform records** (`wfdb`, `record.p_signal`)
  downsampled to 1/minute.
- **eICU has no waveforms.** `vitalPeriodic` is the monitor writing every
  **5 minutes**; `vitalAperiodic` is intermittent (~15 min) and is blood
  pressure only — it contains none of the four signals.
- Fill rates in `vitalPeriodic`: `heartrate` 99.7%, `sao2` 86.6%,
  `respiration` 72.6%. Everything else is thin (invasive lines 4–28%,
  `icp` 0.5%).
- **PULSE does not exist anywhere in eICU.** Searched `nurseCharting` too — the
  only match is `Pulse Ox  Mode`, a device setting. So 3 of MIMIC's 4 signals,
  and the `("HR","PULSE")` interaction pair loses its partner.
- Coverage: `vitalPeriodic` 192,831 stays (96.0%). `nurseCharting` would add
  only 5,495 more, at the cost of mixing two recording sources.

### Timing / leakage
- All offsets are minutes from **that stay's own ICU admission** — verified:
  each stay's vitals restart near 0, and stop just before its
  `unitdischargeoffset` (3981 vs 4015; 5261 vs 5425).
- Under the *old* hospital-discharge anchor the 6h filter removed only ~2% of
  rows (diagnosis 2.54%, treatment 2.02%, pastHistory 2.12%) because hospital
  discharge sits a median 4.5 days after ICU admission.
- Median ICU stay is 1.6 days, so under the ICU anchor the filter should bite
  considerably harder. **Not yet measured.**
- 1,213 stays (0.6%) have `unitdischargeoffset <= 0`.
- **Empty stays exist.** Bed-transfer artifacts of 0 and 6 minutes returned zero
  rows from *every* table including vitals. Frequency across all 200k unknown.

---

## 2. Decisions (agreed)

### D1 — eICU-CRD v2.0 is the second dataset
Reviewers won't accept a single-dataset result. eICU is genuinely
multi-institutional (208 hospitals) where MIMIC-III is a single site, so the
federated premise stops being simulated.
*Accepted cost:* eICU is not like-for-like (no prose, no waveforms). This is a
**same-procedure replication**, and the paper should say so plainly.
Jonas: *"it will be a limitation honestly… it's the only viable dataset for this
so let's proceed."*

### D2 — Cohorts are split by hospital
Each role (text teacher / vitals teacher / student) gets a **disjoint set of
`hospitalid`s**. Jonas's proposal.
*Why:* cohorts become real institutions with different populations and practice
patterns, rather than a synthetic partition of one hospital.

Three conditions this carries:
1. **A hospital split does not give patient disjointness.** 4,070 patients
   (2.92%) have stays at more than one hospital. `fed_split` for eICU must
   resolve these explicitly and **fail loudly** if any `uniquepid` lands in two
   cohorts.
2. **Assign hospitals balanced, not randomly.** Sizes run 3→7,059 and
   per-hospital mortality runs 3.3%→13.5% (p10→p90, max 44%). Bin-pack on stay
   count *and* mortality, or between-site drift confounds the reverse-KD result.
3. **Freeze the assignment before modelling**, on coverage grounds only, blind
   to outcomes.

*Note:* non-IID cohorts are realistic for federated learning — characterise them
in a cohort table rather than glossing over it.

### D3 — The label is `unitdischargestatus` (ICU death), not `hospitaldischargestatus`
**Ratified explicitly** (exact confirmation string given).
*Why:* the task is defined at ICU-stay level because **all five source tables key
on `patientunitstayid`**. The visit-level label sat one level up, so a visit with
3 ICU stays produced 3 rows all marked "died" — one death counted three times,
inflating mortality and giving the sickest patients extra weight.
*In MIMIC the task was admission-level* because notes could not be aligned to a
specific ICU stay. In eICU they can, so the mismatch disappears.
*Consequences, all accepted:* mortality drops 9.04% → 5.43% (worse class
imbalance, relevant to MIMIC's class weighting); label coverage improves (34
blanks vs 1,751); the 6h filter anchor moves to `unitdischargeoffset`.
*Verified behaviour:* 441 visits have 3+ ICU stays reading Alive…Alive→Expired.
Example `013-23495`: survived a carotid-endarterectomy stay, returned in
respiratory arrest, died. Labels read correctly per stay.

### D4 — Table set, trimmed
Six tables. Everything not strictly required was killed to keep the build small.

| role | table | columns kept |
|---|---|---|
| cohort / label | `patient` | 5 |
| text | `admissionDx` | 4 |
| text | `pastHistory` | 4 |
| text | `diagnosis` | 4 |
| text | `treatment` | 3 |
| vitals | `vitalPeriodic` | 5 |

**Killed outright, with reasons:**
- `note` — no prose; UI exhaust.
- `hospital` — `hospitalid` already sits on `patient`; the rest (beds, teaching
  status, region) is paper-writing material, not experiment input.
- `apachePatientResult` — the most leakage-dense table in the database.
  `actualhospitalmortality` *is* the label; `apachescore` /
  `predictedhospitalmortality` are a purpose-built mortality predictor's answer.
  Would only be legitimate as a comparison baseline, which is reporting, not
  experiment.
- `vitalAperiodic` — contains none of the four MIMIC signals.
- `nurseCharting` — duplicate source of the same three signals, hand-charted.
  Buys 5,495 extra stays for 1.6 GB and a mixed recording source. Also the
  largest file in the dataset.
- `patienthealthsystemstayid` — only the old visit-level label needed it.

**Column-level cuts:** every surrogate row ID (`admissiondxid`, `pasthistoryid`,
`diagnosisid`, `treatmentid`, `vitalperiodicid`); `admitdxtext` (99.5% a copy of
`admitdxname`; the 0.5% that differ are bare APACHE numbers reading "specified
40"); `pasthistoryvaluetext` (same string with spaces stripped);
`pasthistorynotetype`; `icd9code` (codes mean nothing to a reading model, 15.9%
blank); `activeupondischarge` on both `diagnosis` and `treatment` (decided at
discharge, so not knowable during the stay).

### D5 — PULSE is dropped; 3 of MIMIC's 4 signals
It does not exist in eICU. Accepted by Jonas. The `("HR","PULSE")` interaction
pair is therefore unavailable.

### D6 — The text teacher reads *manufactured* text (**ratified**)
eICU has no clinical prose, so the corpus is assembled from four structured
tables: `admissionDx`, `pastHistory`, `diagnosis`, `treatment`.
Jonas's reasoning: *"we have no other way of training our text teacher."*
The BioClinicalBERT architecture is unchanged — only the corpus differs.

### D7 — How the note is assembled
1. **`pastHistory` is included.** Comorbidity burden predicts mortality and it
   costs ~64 chars. It repeats across a patient's stays, which is harmless
   because the cohort split keeps a patient inside one cohort.
2. **Full taxonomy path, `|` converted to `, `.** Leaf-only is unsafe:
   `cardiovascular|chest pain / ASHD|coronary artery disease|known` has the leaf
   `known`. Commas are in BioClinicalBERT's pretraining distribution; pipes are
   not.
3. **Dedupe within a stay.** `diagnosis` is 74% repeat rows, `treatment` 70%.
   Repetition carries nothing for attention and burns the token budget.
4. **Section headers** — real notes have them, and *"Past history: dementia"*
   differs in meaning from *"Active problems: dementia"*.
5. **Order: admission reason → active problems → treatments → past history.**
   Ordering controls **truncation, not emphasis** — BERT keeps the first 512
   tokens; position does not increase attention weight when everything fits.
   Past history goes last precisely because it is the most generic and most
   repeated, so it is the right thing to lose first.
6. **Empty-note stays are dropped at BUILD time** — the notes parquet excludes
   them, for symmetry with the vitals features parquet (**agreed; supersedes the
   earlier "keep in parquet, drop at fetch layer" wording**). Same cohort either
   way; only the location moved. Matches MIMIC's *effect*: MIMIC has no character
   threshold (verified — none in config, fetch_data, or any text_teacher file);
   its `TextTeacherDataset._aggregate_notes` inner-joins notes against
   admissions, so an admission is kept iff it has ≥1 note surviving the leakage
   filter. `note == ""` (28,194 stays, 14.0%) dropped, no minimum length.

**Leakage filter:** 6 hours, anchored on `unitdischargeoffset`, applied at build
time. It drops 4.2–8.5% of rows (3–4× harder than the old hospital anchor) but
**only 1.6% of tokens** — because what it removes is overwhelmingly re-charting
of items already recorded earlier, which dedupe was collapsing anyway. Unusually
cheap protection.

### D8 — Navigation scaffolding is stripped from the paths (**ratified**)
eICU stores each item as the trail of clicks used to reach the form
(`notes / Progress Notes / Past History / Organ Systems / Endocrine (R) / …`).
Those leading steps are identical on every entry for every patient, so they
carry no discriminative signal while consuming ~22% of every note.

Stripped: `admission diagnosis`, `All Diagnosis`, `Organ System`, `notes`,
`Progress Notes`, `Past History`, `Organ Systems`.

**Deliberately not stripped:** `Elective` and `Non-operative/Operative Organ
Systems` — these are clinically meaningful (elective vs emergency, surgical vs
medical), and removing `Elective` would leave a bare `Yes` with nothing to
attach it to. `diagnosis` and `treatment` need no equivalent; their taxonomies
are medicine all the way up.

Measured effect: median 1008 → 784 chars, median 261 → 203 tokens, p90 567 → 493
tokens (now under the cap), notes over 512 tokens 11.1% → 7.8%.

### D9 — Observation window is **12 hours**, not MIMIC's 24 (**ratified**)
`OBSERVATION_WINDOW_HOURS = 12`, `EXCLUSION_BUFFER_HOURS = 6` unchanged.

*Why the deviation is forced:* eICU ICU stays are structurally far shorter than
MIMIC hospital admissions (median ~38h). MIMIC's 24h window plus the 6h buffer
requires **30 hours** of stay, which only 56.9% of eICU stays have — and the
excluded stays are not a random sample.

**Attrition analysis (the evidence, and the answer to any reviewer):**

| window | eligible | % kept | retained mortality | excluded mortality | **deaths kept** |
|---|---|---|---|---|---|
| 6h | 172,858 | 86.1% | 4.84% | 9.08% | 76.7% |
| **12h** | **156,622** | **78.0%** | **4.87%** | **7.43%** | **69.9%** |
| 18h | 132,892 | 66.2% | 5.21% | 5.86% | 63.5% |
| 24h | 114,278 | 56.9% | 5.51% | 5.33% | 57.7% |
| 36h | 91,631 | 45.6% | 5.96% | 4.99% | 50.1% |
| 48h | 68,163 | 33.9% | 6.94% | 4.65% | 43.4% |

- **24h would discard 42% of all deaths.** 12h discards 30%; 6h discards 23%.
- **The bias flips direction around 18–24h.** Short ICU stays are two mixed
  populations — patients who died fast, and patients who were fine and stepped
  down fast. A short window excludes mostly rapid deaths (excluded group 9.08%
  mortality); a long window excludes mostly routine one-day survivors, so the
  *retained* group becomes the sicker one (6.94% vs 4.65% at 48h).
- **Why not 6h:** MIMIC splits the window into `N_SEGMENTS = 6`. At eICU's 5-min
  sampling, 12h gives 24 samples per segment — enough for all ten stats. 6h
  gives 12, where `skewness` and `iqr` become noise. 12h is the knee.
- **Inherent, not introduced:** no window avoids this. A patient who died at
  hour 8 cannot have a 24-hour observation window. Any prospective ICU-mortality
  model has this property. It must be stated in the write-up that the retained
  cohort's mortality (4.87%) differs from the population's (5.43%).

*Agreed follow-up:* re-run at 24h later as a **sensitivity analysis**, so the
result is shown to hold under both windows rather than defending one choice.

### D10 — Differential attrition by observation window does NOT bind in eICU
Checked because it mattered in the thesis: in MIMIC, lengthening the window cost
different vital signs different amounts of coverage. **It does not happen here.**
The heart-rate-vs-respiration completeness gap is 11.7% at 6h, 11.6% at 12h,
11.7% at 24h and 11.7% at 48h — completely flat.

*Why:* MIMIC waveforms are continuous recordings that start, stop and drop out,
so gaps compound over a longer window. eICU writes a `vitalPeriodic` row every
5 minutes with whatever the monitor has, and whether respiration is derived at
all is a **per-patient property** — on or off for that patient. A longer window
just samples more of the same steady state.

Within-window completeness (stable across all windows):
`heartrate` ~94%, `sao2` ~83%, `respiration` ~83%.

### D11 — Paired-only cohort: cut every stay lacking both modalities (**agreed**)
Keep only ICU stays present in BOTH build outputs (text notes ∩ vitals features)
= 148,135 stays. Applies to ALL three roles, not just the student.
*Jonas's call, on authority as methodology owner — explicitly NOT an experiment
change:* MIMIC kept unpaired unimodal data only because waveforms were scarce and
the cohort was tiny. With ~148k paired stays (~100× the MIMIC student cohort)
there is no scarcity pressure, and paired-only makes every hospital uniform so
the federated split is clean.
*Accepted cost:* teachers train on paired patients only; no-vitals stays (which
skew lower-acuity — 2.9% vs 4.9% mortality) drop out. One sentence in the write-up.

### D12 — Federated split is hospital-level, for BOTH role and train/val/test (**ratified**)
Script: [`preprocessing/eicu/fed_split.py`](../preprocessing/eicu/fed_split.py) →
`db/e-icu-data/eicu_fed_split.parquet` (one row per stay: ids, hospital,
mortality, role, split). One mortality-stratified greedy bin-packer, applied twice:
- **Roles:** text_teacher / waveform_teacher / student at **40 / 40 / 20** of stays.
- **Train/val/test within each role:** **70 / 15 / 15** of stays, split by whole
  HOSPITALS — so each model is tested on hospitals it never trained on (true
  cross-institution generalisation). MIMIC is single-site and *could not* do
  this; it is eICU structure MIMIC lacked, not a reversal of a MIMIC choice.

*Patient disjointness:* patients spanning >1 hospital (2.42% of paired) are
**dropped** (eICU has no reliable cross-hospital time order). Run fails loudly if
any patient/hospital spans two cells. Survivors: 140,148 stays / 114,315 patients
/ 205 hospitals.

*Balance achieved (whole-hospital packing → close, not exact):*

| role | stays | mortality tr/val/test | hosp tr/val/test |
|---|---|---|---|
| text_teacher | 56,048 (40.0%) | 4.76 / 6.47 / 5.05% | 62 / 9 / 12 |
| waveform_teacher | 56,064 (40.0%) | 4.89 / 6.20 / 6.04% | 65 / 9 / 10 |
| student | 28,036 (20.0%) | 5.08 / 4.33 / 5.14% | 32 / 3 / 3 |

Teachers hit 70/15/15 almost exactly; student is looser (66/17.5/16 — few
hospitals). Accepted soft spots (packing granularity, not methodology): teacher
VAL runs ~1.3–1.7pp hotter (VAL only drives early-stopping); student VAL/TEST are
3 hospitals each. Jonas: *"some deviations are to be expected when you're packing
them by hospital to make the federation cleaner."* Split is fixed across the 10
seeds (only model init varies, as in MIMIC).

---

## 2b. The corpus as built

Script: [`preprocessing/eicu/build_text_notes.py`](../preprocessing/eicu/build_text_notes.py)
Outputs: `db/e-icu-data/eicu_text_notes.parquet`, `db/e-icu-data/eicu_patients.parquet`

| | |
|---|---|
| ICU stays | 200,859 |
| with a note | 172,665 (86.0%) |
| empty note | 28,194 (14.0%) |
| median / p90 tokens | 203 / 493 |
| over BERT's 512 cap | 15,733 (7.8%) |
| section coverage | reason 84.6%, problems 82.5%, treatments 72.8%, history 70.4% |

Token basis: **3.86 chars/token**, calibrated against the real BioClinicalBERT
tokenizer on a 3,000-note sample — not the chars/4 rule of thumb.

**The split can come later.** Text has no fitted preprocessing (the tokenizer is
pretrained), so notes can be built for every stay and assigned to cohorts
afterwards. The **vitals** side is *not* like this — MIMIC fits a scaler on train
only, so that one does depend on the split existing first.

**Two predictions Claude got wrong, recorded so they aren't repeated:**
- Predicted median ~134 tokens; actual 261 before the strip, 203 after. The
  full-path choice (D7.2) roughly doubled it, and the earlier estimate was
  measured on bare values before that choice was made.
- Predicted "well under 10%" would overflow 512; actual 11.1% before the strip.

---

## 3. Open — NOT agreed, do not build on these

- **Wiring the eICU parquets into `db/fetch_data.py`.** Deliberately deferred —
  the three built parquets (`eicu_text_notes`, `eicu_vitals_features`,
  `eicu_fed_split`) are standalone for now.

*Resolved since first draft:* vitals feature recipe → built at `input_dim` **236**
(3 signals, 2 interaction pairs — D3/D5/D9); hospital→role assignment and
hospital-level train/val/test → **D12**; empty-text handling → **D7.6**.

**Observed, accepted as-is (not currently a problem, but recorded):**
- `pastHistory` duplicates its own value: the path's last segment and the value
  are the same string differing only in case, so the section reads *"Peripheral
  Vascular Disease, peripheral vascular disease"*. That section is roughly double
  the length it needs to be. Jonas: use it as-is.
- Truncation is **not random**. Longer ICU stays accumulate more treatments, so
  they are the notes that overflow — and stay length correlates with severity.
  The notes losing their past-history section are disproportionately the sicker
  ones. Worth stating in the write-up.

---

## 4. The schema as it stands

```
Table patient {
  patientunitstayid    int [pk]
  uniquepid            varchar   // the actual person
  hospitalid           int       // which hospital
  unitdischargestatus  varchar   // died or lived through this ICU stay — the label
  unitdischargeoffset  int       // when they left the ICU, in minutes
}

// These four are the raw material for the text side
Table admissionDx {
  patientunitstayid    int [ref: > patient.patientunitstayid]
  admitdxenteredoffset int      // for the 6h filter
  admitdxpath          varchar  // what the value means; "Yes" is meaningless alone
  admitdxname          varchar  // the content
}
Table pastHistory {
  patientunitstayid        int [ref: > patient.patientunitstayid]
  pasthistorypath          varchar  // separates real history from the admin branch
  pasthistoryvalue         varchar  // the content
  pasthistoryenteredoffset int      // for the 6h filter
}
Table diagnosis {
  patientunitstayid  int [ref: > patient.patientunitstayid]
  diagnosisstring    varchar
  diagnosisoffset    int
  diagnosispriority  varchar  // Primary / Major / Other
}
Table treatment {
  patientunitstayid  int [ref: > patient.patientunitstayid]
  treatmentstring    varchar
  treatmentoffset    int
}

// Vitals
Table vitalPeriodic {
  patientunitstayid int [ref: > patient.patientunitstayid]
  observationoffset int      // timestamp; also drives the 6h filter
  heartrate         int      // HR
  sao2              int      // SpO2
  respiration       varchar  // RESP  (parses as numeric — varchar came from blanks)
}
```

**Filtering notes that must survive into the loader:**
- `pastHistory` — drop the `Past History Obtain Options` branch of the path
  (26% of rows: "Performed" alone appears 278,271 times).
- `admissionDx` — `admitdxname` is often just `Yes`/`No`; the path says what the
  answer refers to. Real diagnoses sit under the `All Diagnosis` branch.

---

## 5. Corrections made to other project files
- **`CLAUDE.md` §3 said "1 admission per patient" — that was wrong** and has been
  fixed. Measured from the MIMIC parquet: 9,640 usable admissions across 8,182
  subjects, mean 1.18 admissions/subject, max 23, **11.2% of subjects have more
  than one**. `fed_split` assigns whole *patients* to a cohort; it never reduces
  them to a single admission.

---

## 6. Housekeeping
- **There is no `.gitignore` anywhere in the project.** 5.1 GB of credentialed
  eICU data plus the MIMIC parquet sit in a directory intended for publication.
  Nothing has been pushed yet. Fix before `git init`.
- Plan: once features are built, move the finished files to Proton Drive to run
  the experiments on the stationary PC (14700F / RTX 5070 Ti). Viability not yet
  assessed.

## 7. Next step
The text corpus is **built** (§2b). The two things standing between here and a
trainable eICU pipeline are both in §3:
1. **The vitals features** — rebuild MIMIC's segment/stat/delta recipe on
   `vitalPeriodic` with 3 signals instead of 4, and settle the resulting
   `input_dim`.
2. **The hospital→role assignment** — which hospitals feed the text teacher, the
   vitals teacher and the student (D2's conditions apply: balance on stay count
   and mortality, enforce patient disjointness, freeze before modelling).
