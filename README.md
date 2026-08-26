# PhysioML

Traceable multimodal physiological and neural inference — a companion machine-learning
layer for the [Clinical Data Fabric System](https://github.com/Elnazkarami/clinical-data-fabric-).

**The question this exists to answer:** can a physiological or neural prediction stay
traceable from the model output all the way back to the exact sensor windows,
transformations, features, model version, and source observations that produced it?

> **Status: peripheral signals working end to end.** Provenance spine, wrist-sensor
> quality control, preprocessing, 28 features, subject-wise evaluation and a first
> measured result on WESAD. EEG and fusion are not built. Every claim below is either
> implemented and covered by a test, or listed under *Not built* — nothing here is
> aspirational description of code that does not exist.

---

## The separation of concerns

CDFS remains the data-integrity, standards, audit and provenance engine. PhysioML owns
signal processing and machine learning. Scientific dependencies — NumPy, SciPy,
scikit-learn, MNE, PyTorch — live here and never enter the CDFS core.

```
CDFS  ──▶  canonical observations + identifiers + provenance
             ▼
        PhysioML  ──▶  QC + preprocessing + features + models + predictions
             ▼
CDFS  ◀──  derived ML facts + model provenance + source lineage
```

## What is built

`physioml.core` — the provenance spine — and `physioml.cdfs` — the client that reads
observations and writes predictions back.

| Type | Carries |
| --- | --- |
| `Recording` | A pointer to signal data: device, rate, channels, units, source hash, and the CDFS facts it belongs to |
| `SignalWindow` | A bounded slice, its preprocessing run, and a QC verdict with reason codes |
| `Feature` | One named value, its feature-set version, and the windows it came from |
| `FeatureVector` | An ordered set of features for one window |
| `TrainingRun` | Splits, strategy, seed, hyperparameters, metrics, code commit |
| `ModelArtifact` | What a model expects to be fed |
| `Prediction` | An output, plus both halves of its history |

### The loop, closed and tested

CDFS facts → windows → features → a prediction → a CDFS derived fact → a lineage query
that reaches the original observations. That chain is asserted end to end against a
**running CDFS deployment** rather than a mock, because a mock would only agree with
whatever this repository believed CDFS does.

Correcting an input on the CDFS side now produces an impact report naming the
prediction as stale — CDFS says what is wrong without pretending it can recompute
something it did not produce, which is the handoff PhysioML exists on the other side of.

**It has no runtime dependencies, and CI asserts that.** The chain of evidence is plain
Python, so it can be constructed and tested without installing a scientific stack — if
the provenance model ever needs NumPy to be exercised, it has grown into something else.

### Four things it refuses

These are the load-bearing behaviours, and each has a test:

- **A prediction that cannot name what produced it.** Missing model version, training
  run, or feature-set version is refused at construction. A prediction that cannot be
  accounted for is worse than none, because it will be believed.
- **A subject appearing in two splits.** Windows from one participant share that
  person's physiology, so subject-level leakage turns a reported score into a measure of
  subject recognition. Refused when the training run is created.
- **Features a model was not trained on** — including the *right* features in the
  *wrong order*, which otherwise scores without complaint and misaligns every column.
- **A rejected window with no reason code.** QC labels; it never silently filters, so
  "how much of this subject survived QC" stays answerable.
- **A prediction written back without CDFS fact ids.** Feature identifiers belong to
  PhysioML and mean nothing across the boundary; CDFS checks every id it is given
  exists, so a prediction cannot claim provenance it does not have.

### One design decision worth stating

A window's identity is the physical slice — recording, samples, preprocessing run — and
**not** its QC verdict. Rejecting a window does not change its identifier.

That is what makes cascade invalidation expressible: *"window W was rejected, therefore
every feature naming W is stale."* Had rejection minted a new identifier, features
computed while the window was still considered good would point at an identifier nothing
holds any more.

## The dataset

WESAD — 15 subjects, 24.1 hours of chest and wrist physiology, read straight out of the
2 GB archive without unpacking it. Unpacked it is roughly 17 GB, and a pipeline that
needs that much scratch space before it computes anything is one people run once.

All 15 subjects window in 68 seconds. Windows are cut by *time interval*, not sample
count, because the Empatica E4 samples BVP at 64 Hz, accelerometry at 32 and EDA and
temperature at 4 — windowing by count would silently give the modalities different
durations.

| | epochs | share |
| --- | ---: | ---: |
| baseline | 3,342 | 41.3% |
| meditation | 2,001 | 24.7% |
| stress | 1,814 | 22.4% |
| amusement | 934 | 11.5% |
| **labelled** | **8,091** | **47%** |
| unlabelled | 9,111 | transitions and recovery, excluded |

60-second windows, 5-second stride. The 53% unlabelled is not waste: it is the protocol
transitions plus the recovery periods WESAD's own guide says to ignore, and windows that
straddle a boundary. Those are produced and marked rather than dropped, so how many were
lost stays countable.

## Features, and one that was measured and removed

28 features per 60-second window: pulse rate and amplitude, electrodermal level, slope,
SCR count and area, skin-temperature level and slope, and accelerometry magnitude, jerk
and per-axis variation. Quality control runs first and a rejected signal contributes
nothing, because a heart rate from a flatlined sensor is not a missing value — it is a
confident wrong one.

**Pulse-rate variability is not among them.** SDNN, RMSSD and pNN50 were implemented,
then validated against the chest electrocardiogram WESAD records alongside the wrist, on
the same windows of the same subject:

| | chest ECG (truth) | wrist PPG |
| --- | ---: | ---: |
| heart rate | 73.0 bpm | 69.8 bpm — mean abs. error **7.1 bpm** |
| SDNN | 64.8 ms | 235.8 ms — **3.6× the real value** |

Band-passing the pulse and spacing peaks by the spectrally-estimated rate each improved
it; neither fixed it. The cause is not a threshold: at 64 Hz one sample is 15.6 ms, a
large fraction of the 20–60 ms the measure resolves, and every missed or doubled beat
enters squared.

So they are not emitted. They would have carried enough signal to *raise a model's score*
— being correlated with artifact rate and therefore with movement — while being
scientifically indefensible. A number wrong by a factor of four that looks useful is
worse than an absent one. Rate is kept, at an error in line with what an optical wrist
sensor gives.

## First result

WESAD, 15 subjects, wrist only. 60-second windows at 5-second stride give 8,091 windows;
34 are dropped for incomplete features, leaving **8,057 rows × 28 features**. Adjacent
windows overlap by 55 seconds, so rows within a subject are strongly correlated and the
row count is not a count of independent samples — which is why the evaluation below
splits by person and not by row. Quality
control marked 644 windows for motion, 34 for absent pulse and 24 for electrodermal
discontinuity. Binary stress against baseline, amusement and meditation — 22.2% positive.

Evaluation is **leave-one-subject-out**: fifteen folds, each trained on fourteen people
and tested on the one held out. Scaling is fitted inside the fold. No subject is ever on
both sides.

| model | bal. accuracy | F1 | AUC | PR-AUC | Brier | ECE | worst subject |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| majority class | 0.500 ±0.000 | 0.438 | 0.500 | 0.222 | 0.222 | 0.000 | 0.500 |
| **logistic regression** | **0.887 ±0.058** | 0.873 | 0.947 | 0.876 | 0.077 | 0.101 | **0.796** |
| linear SVM | 0.828 ±0.113 | 0.844 | 0.955 | 0.892 | 0.067 | 0.092 | 0.606 |
| random forest | 0.859 ±0.113 | 0.856 | **0.978** | **0.942** | 0.069 | 0.117 | 0.500 |
| gradient boosting | 0.852 ±0.109 | 0.854 | 0.976 | 0.938 | 0.082 | 0.086 | 0.504 |

The majority row is there so the others mean something: on a task that is 22% positive,
answering "baseline" every time is 78% accurate and 0.500 balanced-accurate.

**Logistic regression wins, and the ranking flips depending on which column you read.**
The ensembles have the better AUC — random forest separates the classes best of anything
here — but logistic regression has the higher balanced accuracy and half the fold-to-fold
spread (±0.058 against ±0.113).

The last column is why that matters. Random forest scores 0.978 AUC across the cohort and
**0.500 — chance — on subject S14**; gradient boosting gets 0.504 on the same person.
Logistic regression gets 0.818 there. A cohort mean hides a model that has learned
nothing at all about someone, which is exactly the failure a deployed physiological
classifier makes in front of a real user. Per-subject scores are reported for every fold
for that reason, and the worst one is carried into the summary rather than averaged away.

Calibration is mediocre across the board — 0.086 to 0.117 expected calibration error, so
a stated 90% is nearer 80%. That is recorded, not corrected; nothing here is calibrated
yet.

Simple models were run first to establish whether the engineered features carry usable
signal before anything deeper is justified. They do, and the simplest one is currently
the most trustworthy across people.

## Not built

EEG preprocessing, montage capability and features · multimodal fusion ·
probability calibration · modality and channel ablations · PhysioML-side cascade
invalidation · anything beyond the peripheral wrist signals.

## Install

```bash
pip install -e ".[dev]"          # core + tooling, no scientific stack
pip install -e ".[signal,ml]"    # signal processing and models
```

The core package has no runtime dependencies, and CI asserts that on every commit —
including that NumPy is absent from a `[dev]`-only install.

## Reproducing the result

```bash
pip install -e ".[signal,ml]"
python scripts/build_features.py ~/Downloads/WESAD.zip wesad_features.npz
python scripts/evaluate.py wesad_features.npz
```

Roughly 80 seconds to build the features and a couple of minutes to score all five
models. The archive is read as a stream and never extracted — 2.1 GB compressed against
about 17 GB unpacked.

## Non-goals for version 1

Large neural networks as the default · real-time streaming · mobile applications ·
automatic diagnosis or clinical decision-making · claims of equivalence between
fundamentally different EEG montages · a second provenance engine duplicating CDFS.

**Not for clinical use.**

---

© 2026 Elnaz Alikarami. All rights reserved.
