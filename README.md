# PhysioML

Traceable multimodal physiological and neural inference — a companion machine-learning
layer for the [Clinical Data Fabric System](https://github.com/Elnazkarami/clinical-data-fabric-).

**The question this exists to answer:** can a physiological or neural prediction stay
traceable from the model output all the way back to the exact sensor windows,
transformations, features, model version, and source observations that produced it?

> **Status: peripheral and neural, both working end to end.** Provenance spine; quality
> control and features for a wrist band, a chest strap and a sleep montage; subject-wise
> evaluation, calibration, ablation and paired comparison; and a closed cascade with CDFS.
> Every claim here is implemented and covered by a test, listed under *Not built*, or
> listed as [withdrawn](docs/what-survives.md) — nothing is aspirational description of
> code that does not exist, and nothing that failed measurement has been quietly removed.

---

## What this is

A machine-learning layer for physiological and neural signals in which **every prediction
can be traced back to the exact windows, transformations, features, model version and
source observations that produced it**. Two datasets, two tasks, one pipeline:

- **WESAD** — wrist and chest sensors, stress against baseline, 15 participants.
- **Sleep-EDF Expanded** — a scalp montage, five-stage sleep scoring, 20 participants.

It is a companion to [CDFS](https://github.com/Elnazkarami/clinical-data-fabric-), which
owns data integrity and provenance; the scientific dependencies live here and never enter
that core.

## Results

**Stress, WESAD, leave-one-subject-out** — [full report](docs/wesad-stress.md)

| model | bal. accuracy | macro F1 | AUC | Brier | ECE | worst subject |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| majority class | 0.500 ±0.000 | 0.438 | 0.500 | 0.222 | 0.222 | 0.500 |
| **logistic regression** | **0.898 ±0.056** | 0.882 | 0.954 | 0.070 | 0.090 | **0.766** |
| random forest | 0.854 ±0.113 | 0.851 | 0.976 | 0.069 | 0.115 | 0.500 |
| gradient boosting | 0.849 ±0.110 | 0.847 | **0.977** | 0.085 | 0.089 | 0.504 |

**Sleep staging, Sleep-EDF, leave-one-subject-out** — [full report](docs/sleep.md)

| model | bal. accuracy | Cohen's κ | accuracy | worst subject |
| --- | ---: | ---: | ---: | ---: |
| majority class | 0.200 ±0.000 | 0.000 | 0.440 | 0.200 |
| logistic regression | **0.737 ±0.104** | 0.662 ±0.148 | 0.748 | 0.420 |
| random forest | 0.725 ±0.070 | **0.710 ±0.090** | 0.793 | 0.534 |

κ 0.710 sits inside the published range for feature-based staging under subject-wise
validation, which is the check that matters: the pipeline is new, the task is not.

## The finding the scores are for

**Movement carries much of the WESAD stress signal.** The accelerometer alone reaches
0.855 against 0.898 for all 28 features, and removing it costs **0.054, 95% interval
[−0.093, −0.016]** across participants. Signal quality alone — twelve columns describing
only how noisy the recording was — reaches **0.663**. The stress condition has
participants standing and talking, so the protocol is legible in the measurement, and any
score reported without testing for that is uninterpretable.

**And two claims did not survive being measured.** Adding a chest strap is +0.010 with an
interval of [−0.073, +0.079]; personal calibration on a person's own data is *worse* than
cohort calibration. Both were published here before they were tested properly.

→ **[What survives measurement](docs/what-survives.md)** — the ledger of what is
supported, what is withdrawn, and what is not established either way.

→ **[Defects found, and what each one cost](docs/defects.md)** — eighteen of them, four
of which changed a published number.

## Traceability, closed and tested

```
weight corrected  →  CDFS supersedes the BMI it derived
                  →  impact report: the prediction is stale, recompute it
                  →  PhysioML finds its inputs moved, and what they moved to
                  →  recomputed value written back, superseding the old one
                  →  one prediction in force, and a chain a reviewer can read
```

Asserted against a running CDFS deployment rather than a mock — including that a
correction leaves exactly one prediction in force afterwards. → [architecture and the
loop](docs/architecture.md)

## What is not claimed

The scores above are not evidence that this measures stress. WESAD's stress condition has
participants standing and speaking, so movement, speech and signal quality all track the
label, and removing the accelerometer does not remove those influences from the remaining
sensors. Two things that were published here have since been withdrawn under measurement,
and one — personal calibration — fails outright when run in the only order a deployment
could use it. The [ledger](docs/what-survives.md) keeps all of it.

## Reading order

| | |
| --- | --- |
| [What survives measurement](docs/what-survives.md) | Supported, withdrawn, undetermined |
| [Stress on WESAD](docs/wesad-stress.md) | First result, and what the model is reading |
| [Calibration](docs/calibration.md) | Whether the probabilities mean anything, and a negative result |
| [A second device](docs/devices.md) | Performance, and coverage — the other half |
| [Sleep staging](docs/sleep.md) | Five stages, and one EEG derivation |
| [Datasets and features](docs/datasets.md) | Both cohorts, and a feature set that was removed |
| [Architecture](docs/architecture.md) | Provenance types and the CDFS round trip |
| [Defects](docs/defects.md) | Eighteen of them, and what each cost |
| [Reproducing](docs/reproducing.md) | Every number, from a command |

## Install

```bash
pip install -e ".[dev]"          # core + tooling, no scientific stack
pip install -e ".[signal,ml]"    # signal processing and models
```

The core package has no runtime dependencies and CI asserts it on every commit.
→ [full instructions and every reproduction command](docs/reproducing.md)

## Not built

Sequence models, which is where sleep staging gains most — a scorer reads the epochs
before and after, and every model here sees one epoch alone · frequency-domain heart-rate
variability, which needs windows longer than the minute these are · deep architectures,
which the ablations do not yet justify · the Sleep Telemetry cohort, and the 58 Sleep
Cassette subjects not downloaded here.


## Non-goals for version 1

Large neural networks as the default · real-time streaming · mobile applications ·
automatic diagnosis or clinical decision-making · claims of equivalence between
fundamentally different EEG montages · a second provenance engine duplicating CDFS.

**Not for clinical use.**

---

© 2026 Elnaz Alikarami. All rights reserved.
