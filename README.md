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

Two datasets, two tasks, one pipeline:

- **WESAD** — wrist and chest sensors, stress against baseline, 15 participants.
- **Sleep-EDF Expanded** — a scalp montage, five-stage sleep scoring, 76 participants.

## What is in it

| | |
| --- | ---: |
| `core` — provenance: recordings, windows, features, runs, artifacts, predictions | 945 lines |
| `peripheral` — wrist and chest: windowing, quality control, preprocessing, 63 features | 1,844 |
| `neural` — sleep EEG: spectra, Hjorth parameters, quality control, 48 features | 508 |
| `io` — WESAD read from its archive, and a from-scratch EDF/EDF+ reader | 770 |
| `evaluation` — splits, metrics, ablation, coverage, paired comparison, personalisation | 1,479 |
| `cdfs`, `models` — the round trip, and five classical models | 696 |
| **tests** | **4,456 lines, 350 tests** |

**The provenance core has no runtime dependencies at all** — no NumPy, no SciPy — and CI
asserts it on every commit by installing without them and running the core tests against
that. The signal processing is written here rather than imported: band-passing, spectral
rate estimation, R-peak detection, an EDF reader. The EDF reader is
[cross-checked](docs/reproducing.md) against the reference C library on the real
recordings, because a reader and a writer built by one person agree with each other about
a format neither may have got right.

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
| majority class | 0.206 ±0.016 | 0.000 | 0.377 | 0.200 |
| logistic regression | **0.695 ±0.111** | 0.607 ±0.135 | 0.721 | 0.227 |
| random forest | 0.674 ±0.102 | **0.656 ±0.124** | 0.765 | 0.310 |

κ 0.656 sits at the lower end of the published range for feature-based staging under
subject-wise validation — the check that matters, since the pipeline is new and the task
is not. **On the first 20 subjects it read 0.710**; the full cohort took it down, and took
the worst participant from 0.534 to 0.310. Twenty was not a small sample of this dataset,
it was an unrepresentative one: wake is 17% of those subjects and 34% of all of them.

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

### And a per-subject score that meant the opposite of what it said

Random forest scores **0.500 balanced accuracy — chance — on subject S14, at an AUC of
1.000.** It ranks every one of that participant's stressed windows above every calm one,
then labels all of them negative: the probabilities it states for them average 0.045
against a true rate of 0.223, so the whole distribution sits on one side of the threshold.

Nothing was failed to be learned. A threshold was in the wrong place for one person, and
across both datasets *every* chance-level per-subject result turned out to be the same
thing. Reporting only balanced accuracy had been calling these participants failures.

## How the results were checked

Each of these was added after something got through without it:

| | |
| --- | --- |
| **A majority-class row in every table** | It is the one model whose score is knowable by hand — and it is what exposed a calibration-metric bug that a reviewer spotted from an inconsistency between two numbers, without seeing the code. |
| **Paired intervals over participants** | 8,057 windows look like a large sample and are fifteen people. Two published claims did not survive this. |
| **Per-subject AUC beside accuracy** | See S14 above. |
| **Coverage beside performance** | The chest pipeline scores well and cannot produce any usable prediction for one participant in fifteen. |
| **A constant-probability baseline** | Expected calibration error is *minimised* by stating the base rate: a constant scores the best calibration in the table at an AUC of exactly 0.500. |
| **Manifests** | Three tables have been rebuilt here, so "were these computed on the same thing" is a live question. |

→ **[What survives measurement](docs/what-survives.md)** — supported, withdrawn, and not
established either way.

→ **[Defects found, and what each cost](docs/defects.md)** — eighteen, four of which
changed a published number.

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

The scores above are not evidence that this measures stress. Removing the accelerometer
does not remove movement, speech or artifact from the sensors that remain, and this
protocol has stressed participants standing and talking. Fifteen participants is a small
cohort and the intervals say so. Nothing here has been validated on a second session, a
different protocol, or anyone outside these two datasets.

The [ledger](docs/what-survives.md) keeps the withdrawn claims beside the surviving ones,
struck through rather than deleted.

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
