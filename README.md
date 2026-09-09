# PhysioML

Traceable multimodal physiological and neural inference — a companion machine-learning
layer for the [Clinical Data Fabric System](https://github.com/Elnazkarami/clinical-data-fabric-).

**The question this exists to answer:** can a physiological or neural prediction stay
traceable from the model output all the way back to the exact sensor windows,
transformations, features, model version, and source observations that produced it?

> **Status: peripheral and neural, both working end to end.** Provenance spine; quality
> control and features for a wrist band, a chest strap and a sleep montage; subject-wise
> evaluation, calibration, ablation and paired comparison; and a closed cascade with CDFS.
> Every claim here is implemented and covered by a test, or listed under *Not built*.
> Nothing is aspirational description of code that does not exist.

---

Two datasets, two tasks, one pipeline:

- **WESAD** — wrist and chest sensors, stress against baseline, 15 participants.
- **Sleep-EDF Expanded** — a scalp montage, five-stage sleep scoring, 76 participants,
  with a second cohort of 22 held out for external validation.

## What is in it

| | |
| --- | ---: |
| `core` — provenance: recordings, windows, features, runs, artifacts, predictions | 945 lines |
| `peripheral` — wrist and chest: windowing, quality control, preprocessing, 63 features | 1,844 |
| `neural` — sleep EEG: spectra, Hjorth parameters, quality control, 48 features | 508 |
| `io` — WESAD read from its archive, and a from-scratch EDF/EDF+ reader | 770 |
| `evaluation` — splits, metrics, ablation, coverage, paired comparison, personalisation | 1,479 |
| `models` — five classical models, and a recurrent one that reads a night as a sequence | 604 |
| `cdfs` — the round trip with the provenance engine | 394 |
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
| random forest | 0.672 ±0.102 | 0.656 ±0.122 | 0.762 | 0.323 |
| **random forest, with neighbouring epochs** | **0.707 ±0.127** | **0.693** | **0.789** | 0.325 |

κ 0.693 sits within the published range for feature-based staging under subject-wise
validation — the check that matters, since the pipeline is new and the task is not. Both
rows are 76 participants, one night each, held out one at a time.

## What the ablations show

**Movement carries much of the WESAD stress signal.** The accelerometer alone reaches
0.855 balanced accuracy against 0.898 for all 28 features, and removing it costs **0.054,
95% interval [−0.093, −0.016]** across participants. Signal quality alone — twelve columns
describing only how noisy the recording was — reaches **0.663**, because a third of stress
windows are flagged for motion against one per cent of baseline windows. The stress
condition has participants standing and talking, so the protocol is legible in the
measurement, and a score reported without testing for that cannot be interpreted.

**A chest strap does not improve on the wrist band.** Paired across the fourteen
participants both configurations can score, adding it is +0.010 with an interval of
[−0.073, +0.079] — while doubling the fold-to-fold spread and losing one participant
entirely to an amplifier that clipped during the stress condition. On this cohort the two
devices are indistinguishable, and that is the answer a hardware decision needs.

**Chance-level scores are usually thresholds, not ignorance.** One participant scores
0.500 balanced accuracy at an AUC of **1.000** — the model ranks every one of their
stressed windows above every calm one and labels them all negative. Reporting the ranking
beside the decision is what makes that legible; the remedy is a per-person operating point,
not a better model.

**One EEG derivation carries most of a sleep montage.** Fpz-Cz alone reaches κ 0.569
against 0.656 for all four channels across 76 participants. Chin electromyography
contributes nothing measurable: +0.000 with an interval of ±0.002.

**The sleep model carries to a different protocol.** Fitted on 76 Sleep Cassette
participants and scored on 22 Sleep Telemetry participants — a different recording setup
and partly a medicated population — agreement falls by **0.019 κ**, from 0.656 to 0.637.
With neighbouring epochs it reaches **0.680 on the held-out cohort, above the plain model's
0.656 within its own.**

**A recurrent model buys nothing on agreement and a great deal on the rare stages.**
Reading a whole night as a sequence gives the same κ as handing a random forest its
neighbours as columns — 0.708 either way, head to head [−0.024, +0.017] — while finding
**N1 nearly twice as often, 0.614 against 0.380**, at four times the compute. Which is
better depends on whether you want agreement with a scorer or the transitions themselves.

**Reading the neighbouring epochs is worth more than any model choice here.** A scorer
judges an epoch partly by what surrounds it; handing that context to the same random forest
raises κ from 0.656 to **0.693, improving 68 of 76 participants** — and lifts N1, the
transition stage every scorer struggles with, from 0.309 recall to 0.405.

→ **[How the results were validated](docs/validation.md)** — the checks every number here
passed, and why each one is necessary on this kind of data.

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

## Limitations

**This does not measure stress in general.** WESAD induces stress with a protocol that has
participants standing and speaking, so movement, speech and signal quality all track the
label; removing the accelerometer does not remove those influences from the sensors that
remain. The ablations identify the shortcuts rather than eliminating them.

**The cohorts are small.** Fifteen participants for stress, seventy-six for sleep. Several
comparisons have intervals wide enough that they establish nothing, and those are reported
as such rather than as small effects.

**External validation exists for sleep and not for stress.** Sleep results are fitted on
Sleep Cassette and scored on Sleep Telemetry — a different protocol, different
participants — but that is still one dataset family, one recording tradition, and one
scoring standard. The stress results have no equivalent and are validated only across
participants within WESAD. Sleep staging is scored around a sleep interval located by
the reference annotation, which is a benchmark scope rather than a demonstration of finding
sleep in an unrestricted recording.

## Reading order

| | |
| --- | --- |
| [How the results were validated](docs/validation.md) | The checks behind every number |
| [Stress on WESAD](docs/wesad-stress.md) | First result, and what the model is reading |
| [Calibration](docs/calibration.md) | Whether the probabilities mean anything, and a negative result |
| [A second device](docs/devices.md) | Performance, and coverage — the other half |
| [Sleep staging](docs/sleep.md) | Five stages, and one EEG derivation |
| [Datasets and features](docs/datasets.md) | Both cohorts, and a feature set that was removed |
| [Architecture](docs/architecture.md) | Provenance types and the CDFS round trip |
| [Reproducing](docs/reproducing.md) | Every number, from a command |

## Install

```bash
pip install -e ".[dev]"          # core + tooling, no scientific stack
pip install -e ".[signal,ml]"    # signal processing and models
```

The core package has no runtime dependencies and CI asserts it on every commit.
→ [full instructions and every reproduction command](docs/reproducing.md)

## Not built

**Frequency-domain heart-rate variability.** The low-frequency band starts at 0.04 Hz and
needs windows of several minutes to resolve; these are one minute long. The chest
electrocardiogram at 700 Hz would support it, but WESAD's condition blocks run five to
twenty minutes, so re-windowing at five would leave two or three stress windows per
participant — around forty across the cohort. The limit is the protocol, not the code.

**Anything beyond one modality at a time in the neural pipeline.** The sleep model reads
EEG, EOG and EMG as columns of one table; it does not learn a separate representation per
channel and combine them.


## Non-goals for version 1

Large neural networks as the default · real-time streaming · mobile applications ·
automatic diagnosis or clinical decision-making · claims of equivalence between
fundamentally different EEG montages · a second provenance engine duplicating CDFS.

**Not for clinical use.**

---

© 2026 Elnaz Alikarami. All rights reserved.
