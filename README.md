# PhysioML

Traceable multimodal physiological and neural inference — a companion machine-learning
layer for the [Clinical Data Fabric System](https://github.com/Elnazkarami/clinical-data-fabric-).

**The question this exists to answer:** can a physiological or neural prediction stay
traceable from the model output all the way back to the exact sensor windows,
transformations, features, model version, and source observations that produced it?

> **Status: early.** The provenance spine is built and tested. No signal processing, no
> models, and no results yet. Every claim below is either implemented and covered by a
> test, or listed under *Not built* — nothing here is aspirational description of code
> that does not exist.

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

`physioml.core` — the provenance spine, and the only part that exists so far.

| Type | Carries |
| --- | --- |
| `Recording` | A pointer to signal data: device, rate, channels, units, source hash, and the CDFS facts it belongs to |
| `SignalWindow` | A bounded slice, its preprocessing run, and a QC verdict with reason codes |
| `Feature` | One named value, its feature-set version, and the windows it came from |
| `FeatureVector` | An ordered set of features for one window |
| `TrainingRun` | Splits, strategy, seed, hyperparameters, metrics, code commit |
| `ModelArtifact` | What a model expects to be fed |
| `Prediction` | An output, plus both halves of its history |

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

### One design decision worth stating

A window's identity is the physical slice — recording, samples, preprocessing run — and
**not** its QC verdict. Rejecting a window does not change its identifier.

That is what makes cascade invalidation expressible: *"window W was rejected, therefore
every feature naming W is stale."* Had rejection minted a new identifier, features
computed while the window was still considered good would point at an identifier nothing
holds any more.

## Not built

Signal I/O and device adapters · peripheral physiology (BVP, EDA, temperature,
accelerometry) · EEG preprocessing, montage capability and features · multimodal fusion ·
models and calibration · subject-aware evaluation and ablations · the CDFS read and write
clients · cascade invalidation · any dataset, and therefore any result.

## Install

```bash
pip install -e ".[dev]"          # core + tooling, no scientific stack
pip install -e ".[signal,ml]"    # when the processing layers land
```

## Non-goals for version 1

Large neural networks as the default · real-time streaming · mobile applications ·
automatic diagnosis or clinical decision-making · claims of equivalence between
fundamentally different EEG montages · a second provenance engine duplicating CDFS.

**Not for clinical use.**

---

© 2026 Elnaz Alikarami. All rights reserved.
