# Architecture and the CDFS loop

How PhysioML is put together, what the provenance types carry, and the round trip with
[CDFS](https://github.com/Elnazkarami/clinical-data-fabric-) that closes a correction.
## The separation of concerns

CDFS remains the data-integrity, standards, audit and provenance engine. PhysioML owns
signal processing and machine learning. Scientific dependencies — NumPy, SciPy,
scikit-learn — live here and never enter the CDFS core. (Nothing deeper is installed:
there is no neural-network code in this repository, so listing a deep-learning framework
among its dependencies would describe an intention rather than the software.)

```
CDFS  ──▶  canonical observations + identifiers + provenance
             ▼
        PhysioML  ──▶  QC + preprocessing + features + models + predictions
             ▼
CDFS  ◀──  derived ML facts + model provenance + source lineage
```

## What is built

`physioml.core` — the provenance spine — and `physioml.cdfs` — the client that reads
observations and writes predictions back. Around them, `physioml.io` reads the two archive
formats, `physioml.peripheral` and `physioml.neural` turn signals into features, and
`physioml.evaluation` scores them subject by subject.

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

And back the other way, which is the half that makes it a loop:

```
weight corrected  →  CDFS supersedes the BMI it derived from it
                  →  impact report: the prediction is stale, recompute it
                     in the system that produced it
                  →  PhysioML finds its inputs moved, and what they moved to
                  →  recomputed value written back, superseding the old one
                  →  one prediction in force, and a chain a reviewer can read
```

CDFS says what is wrong without pretending it can recompute something it did not
produce. PhysioML asks whether the facts its predictions were computed from are still
the ones in force — a prediction resting on a retracted value should not be read as
current, however plausible it still looks — and writes the recomputation back as a
**replacement**, not a second opinion. Both halves are asserted against a running
deployment, including that a correction leaves exactly one prediction in force
afterwards and that its history runs from the retracted value to the one that stands.

The replacement carries a reason naming the input that moved (`bmi 40.78 → 44.97`), not
"model rerun". CDFS requires a reason on anything that supersedes a fact. That is a CDFS
design decision taken in the spirit of 21 CFR 11.10(e), which requires audit trails that
preserve prior values and record who changed what and when; the regulation does not itself
prescribe a free-text reason field. The requirement goes beyond it deliberately, because
the input that changed is the answer to the question a reviewer is actually asking.

**It has no runtime dependencies, and CI asserts that.** The chain of evidence is plain
Python, so it can be constructed and tested without installing a scientific stack — if
the provenance model ever needs NumPy to be exercised, it has grown into something else.

### Five things it refuses

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


---

[← back to the README](../README.md)
