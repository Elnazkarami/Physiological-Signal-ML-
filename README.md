# PhysioML

Traceable multimodal physiological and neural inference — a companion machine-learning
layer for the [Clinical Data Fabric System](https://github.com/Elnazkarami/clinical-data-fabric-).

**The question this exists to answer:** can a physiological or neural prediction stay
traceable from the model output all the way back to the exact sensor windows,
transformations, features, model version, and source observations that produced it?

> **Status: both peripheral devices working end to end.** Provenance spine, quality
> control and features for a wrist band and a chest strap, subject-wise evaluation,
> calibration, ablation by sensor and by device, and a closed cascade with CDFS. EEG is
> not built. Every claim below is either
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
"model rerun". CDFS requires a reason on anything that supersedes a fact — 21 CFR
11.10(e) applies to a model's corrections as much as a monitor's — and the input that
changed is the answer to the question a reviewer is actually asking.

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

These numbers are not the whole claim. [What the model is actually
reading](#what-the-model-is-actually-reading) takes the sensors apart, and the answer
lowers the figure that can honestly be called physiological.

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
a stated 90% is nearer 80%. [Calibration](#calibration-fixes-the-spread-not-the-average)
improves it, and not in the way the average suggests.

Simple models were run first to establish whether the engineered features carry usable
signal before anything deeper is justified. They do, and the simplest one is currently
the most trustworthy across people.

## What the model is actually reading

A score says a model works. It does not say what it is working *from*, and on a wrist
device that distinction decides what the result means. So each signal was scored alone,
and then removed from the whole — two different questions, because a sensor can do well
alone and still cost nothing when dropped, if another one carries the same information.

Logistic regression, same folds throughout:

| features | n | bal. accuracy | AUC | worst subject |
| --- | ---: | ---: | ---: | ---: |
| all signals | 28 | 0.887 ±0.058 | 0.947 | 0.796 |
| **accelerometer alone** | 8 | **0.855 ±0.077** | 0.928 | 0.634 |
| electrodermal alone | 9 | 0.819 ±0.117 | 0.893 | 0.500 |
| pulse alone | 6 | 0.744 ±0.108 | 0.824 | 0.548 |
| temperature alone | 5 | 0.702 ±0.145 | 0.744 | 0.337 |
| without accelerometer | 20 | 0.839 ±0.108 | 0.912 | 0.603 |
| without electrodermal | 19 | 0.853 ±0.108 | 0.923 | 0.578 |
| without pulse | 22 | 0.885 ±0.059 | 0.939 | 0.748 |
| without temperature | 23 | 0.889 ±0.054 | 0.953 | 0.794 |

**The accelerometer alone gets 0.855 — within 0.032 of all 28 features together.** It is
the largest contributor on removal (−0.048) and the pattern repeats with random forest,
where accelerometry alone reaches 0.854 against 0.859 for everything.

That is a finding about the dataset, not a better model. WESAD induces stress with the
Trier Social Stress Test: the participant stands, speaks in front of a panel and does
mental arithmetic, while baseline is seated reading and meditation is seated breathing.
Posture and movement differ across conditions **by design**. A classifier reaching 0.855
from accelerometry is substantially detecting the protocol rather than the physiology,
and it would not survive contact with a setting where stressed people sit still.

The honest number for the physiological claim is the one with movement removed:
**0.839 balanced accuracy from pulse, electrodermal activity and temperature alone.**
Lower than the headline, and it is the one that means what it appears to mean.

Two smaller results in the same table. Pulse contributes +0.003 — near nothing beyond
what the other sensors already supply, consistent with a wrist optical sensor that gives
rate and no usable variability. Temperature contributes −0.002: the model is very
slightly *better* without it, which is reported rather than rounded away.

One detail worth the space: for random forest, accelerometry alone has a worst subject of
0.753, while the full feature set drops to 0.500 on S14. Adding physiological features to
movement made that model fail completely on one person.

## Calibration fixes the spread, not the average

A probability is calibrated if the windows it calls 30% likely turn out stressed about
30% of the time. Ranking does not need this. A number a person reads does.

Calibration is fitted on **held-out subjects**, like everything else here. The usual
shortcut is an inner stratified split, which puts the same participants in the fit set
and the calibration set; the calibrator then learns that person's particular
overconfidence rather than the model's general one, and reports itself better calibrated
than it is. The inner split groups by subject and reuses the same splitter as the outer
evaluation.

Isotonic regression, leave-one-subject-out, decisions left untouched:

| model | bal. accuracy | AUC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| logistic | 0.887 | 0.947 | 0.077 | 0.101 |
| logistic + isotonic | 0.887 | 0.947 | **0.070** | **0.086** |
| random forest | 0.859 | 0.978 | 0.069 | 0.117 |
| random forest + isotonic | 0.859 | 0.975 | 0.067 | **0.095** |
| gradient boosting | 0.852 | 0.976 | 0.082 | 0.086 |
| gradient boosting + isotonic | 0.852 | 0.974 | **0.067** | 0.088 |

Balanced accuracy is identical everywhere, by design: calibration restates confidence and
leaves the operating point alone, so a before-and-after row shows the effect of one change
rather than two. Isotonic regression is monotone, so ranking — and therefore AUC — is
unchanged too.

**On the average, this looks like a small win and for gradient boosting like none at all.
The average is the wrong number.** Per subject, for logistic regression:

| | mean | spread (sd) | worst subject |
| --- | ---: | ---: | ---: |
| uncalibrated | 0.101 | 0.067 | 0.254 |
| isotonic | 0.086 | **0.031** | **0.171** |

The spread across people more than halves. The subjects who were badly served improve a
lot — S15 from 0.195 to 0.066, S11 from 0.254 to 0.171, S16 from 0.148 to 0.093 — while
subjects who were already fine get slightly worse, S3 from 0.044 to 0.088. That is the
trade a calibrator makes: it pulls everyone toward the cohort's confidence, which helps
whoever was furthest out.

The underlying problem is visible in the stated rates. Every participant's true stress
share is between 0.21 and 0.24 — the protocol fixes it. The uncalibrated model's *average
stated probability* ranges from 0.187 for S3 to **0.475 for S11**, two and a half times
apart for people whose actual rate is the same. Calibration narrows that to 0.133–0.390.
It does not close it, and no single global calibrator can: S11's confidence is wrong in a
way that is specific to S11.

So the honest statement is that these probabilities are better than they were and are
still not good enough to put in front of someone as a percentage. What closes the gap is
[a few minutes of the person's own data](#seven-minutes-of-your-own-data).

## Adding a second device

WESAD also records a RespiBAN chest strap at 700 Hz: electrocardiogram, respiration,
muscle activity, plus its own electrodermal, temperature and accelerometry channels. 35
more features, joined to the wrist window by time interval.

The cardiac payoff is real. Heart-rate variability was [removed from the wrist feature
set](#features-and-one-that-was-measured-and-removed) after producing 236 ms of SDNN
against a true 65 — at 64 Hz one sample is 15.6 ms of the 20–60 ms the measure resolves.
At 700 Hz one sample is 1.43 ms, and on beat trains built to a stated SDNN the recovered
value comes back within about a millisecond. SDNN, RMSSD and pNN50 are ordinary here.

Whether the strap is worth wearing is a different question, and the ablation answers it on
identical rows and identical folds:

| features | n | bal. accuracy | AUC | worst subject |
| --- | ---: | ---: | ---: | ---: |
| both devices | 63 | **0.904 ±0.129** | 0.978 | 0.500 |
| wrist alone | 28 | 0.874 ±0.062 | 0.941 | **0.764** |
| chest alone | 35 | 0.881 ±0.141 | 0.970 | 0.500 |

**The second device raises the average and makes the model less reliable for the person it
serves worst.** Balanced accuracy goes up 0.029, the fold-to-fold spread doubles, and the
worst subject falls from 0.764 to chance. For a cohort statistic that is an improvement.
For something a person wears it is close to the opposite, and the mean alone would have
reported it as a straightforward win.

Per signal, the ranking is the same one the wrist found, more so:

| alone | n | bal. accuracy | AUC | worst |
| --- | ---: | ---: | ---: | ---: |
| **chest accelerometer** | 8 | **0.915 ±0.084** | 0.969 | 0.729 |
| wrist accelerometer | 8 | 0.857 ±0.070 | 0.928 | 0.668 |
| chest electrocardiogram | 7 | 0.797 ±0.136 | 0.953 | 0.510 |
| chest electrodermal | 9 | 0.738 ±0.146 | 0.846 | 0.438 |
| chest muscle activity | 3 | 0.673 ±0.163 | 0.716 | 0.349 |
| chest respiration | 3 | 0.659 ±0.078 | 0.691 | 0.579 |
| chest temperature | 5 | 0.585 ±0.125 | 0.721 | 0.431 |

**The chest accelerometer alone beats all 63 features together** — 0.915 against 0.904,
with a better worst subject and half the spread. It is the single best sensor in this
dataset, and it is a posture sensor. The stress condition has participants standing to
speak; a strap on the torso reads that more directly than a band on the wrist does.

That makes the accelerometry finding from the wrist harder to explain away rather than
easier. With movement removed from both devices:

| physiology only | n | bal. accuracy | worst subject |
| --- | ---: | ---: | ---: |
| both devices | 47 | 0.871 ±0.135 | 0.500 |
| chest only | 27 | 0.839 ±0.158 | 0.495 |
| wrist only | 20 | 0.834 ±0.110 | 0.650 |

Adding a chest strap's worth of electrocardiogram, respiration and muscle activity to a
wrist band buys **0.037 balanced accuracy and costs 0.150 on the worst subject**. The
electrocardiogram is the largest single contributor on removal (+0.021), so the cardiac
features do carry signal — they are just more subject-specific than the wrist's, which is
what an inter-individual SDNN range of 40 to 100 ms would predict.

### One subject is missing from that table

S16's electrocardiogram clips against the amplifier rail, and it does so much more in the
stress condition — 2.9% of samples against 1.5% at baseline. The quality-control rule
that catches it therefore rejects that participant's **entire positive class**, and the
detector's output for those windows is not physiology anyway: 128 bpm with an RMSSD of
6.9 ms, which is the shape of a detector tracking a flat clipped plateau.

That is a quality-control threshold correlated with the label, which is worth stating
plainly rather than tuning until it goes away.

It also exposed a bug of my own. A fold whose test subject has only one class cannot yield
a balanced accuracy — it is the mean of per-class recall, and an absent class has none.
Scored anyway it returns 1.0 or 0.0 for whichever class is present. It showed up as **the
majority-class baseline scoring 0.533 instead of exactly 0.500**, which is the reason that
row is in every table. Such folds are now skipped and the subject is named in the output,
because a mean over fourteen subjects reported as though it were fifteen is a quiet lie.

## Seven minutes of your own data

Cohort calibration narrows the spread of stated confidence across people and cannot close
it, because S11's confidence is wrong in a way specific to S11. The remedy a deployment
actually has is a short enrolment: a few labelled minutes from the person, before the
model is trusted on them.

**The model is never trained on that enrolment.** It stays leave-one-subject-out; only the
calibrator sees the person's own data, which keeps this a statement about calibration
rather than about fine-tuning. Enrolment windows, and every window overlapping one, are
removed from the evaluation — so all three columns below are scored on identical rows.

| enrolment | wall clock | ECE, none | ECE, cohort | ECE, personal | worst subject |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5% of each condition | 7.1 min | 0.098 | 0.088 | **0.038** | 0.245 → 0.166 → **0.104** |
| 10% | 9.2 min | 0.099 | 0.090 | 0.047 | 0.250 → 0.169 → 0.187 |
| 20% | 13.8 min | 0.101 | 0.096 | **0.040** | 0.260 → 0.175 → **0.102** |

**Seven minutes of a person's own labelled data more than halves the calibration error,
and beats cohort calibration by a factor of over two.** The subjects the cohort calibrator
could not reach are the ones that move most: S11 from 0.260 to 0.021, S15 from 0.189 to
0.008, S13 from 0.157 to 0.015.

More enrolment is not reliably better — 10% scores worse than 5% here, and one subject
(S6) is made worse by personalisation at every size tried. With fifteen participants those
differences are within the noise of the estimate, which is the honest reading rather than
a dose-response curve.

### The enrolment has to contain the thing being calibrated

Three ways of choosing those minutes were implemented, and the differences are findings
rather than options.

**Scattering single windows across the session does not work at all.** Each enrolment
window forces a window-length exclusion on either side, so it costs two minutes of session
to buy one. Twenty-four of them consume a twenty-minute recording completely: the first
version of this returned an empty evaluation set for every subject, which is the
arithmetic of the idea rather than a bug in it. Enrolment is taken in contiguous blocks,
which cost the same two minutes however long they are.

**Enrolling someone before the session starts does not work either.** A protocol runs its
conditions in blocks, so the opening minutes are one condition, and a calibrator given one
class has nothing to calibrate.

**Blocks spread evenly across the session may or may not work,** depending on where the
stress episode happens to fall relative to them.

What works is taking the opening slice of *each condition the session passes through*. The
practical statement the three make together: personalising a stress model needs labelled
stress from that person. Time on the device is not enough, and neither is a lot of it.

One number in this section was wrong before it was right. Enrolment cost was first
reported as one minute per window — 107 minutes for a session that ran for 60. These
windows overlap by 55 of their 60 seconds, so the cost is the union of what they cover,
not the sum of their lengths, and the cost is the whole point of the method.

## Not built

EEG preprocessing, montage capability, features and channel ablation ·
frequency-domain heart-rate variability, which needs windows longer than
the minute these are · deep architectures, which the ablations do not yet justify.

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
python scripts/ablate.py wesad_features.npz --model logistic
python scripts/evaluate.py wesad_features.npz --calibrated

python scripts/build_features.py ~/Downloads/WESAD.zip wesad_fused.npz --device both
python scripts/ablate.py wesad_fused.npz --by device
python scripts/personalise.py wesad_features.npz
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
