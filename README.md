# PhysioML

Traceable multimodal physiological and neural inference — a companion machine-learning
layer for the [Clinical Data Fabric System](https://github.com/Elnazkarami/clinical-data-fabric-).

**The question this exists to answer:** can a physiological or neural prediction stay
traceable from the model output all the way back to the exact sensor windows,
transformations, features, model version, and source observations that produced it?

> **Status: peripheral and neural, both working end to end.** Provenance spine; quality
> control and features for a wrist band, a chest strap and a sleep montage; subject-wise
> evaluation, calibration and personalisation; ablation by sensor, by device and by
> electrode; and a closed cascade with CDFS. Two datasets, two tasks, one pipeline. Every claim below is either
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

## The datasets

Two, deliberately. WESAD carries the peripheral work and Sleep-EDF the neural work, and
the same pipeline reads both — the same windowing, the same quality-control shape, the
same subject-wise evaluation, the same tables. A second dataset is the cheapest test of
whether any of that generalised or was written for one file format.

### WESAD — wrist and chest, awake

15 subjects, 24.1 hours of chest and wrist physiology, read straight out of the
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

### Sleep-EDF Expanded — a scalp montage, asleep

20 subjects of the Sleep Cassette cohort, one night each, in European Data Format. Read
without a toolbox: EDF is a header of fixed-width ASCII fields followed by interleaved
16-bit records, and each file is memory-mapped so that asking for one channel touches only
its own columns rather than loading a night of polysomnography to reach it.

| | epochs | share |
| --- | ---: | ---: |
| N2 | 9,200 | 44.6% |
| REM | 3,756 | 18.2% |
| wake | 3,449 | 16.7% |
| N3 | 2,981 | 14.5% |
| N1 | 1,240 | 6.0% |
| **scored and kept** | **20,626** | **35.8%** |
| trimmed | 36,950 | recorder running before bed and after waking |
| unscored | 3,013 | movement time and epochs nobody scored |

30-second epochs, because that is what the scorer used. Consecutive rows share no signal,
which is the one thing that is *easier* here than on WESAD.

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
it; neither fixed it. At 64 Hz one sample is 15.6 ms, a large fraction of the 20–60 ms the
measure resolves, and every missed or doubled beat enters squared.

**That is a finding about this implementation on these recordings, not a claim about wrist
photoplethysmography.** What was measured is that *this* detector, on *this* dataset,
at 64 Hz, produced a number wrong by a factor of 3.6. Artifact correction, ectopic-beat
handling, and interpolation to a finer time base are all standard and none of them are
implemented here; published work does recover usable variability from wrist optics under
favourable conditions. The claim being made is narrow on purpose: these features, as
built, could not be defended, so they are not emitted.

So they are not emitted. They would have carried enough signal to *raise a model's score*
— being correlated with artifact rate and therefore with movement — while being
indefensible as physiology. A number wrong by a factor of four that looks useful is worse
than an absent one. Rate is kept, at an error in line with what an optical wrist sensor
gives. The [chest electrocardiogram](#adding-a-second-device) supplies the variability
this could not, and the same measures are ordinary there.

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
| **logistic regression** | **0.898 ±0.056** | 0.882 | 0.954 | 0.891 | 0.070 | 0.090 | **0.766** |
| linear SVM | 0.844 ±0.112 | 0.859 | 0.961 | 0.911 | 0.062 | 0.082 | 0.620 |
| random forest | 0.854 ±0.113 | 0.851 | 0.976 | 0.941 | 0.069 | 0.115 | 0.500 |
| gradient boosting | 0.849 ±0.110 | 0.847 | **0.977** | 0.938 | 0.085 | 0.089 | 0.504 |

The majority row is there so the others mean something: on a task that is 22% positive,
answering "baseline" every time is 78% accurate and 0.500 balanced-accurate.

These numbers are not the whole claim. [What the model is actually
reading](#what-the-model-is-actually-reading) takes the sensors apart, and the answer
lowers the figure that can honestly be called physiological.

**Logistic regression wins, and the ranking flips depending on which column you read.**
The ensembles have the better AUC — gradient boosting separates the classes best of
anything here — but logistic regression has the higher balanced accuracy and half the
fold-to-fold spread (±0.056 against ±0.113).

The last column is why that matters. Random forest scores 0.976 AUC across the cohort and
**0.500 — chance — on subject S14**; gradient boosting gets 0.504 on the same person.
Logistic regression does not fail on anyone: its worst subject is 0.766. A cohort mean hides a model that has learned
nothing at all about someone, which is exactly the failure a deployed physiological
classifier makes in front of a real user. Per-subject scores are reported for every fold
for that reason, and the worst one is carried into the summary rather than averaged away.

Calibration is mediocre across the board — 0.082 to 0.115 expected calibration error, so
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
| all signals | 28 | 0.898 ±0.056 | 0.954 | 0.766 |
| **accelerometer alone** | 8 | **0.855 ±0.077** | 0.928 | 0.634 |
| electrodermal alone | 9 | 0.822 ±0.110 | 0.886 | 0.537 |
| pulse alone | 6 | 0.744 ±0.108 | 0.824 | 0.548 |
| temperature alone | 5 | 0.702 ±0.145 | 0.744 | 0.337 |
| without accelerometer | 20 | 0.844 ±0.102 | 0.916 | 0.619 |
| without electrodermal | 19 | 0.853 ±0.108 | 0.923 | 0.578 |
| without pulse | 22 | 0.882 ±0.062 | 0.945 | 0.760 |
| without temperature | 23 | 0.900 ±0.051 | 0.956 | 0.769 |

**The accelerometer alone gets 0.855 — within 0.043 of all 28 features together.** It is
the largest contributor on removal (−0.054) and the pattern repeats with random forest,
where accelerometry alone reaches 0.854 against 0.854 for everything.

That is a finding about the dataset, not a better model. WESAD induces stress with the
Trier Social Stress Test: the participant stands, speaks in front of a panel and does
mental arithmetic, while baseline is seated reading and meditation is seated breathing.
Posture and movement differ across conditions **by design**. A classifier reaching 0.855
from accelerometry is substantially detecting the protocol rather than the physiology,
and it would not survive contact with a setting where stressed people sit still.

The honest number for the physiological claim is the one with movement removed:
**0.844 balanced accuracy from pulse, electrodermal activity and temperature alone.**
Lower than the headline, and it is the one that means what it appears to mean.

Two smaller results in the same table. Pulse contributes +0.016 — little beyond what the
other sensors already supply, consistent with a wrist optical sensor that gives rate and
no usable variability. Temperature contributes −0.002: the model is very slightly
*better* without it, which is reported rather than rounded away.

One detail worth the space: for random forest, accelerometry alone has a worst subject of
0.753, while the full feature set drops to 0.500 on S14. Adding physiological features to
movement made that model fail completely on one person.

### Signal quality alone gets to 0.663

Movement is not the only thing the protocol leaves in the recording. Quality control is
not neutral with respect to the condition either: a stressed participant here is standing
and talking, so their signal is noisier than the same person sitting still. If the
artifact rate alone can pick the condition out, part of any score is being earned by
measurement quality rather than physiology.

So it was measured. Twelve columns describing only *how the recording went* — per signal,
whether quality control flagged it, whether it rejected it, and how many reasons it gave —
and nothing about the participant:

| input | columns | bal. accuracy | AUC |
| --- | ---: | ---: | ---: |
| majority class | — | 0.500 ±0.000 | 0.500 |
| **quality indicators only** | 12 | **0.663 ±0.114** | 0.663 |
| physiology only | 28 | 0.898 ±0.056 | 0.954 |
| physiology + quality | 40 | 0.897 ±0.056 | 0.953 |

**How noisy the recording was predicts the experimental condition at 0.663.** That is far
from the full model, and far from chance. It is a shortcut available to any model trained
on this dataset, and the fact that it exists is a property of laboratory stress protocols
rather than of this pipeline.

Adding those columns to the physiological ones changes nothing at all (0.898 to 0.897),
which says the physiological features already carry what the quality indicators know —
unsurprising, since most quality flags here are driven by movement and the accelerometer
measures that directly. The shortcut is real but it is not additive; it is the same
shortcut the [accelerometry result](#what-the-model-is-actually-reading) already found,
arriving through a different door.

Build the table with `--quality` to reproduce this. It is off by default, because a
feature that predicts the label without describing the person is a shortcut and not a
finding, and it should have to be asked for.

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
| logistic | 0.898 | 0.954 | 0.070 | 0.090 |
| logistic + isotonic | 0.898 | 0.951 | **0.061** | **0.070** |
| random forest | 0.854 | 0.976 | 0.069 | 0.115 |
| random forest + isotonic | 0.854 | 0.973 | 0.069 | **0.095** |
| gradient boosting | 0.849 | 0.977 | 0.085 | 0.089 |
| gradient boosting + isotonic | 0.849 | 0.975 | **0.071** | 0.094 |

Balanced accuracy is identical everywhere, by design: calibration restates confidence and
leaves the operating point alone, so a before-and-after row shows the effect of one change
rather than two. Isotonic regression is monotone, so ranking — and therefore AUC — is
unchanged too.

**On the average this looks like a modest win, and for gradient boosting like none at
all. The average is the wrong number.** Per subject, for logistic regression:

| | mean | spread (sd) | worst subject |
| --- | ---: | ---: | ---: |
| uncalibrated | 0.090 | 0.061 | 0.213 |
| isotonic | 0.070 | **0.028** | **0.129** |

The spread across people more than halves. The subjects who were badly served improve
most — S15 from 0.175 to 0.047, S11 from 0.213 to 0.125, S13 from 0.188 to 0.129 — which
is the trade a calibrator makes: it pulls everyone toward the cohort's confidence, and
that helps whoever was furthest out.

The underlying problem is visible in the stated rates. Every participant's true stress
share is between 0.21 and 0.24 — the protocol fixes it. The uncalibrated model's *average
stated probability* ranges from 0.171 to **0.434**, two and a half times apart for people
whose actual rate is the same. Calibration narrows that to 0.131–0.350. It does not close
it, and no single global calibrator can: that participant's confidence is wrong in a way
specific to them.

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
| both devices | 63 | **0.897 ±0.128** | 0.979 | 0.500 |
| wrist alone | 28 | 0.884 ±0.067 | 0.947 | **0.707** |
| chest alone | 35 | 0.882 ±0.139 | 0.974 | 0.500 |

**The second device buys almost nothing on average and makes the model far less reliable
for the person it serves worst.** Balanced accuracy goes up 0.013, the fold-to-fold spread
doubles, and the worst subject falls from 0.707 to chance. For a cohort statistic that is
a small improvement. For something a person wears it is close to the opposite, and the
mean alone would have reported it as a win.

Per signal, the ranking is the same one the wrist found, more so:

| alone | n | bal. accuracy | AUC | worst |
| --- | ---: | ---: | ---: | ---: |
| **chest accelerometer** | 8 | **0.915 ±0.084** | 0.969 | 0.729 |
| wrist accelerometer | 8 | 0.857 ±0.070 | 0.928 | 0.668 |
| chest electrocardiogram | 7 | 0.797 ±0.136 | 0.953 | 0.510 |
| chest electrodermal | 9 | 0.754 ±0.146 | 0.844 | 0.498 |
| wrist electrodermal | 9 | 0.818 ±0.123 | 0.877 | 0.533 |
| chest muscle activity | 3 | 0.673 ±0.163 | 0.716 | 0.349 |
| chest respiration | 3 | 0.659 ±0.078 | 0.691 | 0.579 |
| chest temperature | 5 | 0.585 ±0.125 | 0.721 | 0.431 |

**The chest accelerometer alone beats all 63 features together** — 0.915 against 0.897,
with a better worst subject (0.729) and two-thirds the spread. It is the single best sensor in this
dataset, and it is a posture sensor. The stress condition has participants standing to
speak; a strap on the torso reads that more directly than a band on the wrist does.

That makes the accelerometry finding from the wrist harder to explain away rather than
easier. With movement removed from both devices:

| physiology only | n | bal. accuracy | worst subject |
| --- | ---: | ---: | ---: |
| both devices | 47 | 0.873 ±0.131 | 0.500 |
| chest only | 27 | 0.835 ±0.157 | 0.499 |
| wrist only | 20 | 0.837 ±0.108 | 0.643 |

Adding a chest strap's worth of electrocardiogram, respiration and muscle activity to a
wrist band buys **0.036 balanced accuracy and costs 0.143 on the worst subject**. The
cardiac features do carry signal — they are just more subject-specific than the wrist's,
which is what an inter-individual SDNN range of 40 to 100 ms would predict.

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
| 5% of each condition | 7.1 min | 0.088 | 0.073 | **0.038** | 0.215 → 0.127 → **0.095** |
| 10% | 9.2 min | 0.091 | 0.076 | 0.042 | 0.221 → 0.131 → 0.140 |
| 20% | 13.8 min | 0.094 | 0.083 | **0.038** | 0.235 → 0.140 → 0.100 |

**Seven minutes of a person's own labelled data cuts the calibration error by more than
half, and beats cohort calibration by close to two to one.** The worst-served subject
improves from 0.215 to 0.095.

More enrolment is not reliably better — 10% scores worse than 5% here on both the mean and
the worst subject. With fifteen participants that difference is within the noise of the
estimate, which is the honest reading rather than a dose-response curve.

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

## Sleep staging, on a second dataset

WESAD has no electroencephalography, so the neural half of this project had nothing to be
validated against — and code validated against nothing is what this repository exists to
argue with. [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/) supplies
it: whole nights of polysomnography, two EEG derivations at 100 Hz, an electro-oculogram,
chin electromyography, and a hypnogram an expert scored in 30-second epochs.

Twenty subjects, one night each. The epochs are the scorer's epochs, so unlike the
peripheral tables **consecutive rows share no signal at all** — the overlap that makes
every within-subject split delicate on WESAD does not exist here.

Three properties of the data are handled rather than assumed. **The recording is far
longer than the sleep**: a Sleep Cassette night runs about twenty hours, and the first
file opens with eight and a half hours of a single annotation saying the subject is awake.
Left alone, wake is three-quarters of the epochs and a classifier answering "awake" scores
extremely well; trimming to the sleep period plus 30 minutes cuts 36,950 epochs of 57,576
and leaves a realistic distribution. **Stages are scored under Rechtschaffen and Kales**,
which separates 3 from 4; they are merged into N3 as modern practice does. **Both nights
of a subject are one subject** — treating them as two people would put the same person on
both sides of every split, the leak the whole evaluation exists to prevent, arriving
through the file naming.

20,626 epochs, 48 features, five stages — N2 45%, REM 18%, W 17%, N3 14%, N1 6%.

| model | bal. accuracy | Cohen's κ | accuracy | macro F1 | worst subject |
| --- | ---: | ---: | ---: | ---: | ---: |
| majority class | 0.200 ±0.000 | 0.000 | 0.440 | 0.121 | 0.200 |
| logistic regression | **0.737 ±0.104** | 0.662 ±0.148 | 0.748 | 0.686 | 0.420 |
| random forest | 0.725 ±0.070 | **0.710 ±0.090** | 0.793 | 0.702 | 0.534 |
| gradient boosting | 0.714 ±0.078 | 0.708 ±0.102 | **0.794** | 0.699 | 0.540 |

Sleep staging is reported in Cohen's κ, not accuracy — the stages are unevenly distributed
enough that raw agreement flatters everything, which the majority row makes concrete at
44% accuracy and κ of exactly zero. **κ = 0.710 with 79% accuracy sits inside the range
published for feature-based automatic staging under subject-wise validation**, which is
the check that matters here: the pipeline is new, the task is not, and a number far outside
that range would mean something was wrong rather than something was discovered.

### Which model is better depends on which stage you care about

| per-stage recall | N1 | N2 | N3 | REM | W |
| --- | ---: | ---: | ---: | ---: | ---: |
| logistic regression | **0.562** | 0.725 | 0.909 | 0.716 | 0.773 |
| random forest | 0.303 | 0.815 | 0.886 | 0.761 | 0.858 |
| gradient boosting | 0.288 | 0.851 | 0.867 | 0.710 | 0.855 |

**Logistic regression finds N1 nearly twice as often as random forest and agrees with the
scorer less overall.** That is the whole disagreement between the two columns above: κ
rewards agreeing with a scorer whose night is 45% N2, and balanced accuracy rewards seeing
the stage that is 6% of it. Neither is the right answer in general; they are answers to
different questions, and reporting only one would hide that there was a choice.

N1 is the stage every automatic scorer fails on, and the confusion says why it is hard
rather than which model is bad. Of the N1 epochs random forest gets wrong, **31% go to REM
and 21% to wake** — N1 is the transition into sleep and shares its theta with REM, and it
is also the stage human scorers agree on least. A model that confuses N1 with REM and wake
is failing the way the problem fails.

### One electrode gets most of the way

The montage question, answered on identical rows and folds with random forest:

| channel | features | bal. accuracy | κ | worst subject |
| --- | ---: | ---: | ---: | ---: |
| everything | 48 | 0.725 ±0.070 | 0.710 | 0.534 |
| **EEG Fpz-Cz alone** | 20 | **0.683 ±0.088** | **0.657** | 0.392 |
| EEG Pz-Oz alone | 20 | 0.663 ±0.074 | 0.625 | 0.530 |
| EOG horizontal alone | 5 | 0.637 ±0.071 | 0.547 | 0.446 |
| chin EMG alone | 3 | 0.269 ±0.044 | 0.094 | 0.190 |
| without Fpz-Cz | 28 | 0.695 ±0.076 | 0.664 | 0.518 |
| without Pz-Oz | 28 | 0.710 ±0.090 | 0.682 | 0.452 |
| without EOG | 43 | 0.712 ±0.075 | 0.694 | 0.547 |
| without chin EMG | 45 | 0.725 ±0.071 | 0.710 | 0.527 |

**A single frontal derivation reaches κ 0.657 against 0.710 for the whole montage** — 92%
of the agreement from one electrode pair, which is the finding a wearable would be
designed around. Fpz-Cz beats Pz-Oz on both measures and contributes twice as much on
removal, consistent with it being the derivation single-channel staging is usually built
on.

Two channels earn less than they look like they should. The electro-oculogram alone
reaches κ 0.547, which is far above chance for five features — but removing it from the
full set costs 0.012, because the EEG channels already carry most of what it says. And
**chin electromyography contributes nothing at all**: alone it is barely above chance
(κ 0.094 against 0.000), and removing it leaves the model unchanged. Muscle atonia is half
the textbook definition of REM, so that deserves an explanation rather than a shrug: in
these files the EMG is stored at 1 Hz as an envelope, giving 30 samples per epoch and
three crude amplitude features. The finding is about this recording of that channel, not
about chin tone.

## Not built

Sequence models, which is where sleep staging gains most — a scorer reads the epochs
before and after, and every model here sees one epoch alone · frequency-domain heart-rate
variability, which needs windows longer than the minute these are · deep architectures,
which the ablations do not yet justify · the Sleep Telemetry cohort, and the 58 Sleep
Cassette subjects not downloaded here.

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

python scripts/build_features.py ~/Downloads/WESAD.zip wesad_qc.npz --quality
```

For the sleep recordings — the `sleep-cassette` files from
[Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/):

```bash
python scripts/build_sleep_features.py ~/sleep-edf sleep_features.npz
python scripts/evaluate.py sleep_features.npz --positive none
python scripts/ablate.py sleep_features.npz --by channel --positive none \
    --model random_forest
```

About 45 seconds to build 20 nights, and the archive is read in place — the reader
memory-maps each file and touches only the channels it is asked for.

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
