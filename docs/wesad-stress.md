# Stress classification on WESAD

The first measured result, and then the two ablations that decide what it means:
which sensor the model is reading, and whether signal quality alone can predict the
condition.
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

| model | bal. accuracy | macro F1 | AUC | PR-AUC | Brier | ECE | worst subject |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| majority class | 0.500 ±0.000 | 0.438 | 0.500 | 0.222 | 0.222 | 0.222 | 0.500 |
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

How much of that ordering is real, paired across the fifteen participants:

| against logistic regression | mean | 95% interval | improved |
| --- | ---: | ---: | ---: |
| random forest | −0.043 | [−0.101, +0.007] | 5 of 15 |
| gradient boosting | −0.049 | [−0.100, −0.011] | 4 of 15 |
| linear SVM | −0.054 | [−0.108, −0.010] | 4 of 15 |

Logistic regression is ahead of gradient boosting and the SVM by an interval that excludes
zero. **Against random forest it is not** — that interval crosses zero, so the two are not
separated by this cohort, and the table's ordering of them is not something fifteen
participants can support.

The last column is why that matters. Random forest scores 0.976 AUC across the cohort and
**0.500 — chance-level balanced accuracy — on subject S14**; gradient boosting gets 0.504
on the same person. Logistic regression's lowest observed participant score is 0.766.

**And that column, read alone, says the wrong thing.** Balanced accuracy is the label the
model emits at the threshold it was scored at. Area under the curve is whether it ordered
that person's windows correctly at all. They come apart:

| random forest, S14 | |
| --- | ---: |
| balanced accuracy | 0.500 |
| **AUC** | **1.000** |
| mean stated probability | 0.045 |
| actual stress share | 0.223 |

The model ranks **every one of S14's stressed windows above every calm one** — a perfect
ordering — and then labels all of them "not stressed", because the probabilities it states
for that participant sit five times below their true rate and the whole distribution falls
on one side of the boundary. Nothing was failed to be learned. A threshold was in the
wrong place for one person.

That pattern holds everywhere it was checked. Across all fifteen subjects the worst
*ranking* is 0.912 AUC for random forest, and in the fused table the worst is 0.874 while
one participant still scores 0.500 balanced accuracy at 0.972 AUC. **Every chance-level
per-subject result in this project is a threshold artifact on a well-ranked participant**,
which is a different problem with a different remedy — a per-person operating point, not a
better model. The per-subject tables now report both columns, because reporting only the
first called these cases failures. A cohort mean hides a model that has learned
nothing at all about someone, which is exactly the failure a deployed physiological
classifier makes in front of a real user. Per-subject scores are reported for every fold
for that reason, and the worst one is carried into the summary rather than averaged away.

Calibration is mediocre across the board — 0.089 to 0.115 expected calibration error.
A single such number does not say *which* confidences are wrong, so the reliability table
is [in the calibration report](calibration.md); the short version is that the
overconfidence is worst in the middle of the range, not at the top.

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
the largest contributor on removal, and it is the one ablation here that survives being
measured properly: paired across the fifteen participants, removing accelerometry costs
**0.054 with a 95% interval of [−0.093, −0.016]**, improving only 4 of 15 people. The
interval excludes zero. The pattern repeats with random forest, where accelerometry alone
reaches 0.854 against 0.854 for everything.

That is a finding about the dataset, not a better model. WESAD induces stress with the
Trier Social Stress Test: the participant stands, speaks in front of a panel and does
mental arithmetic, while baseline is seated reading and meditation is seated breathing.
Posture and movement differ across conditions **by design**. A classifier reaching 0.855
from accelerometry alone is, to that extent, detecting the protocol rather than the
physiology. Whether it would survive a setting where stressed people sit still is not
something this dataset can answer — it contains no such condition — but the burden is now
on the claim that it would.

With movement removed, the score is **0.844 from pulse, electrodermal activity and
temperature alone.** That is lower than the headline, and it is *not* a confound-free
measure of stress physiology either. Pulse, electrodermal activity and skin temperature
all respond to posture, speech, movement artifact and the timing of the protocol blocks;
dropping the explicit motion features does not remove those influences, it only removes
the most direct measurement of them.

What the ablation establishes is narrower and still worth having: **there are strong
protocol-related shortcuts available in this dataset, and any score reported without
testing for them is uninterpretable.** It identifies contribution within this setup. It
does not identify a mechanism, and it cannot: that would need validation under different
movement and task conditions — stressed participants sitting still, calm participants
moving — which WESAD does not contain.

Two smaller results in the same table, and neither survives the same treatment. Pulse
contributes +0.016 and temperature −0.002; the temperature difference has a paired
interval of [−0.009, +0.013] across fifteen participants, with 9 of them improving when it
is removed. **That is not "very slightly better without it", it is no difference at all**,
and the earlier wording read a rounding error as a finding. What can be said is that
neither sensor adds much beyond what the others already supply — consistent, for the
pulse, with a wrist optical sensor that gives rate and no usable variability.

One detail worth the space: for random forest, accelerometry alone has a worst subject of
0.753, while the full feature set drops to 0.500 on S14 — where its AUC is 1.000. Adding
the physiological features to movement cost that model its usable operating point for one
person while leaving its ranking of them perfect.

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
| signal-derived features | 28 | 0.898 ±0.056 | 0.954 |
| signal-derived + quality | 40 | 0.897 ±0.056 | 0.953 |

**How noisy the recording was predicts the experimental condition at 0.663.** That is far
from the full model, and far from chance. It is a shortcut available to any model trained
on this dataset, and the fact that it exists is a property of laboratory stress protocols
rather than of this pipeline.

Adding those columns to the 28 signal-derived ones changes nothing measurable (0.898 to
0.897). That is consistent with the signal features already carrying what the quality
indicators know — most flags here are driven by movement, which the accelerometer measures
directly — but it does not prove the two contain the same information. A feature can be
redundant with a *combination* of others without duplicating any of them.

Note also that the 28 features are not "physiology": eight are accelerometry. The honest
label is signal-derived.

### Where the shortcut comes from

Quality control is not evenly distributed across the protocol. The share of windows it
flags, by condition:

| condition | windows | ACC | BVP | EDA | TEMP |
| --- | ---: | ---: | ---: | ---: | ---: |
| amusement | 928 | 0.000 | 0.000 | 0.000 | 0.000 |
| baseline | 3,341 | 0.000 | 0.010 | 0.000 | 0.000 |
| meditation | 2,001 | 0.000 | 0.000 | 0.000 | 0.000 |
| **stress** | 1,787 | 0.000 | **0.335** | 0.013 | 0.000 |

**A third of stress windows are flagged for motion on the pulse signal, against one in a
hundred at baseline and none at all in the other two conditions.** That is the whole
0.663: the motion flag is very nearly a stress detector by itself, because a participant
standing and speaking moves and a seated one does not.

Nothing is rejected — the rejected fraction is 0.000 in every cell, because motion marks a
window questionable rather than unusable. So the flags remove nothing and still carry the
label. That is the right design for quality control and it is exactly why these indicators
have to be kept out of the feature set unless they are asked for: a column that predicts
the condition without describing the participant is a shortcut, and this one is a very
good shortcut.

It also explains why adding them to the signal features changes nothing. The accelerometer
already measures what makes the flag fire.

Build the table with `--quality` to reproduce this. It is off by default, because a
feature that predicts the label without describing the person is a shortcut and not a
finding, and it should have to be asked for.


---

[← back to the README](../README.md)
