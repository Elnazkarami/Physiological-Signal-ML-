# Adding a chest strap: performance, and coverage

Whether a second device earns its place. Two questions, answered separately: how good
the prediction is when the inputs are available, and how often they are.
## Adding a second device

WESAD also records a RespiBAN chest strap at 700 Hz: electrocardiogram, respiration,
muscle activity, plus its own electrodermal, temperature and accelerometry channels. 35
more features, joined to the wrist window by time interval.

The cardiac payoff is real. Heart-rate variability was [removed from the wrist feature
set](datasets.md#features-and-one-that-was-measured-and-removed) after producing 236 ms of SDNN
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
doubles, and the worst subject falls from 0.707 to chance — at the threshold. That
participant's AUC is 0.972, so the fall is again an operating point rather than a loss of
signal; the worst *ranking* in the fused table is 0.874. For a cohort statistic that is
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

### Coverage is the other half of the comparison

Excluding a participant whose signal quality control rejected is honest and insufficient.
The table above answers *when the inputs are available, how good is the prediction* — and
silently drops the question that matters at least as much for anything worn: **how often
are they available?**

| configuration | rows | subjects | scorable | stress windows | cannot be scored |
| --- | ---: | ---: | ---: | ---: | --- |
| wrist only | 8,057 | 15 | **15** | 1,787 | — |
| wrist + chest | 7,828 | 15 | **14** | 1,641 | S16 |
| chest only | 7,828 | 15 | **14** | 1,641 | S16 |

Adding the chest strap costs a whole participant. Retention of the wrist table's windows,
by condition, for the two people who lose any:

| subject | amusement | baseline | meditation | stress |
| --- | ---: | ---: | ---: | ---: |
| S16 | 0.18 | 1.00 | 1.00 | **0.00** |
| S3 | 1.00 | 0.94 | 0.94 | 0.78 |
| everyone else | ≈1.00 | ≈1.00 | ≈1.00 | ≈1.00 |

**S16 retains none of its stress windows.** Its electrocardiogram clips against the
amplifier rail, and it clips more during the stress condition — 2.9% of samples against
1.5% at baseline — so the rule that catches it removes that participant's entire positive
class. The detector's output there is not physiology anyway: 128 bpm with an RMSSD of
6.9 ms is a detector tracking a flat clipped plateau. A quality-control threshold that
correlates with the label is worth stating rather than tuning until it disappears.

So the chest-dependent pipeline produces **no usable prediction at all** for that person.
A comparison that reports performance after exclusion and stops there recommends the
strap.

On the fourteen participants every configuration can answer for — the only cohort on which
the devices can be compared without also measuring who each of them dropped:

| configuration | bal. accuracy | AUC | worst subject | worst ranking |
| --- | ---: | ---: | ---: | ---: |
| wrist only | 0.882 ±0.071 | 0.946 | **0.701** | 0.840 |
| chest only | 0.881 ±0.140 | 0.974 | 0.500 | 0.869 |
| wrist + chest | **0.892 ±0.128** | **0.979** | 0.500 | 0.872 |

**On the common subset the strap buys nothing that can be distinguished from noise.**
Paired participant by participant, with a percentile bootstrap resampling the fourteen
people:

| against wrist only | mean | 95% interval | improved |
| --- | ---: | ---: | ---: |
| wrist + chest | +0.010 | [−0.073, +0.079] | 8 of 14 |
| chest only | −0.001 | [−0.090, +0.081] | 8 of 14 |

Both intervals include zero, and both split the cohort almost evenly. The honest statement
is that **on these fifteen participants the two devices are indistinguishable, and adding
the second to the first is indistinguishable from either** — while doubling the spread,
taking the worst participant to a threshold failure, and costing one person in fifteen
entirely. The 0.010 is a description of this sample, not evidence of an improvement.

That is a stronger reason not to add the strap than the one I first reported, and it is a
reason that a cohort mean by itself could never have produced.


---

[← back to the README](../README.md)
