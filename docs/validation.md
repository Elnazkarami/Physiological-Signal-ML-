# How the results were validated

Physiological machine learning is unusually easy to get a good-looking number
out of. The label often correlates with something about the recording rather
than the person; participants differ from each other more than models differ
from each other; and a cohort of fifteen or twenty is small enough that a
difference can look decisive and be noise. This page is the set of checks every
result here passed before it was reported.

## Every table carries the number a result has to beat

A majority-class row appears in every comparison. On the WESAD stress task it
scores 78% accuracy and 0.500 balanced accuracy; on the five-stage sleep task,
38% accuracy and a Cohen's κ of exactly zero. It is the one model whose score is
knowable by hand, which makes it a working check on the metrics themselves.

## Differences are paired across participants and carry an interval

Windows are not independent observations: 8,057 rows are fifteen people. Every
comparison is made within each participant and resampled over participants, so
the interval reflects the sample size that actually exists.

This is what separates a result from a description of a sample. Removing
accelerometry from the WESAD features costs 0.054 balanced accuracy with an
interval of [−0.093, −0.016] — a finding. Adding a chest strap to a wrist band
gains 0.010 with an interval of [−0.073, +0.079] — not one.

## Ranking is reported beside the decision

Balanced accuracy is the label a model emits at the threshold it was scored at;
area under the curve is whether it ordered that participant's windows correctly.
They come apart, and where they do the diagnosis matters: one WESAD participant
scores 0.500 balanced accuracy at an AUC of **1.000**. The model ranks every one
of their stressed windows above every calm one and labels them all negative,
because the probabilities it states for them average 0.045 against a true rate of
0.223. That is a threshold in the wrong place for one person, not a model that
learned nothing, and only the second column can tell you which.

## Coverage is reported beside performance

A pipeline that cannot produce a prediction for somebody has not scored badly on
them — they are absent, and the mean is over a smaller cohort than it appears.
Every device comparison reports how many participants and windows survived
quality control, and names anyone who could not be scored.

## Calibration is checked against a constant

Expected calibration error is minimised by stating the base rate: on WESAD a
constant 0.22 scores the best calibration error in the table at an AUC of exactly
0.500, having learned nothing. So calibration is never reported without that
baseline and a proper scoring rule beside it.

## Quality control is measured for label correlation

Signal quality is not evenly distributed across an experimental protocol. On
WESAD, a third of stress windows are flagged for motion against one per cent of
baseline windows, and quality indicators alone reach 0.663 balanced accuracy.
Any score that does not test for this is uninterpretable, so the test is built
in and reported.

## The readers are checked against independent implementations

The EDF reader is compared with `pyedflib`, a wrapper around the reference C
library, on the real recordings: record counts, per-channel sample counts, mixed
sampling rates within one data record, physical scaling — agreeing to about
1e-13 — and every annotation onset, duration and label.

## The provenance core is checked for isolation

CI installs the package without any scientific stack, asserts NumPy is absent,
and runs the provenance tests against that. The same check runs locally with an
import blocker, documented in [reproducing](reproducing.md).

## Splits, leakage, and what is recorded

No participant appears on both sides of a split, asserted across every split
strategy and again at the end of a real evaluation. Scaling and calibration are
fitted inside the fold. Quality control runs before features are computed —
`extract` cannot be called without a verdict. Any evaluation can write a manifest
naming the folds, features, versions, seeds and training-run identifiers behind
its numbers.

---

[← back to the README](../README.md)
