# Calibration, and a personalisation result that did not survive

Whether the probabilities mean what they say, what cohort calibration fixes, and why
fitting a calibrator on a person's own data made things worse.
## Calibration fixes the spread, not the average

A probability is calibrated if the windows it calls 30% likely turn out stressed about
30% of the time. Ranking does not need this. A number a person reads does.

Where the model is wrong, measured rather than asserted — logistic regression, pooled
across folds:

| stated | windows | mean stated | actually stressed |
| --- | ---: | ---: | ---: |
| 0.0–0.1 | 4,504 | 0.027 | 0.020 |
| 0.2–0.3 | 391 | 0.245 | 0.082 |
| 0.4–0.5 | 165 | 0.448 | 0.212 |
| 0.6–0.7 | 194 | 0.648 | 0.464 |
| 0.8–0.9 | 208 | 0.854 | 0.716 |
| 0.9–1.0 | 1,283 | 0.986 | 0.905 |

**The overconfidence is worst in the middle.** A window called 45% likely is stressed 21%
of the time; one called 85% likely is stressed 72% of the time. The two ends are close to
honest — the model is either confident and roughly right, or unconfident and roughly
right, and uncertain in a way that overstates the risk. A single expected-calibration-error
number cannot say any of that, which is why it is not the only thing reported.

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

Balanced accuracy is identical everywhere, by design: `predict` stays the base model's, so
the reported labels keep its decision rule rather than thresholding the calibrated
probability at 0.5. A before-and-after row therefore shows the effect of one change rather
than two.

AUC moves slightly, and that is not a rounding artifact. Isotonic regression is monotone
*non-decreasing*, not strictly increasing: its flat regions map distinct scores onto one
value, and those ties change the area under the curve. 0.954 to 0.951 for logistic
regression.

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
[a few minutes of the person's own data](#personal-calibration-on-short-labelled-blocks--a-negative-result).

## Personal calibration on short labelled blocks — a negative result

Cohort calibration narrows the spread of stated confidence across people and cannot close
it. The obvious next step is a short enrolment: a few labelled minutes from the person
themselves, used to fit a calibrator for them alone. It does not work here, and the way it
fails is worth more than the result I first reported.

**The model is never trained on the enrolment.** It stays leave-one-subject-out; only the
calibrator sees the person's own data. Enrolment windows, and every window overlapping
one, are removed from the evaluation, so all four columns are scored on identical rows.

With 7.1 minutes of labelled signal per participant, taken as blocks from the start of
each condition:

| variant | ECE | Brier | AUC |
| --- | ---: | ---: | ---: |
| uncalibrated | 0.088 | 0.067 | 0.955 |
| cohort calibration | 0.073 | **0.061** | 0.952 |
| **constant at enrolment prevalence** | **0.014** | 0.175 | 0.500 |
| personal calibration | 0.076 | 0.076 | 0.891 |

**Personal calibration is worse than cohort calibration on every measure**, and it damages
the ranking badly — AUC falls from 0.952 to 0.891. Isotonic regression fitted on seven
minutes of one person's data overfits: it produces a coarse step function with extreme
values, and applying it to that person's remaining night is worse than leaving the
probabilities alone.

**And the constant beats everything on calibration error while being useless.** Answering
"0.22" to every window — the share of the enrolment that was positive — scores the best
expected calibration error in the table, 0.014, with an AUC of exactly 0.500. It cannot
tell one window from another. This protocol fixes each participant's stress share near
22%, so a calibrator that has learned only the base rate looks superbly calibrated.

That is the finding: **expected calibration error alone does not measure what it is being
read as measuring.** It is minimised by stating the prevalence, which is why the Brier
score is beside it here — a proper scoring rule that the constant predictor loses badly
(0.175 against 0.061). Any calibration result reported without a constant baseline and a
proper score next to it is unfalsifiable.

### What this does and does not establish

The enrolment is taken from the *beginning of each condition* across the session, which
means the calibrator has seen labelled examples from every condition the evaluation later
scores. That is **retrospective, within-session, condition-informed calibration**. It is
not a claim that somebody can enrol for seven minutes and then receive calibrated
predictions, and two costs are being conflated if it is read that way:

- **Labelled signal**: 7.1 minutes, the union of the enrolment windows.
- **Elapsed time before enrolment is complete**: potentially most of the session, because
  the last condition block may occur near the end of it.

A prospective claim would need calibration fitted on earlier data and evaluated on later
data — ideally a different session. Excluding overlapping windows removes shared samples;
it does not remove the temporal structure that neighbouring windows share.

### The enrolment has to contain the thing being calibrated

Three ways of choosing those minutes were implemented, and the differences are structural.

**Scattering single windows does not work at all.** Each enrolment window forces a
window-length exclusion on either side, so it costs two minutes of session to buy one.
Twenty-four of them consume a twenty-minute recording: the first version returned an empty
evaluation set for every subject, which is the arithmetic of the idea rather than a bug in
it. Enrolment is taken in contiguous blocks, which cost the same two minutes however long
they are.

**Enrolling before the session starts does not work either.** A protocol runs its
conditions in blocks, so the opening minutes are one condition, and a calibrator given one
class has nothing to calibrate. **Blocks spread evenly** may or may not find the stress
episode depending on where it falls. Taking the opening slice of each condition is what
works — and it is also what makes the result retrospective.

One number here was wrong before it was right. Enrolment cost was first reported as one
minute per window — 107 minutes for a session that ran for 60. These windows overlap by 55
of their 60 seconds, so the cost is the union of what they cover, not the sum of their
lengths.


---

[← back to the README](../README.md)
