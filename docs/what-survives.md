# What survives measurement

Every claim in this repository has been through the same test: paired across
participants, resampled over participants, and reported with an interval. Some
of them did not come back. This page is the ledger, because a reader who finds a
withdrawn claim next to a surviving one and cannot tell which is which has been
given nothing.

The unit is the participant. Fifteen people is the sample size; 8,057 windows is
not, and an interval built from windows would be far too narrow.

## Supported

| claim | evidence |
| --- | --- |
| **Movement carries much of the stress signal on WESAD.** | Removing accelerometry costs **0.054** balanced accuracy, 95% interval **[−0.093, −0.016]**, improving only 4 of 15 participants. The accelerometer alone reaches 0.855 against 0.898 for all 28 features. |
| **Signal quality alone partly identifies the condition.** | Twelve columns describing only how the recording went reach **0.663** balanced accuracy against a 0.500 baseline. |
| **Logistic regression beats gradient boosting and the linear SVM.** | Paired intervals **[−0.100, −0.011]** and **[−0.108, −0.010]**, both excluding zero. |
| **Every chance-level per-subject score here is a threshold artifact, not a failure to learn.** | S14 scores 0.500 balanced accuracy at **1.000 AUC**. The worst *ranking* across fifteen participants is 0.912. |
| **Cohort calibration narrows the spread of stated confidence.** | Per-participant ECE spread falls from sd 0.061 to **0.028**, worst 0.213 to 0.129. |
| **Wrist pulse variability, as implemented here, is not defensible.** | 236 ms of SDNN against a 65 ms reference estimate. The features are not emitted. |
| **Sleep staging from a scalp montage works.** | κ **0.710** at 79% accuracy, inside the published range for feature-based automatic staging under subject-wise validation. |

## Withdrawn

| claim, as first published | what the measurement says |
| --- | --- |
| ~~"Adding the chest strap buys 0.029 balanced accuracy."~~ | +0.010, interval **[−0.073, +0.079]**, 8 of 14 improving. On this cohort the two devices are **indistinguishable**. |
| ~~"Seven minutes of your own data more than halves the calibration error and beats cohort calibration two to one."~~ | Personal calibration is **worse** than cohort on every measure, and damages ranking (AUC 0.952 → 0.891). The earlier figure was an artifact of a metric bug plus a missing baseline. Done **prospectively** — the only order a deployment could use — it reaches ECE 0.321 against 0.090 for doing nothing, costs 33 minutes rather than 7, and cannot be performed at all for 7 of 15 participants. |
| ~~"The model is very slightly better without temperature."~~ | +0.002, interval **[−0.009, +0.013]**, 9 of 15 improving. No difference. A rounding error read as a finding. |
| ~~"Adding physiological features made the model fail completely on one person."~~ | That participant's AUC is **1.000**. The model ranks them perfectly and mislabels them; the threshold moved, the knowledge did not. |
| ~~"Isotonic calibration leaves AUC unchanged."~~ | It is monotone *non-decreasing*; its flat regions create ties. 0.954 → 0.951. |
| ~~"A majority baseline has an ECE of 0.000."~~ | 0.222 — equal to its Brier score, as it must be. The bin edges were dropping predictions of exactly zero. |

## Verified rather than asserted

| claim | how |
| --- | --- |
| **The EDF files are read correctly.** | Cross-checked against pyedflib — a wrapper on the reference C library — on the real recordings: record counts, per-channel sample counts, mixed 100 Hz and 1 Hz rates in one file, physical scaling (agreeing to ~1e-13), and every annotation onset, duration and label. |
| **No participant appears on both sides of a split.** | Asserted across all three split strategies and again at the end of a real evaluation. |
| **Quality control is applied before features are computed.** | `extract` requires a verdict; it cannot be called without one. |
| **A window's identity survives a quality-control revision.** | Tested directly, which is what makes the invalidation walk possible at all. |
| **The core needs no scientific stack.** | CI installs `[dev]` only, asserts NumPy is absent, and runs the provenance tests against that. Reproducible locally with an import blocker. |

## Not established either way

- **Whether removing accelerometry leaves a confound-free physiological score.** It does
  not: pulse, electrodermal activity and temperature all respond to posture, speech and
  artifact. The 0.844 figure is signal with the most direct movement measurement removed,
  not stress physiology measured cleanly.
- **Whether logistic regression is better than random forest.** Paired interval
  [−0.101, +0.007] — crosses zero.
- **Whether a model trained here would fail where stressed people sit still.** WESAD
  contains no such condition. The burden is on the claim that it would not.
- **Whether the quality indicators are redundant with the signal features.** Adding them
  changes nothing measurable, which is consistent with redundancy but does not establish
  it — a feature can be redundant with a *combination* of others without duplicating any.
- **Whether a short enrolment could work across sessions.** Within one session,
  prospectively, it is worse than doing nothing — measured. Calibrating on one night and
  scoring another has not been run; given the within-session result it is unlikely to
  rescue the method, but that is an expectation and not a measurement.

---

[← back to the README](../README.md)
