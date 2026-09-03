# The datasets, and the features taken from them

Two datasets, read by one pipeline: WESAD for the peripheral work and Sleep-EDF Expanded
for the neural work. Includes the feature set that was built, measured, and removed.
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

76 subjects of the Sleep Cassette cohort, one night each, in European Data Format. Read
without a toolbox: EDF is a header of fixed-width ASCII fields followed by interleaved
16-bit records, and each file is memory-mapped so that asking for one channel touches only
its own columns rather than loading a night of polysomnography to reach it.

| | epochs | share |
| --- | ---: | ---: |
| N2 | 34,534 | 35.9% |
| wake | 32,442 | 33.7% |
| REM | 12,185 | 12.7% |
| N1 | 10,518 | 10.9% |
| N3 | 6,458 | 6.7% |
| **scored and kept** | **96,137** | **42.0%** |
| trimmed | 120,669 | recorder running before bed and after waking |
| unscored | 12,196 | movement time and epochs nobody scored |

The denominator is 229,002 epochs across the seventy-six nights: 96,137 kept, 120,669
trimmed, 12,196 unscored. Of the 216,806 that carry a score, 44.3% are kept.

30-second epochs, because that is what the scorer used. Consecutive rows share no signal,
which is the one thing that is *easier* here than on WESAD.

## Features, and one that was measured and removed

28 features per 60-second window: pulse rate and amplitude, electrodermal level, slope,
SCR count and area, skin-temperature level and slope, and accelerometry magnitude, jerk
and per-axis variation. Quality control runs first and a rejected signal contributes
nothing, because a heart rate from a flatlined sensor is not a missing value — it is a
confident wrong one.

**Pulse-rate variability is not among them.** SDNN, RMSSD and pNN50 were implemented,
then compared against the chest electrocardiogram WESAD records alongside the wrist, on
the same windows of the same subject. The electrocardiogram is a *reference estimate*, not
ground truth: it is this pipeline's own R-peak detection on a cleaner signal, validated
against synthetic beat trains rather than against annotated beats, which WESAD does not
provide.

| | chest ECG (reference) | wrist PPG |
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
gives. The [chest electrocardiogram](devices.md) supplies the variability
this could not, and the same measures are ordinary there.


---

[← back to the README](../README.md)
