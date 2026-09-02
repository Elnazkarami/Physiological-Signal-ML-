# Defects found, and what each one cost

Two external reviews and the test suite between them found the following. They
are recorded because the interesting question about a result is not whether it
is impressive but whether it would have survived somebody looking, and because
four of these changed a published number.

Each row links to the test that now pins it.

## Found by review

| defect | what it did | effect on results |
| --- | --- | --- |
| **Calibration bins were open at both ends** | The first bin was `(0.0, 0.1]`, so a prediction of exactly 0.0 fell in no bin and was dropped from the weighted average. Asymmetric — `1.0` was always counted — which is why it survived. | Majority baseline ECE 0.000 → **0.222**. Isotonic clips to [0,1] and hits the endpoint where the raw model does not, so the omission **flattered calibration specifically**. Overturned the personalisation result. |
| **No constant baseline beside a calibration claim** | ECE is minimised by stating the base rate. | A constant at the enrolment prevalence scores **ECE 0.014 at AUC 0.500** — the best calibration in the table, from a predictor that cannot tell one window from another. |
| **Electrodermal decomposition padded with zeros** | `mode="same"` pulled the tonic level to zero at each window edge; the phasic residual spiked there. | A perfectly flat signal produced **one skin-conductance response per minute**. Mean response amplitude was **12× too large**. Every downstream table was rebuilt. |
| **All-NaN windows passed quality control** | Every comparison against NaN is false, so a window of them cleared each threshold in turn. | Features became NaN, were dropped as non-finite, and the row lost that signal with nothing on record. |
| **Unknown signals defaulted to accelerometry** | `MODALITIES.get(name, Modality.ACC)`. | Chest EMG was recorded as an accelerometer, and carried that through the provenance chain. |
| **Recordings carried no checksum** | Identity was metadata alone. | A corrected export and the export it corrected were the same recording. |
| **Preprocessing was recorded but not used** | Windows defaulted to an empty preprocessing id while the extractor and the QC check both filtered with the module default. | Provenance and arithmetic could disagree with nothing to notice. Fixing it exposed a second bug in the fix: the windows took the new settings while the epoch kept the old. |
| **Prediction identity omitted probability** | Not in the content hash. | Two predictions saying "stressed" at 0.51 and 0.99 shared an id — and CDFS writes class and confidence as two facts from one prediction. |
| **Extraction accepted a missing QC verdict** | `qc` defaulted to `None`. | The order the README promises — window, judge, measure — was skippable by accident. |
| **Differences reported without uncertainty** | Bare means. | Two claims withdrawn: the chest strap and the temperature ablation. |

## Found by measurement, before review

| defect | how it showed |
| --- | --- |
| **Pulse variability wrong by 3.6×** | 236 ms SDNN against a 65 ms chest reference. The features were removed rather than shipped. |
| **Heart rate estimated from the wrong signal** | The 5–15 Hz band-passed ECG has its dominant frequency in the QRS band: it reported **193 bpm for a 70 bpm heart**. Synthetic tests passed; only real data caught it. |
| **A breathing rate that was the filter's own shape** | Every window of every subject read exactly **6.00 breaths per minute**, then 13.00 after the band was narrowed. A constant column is worse than a missing one. |
| **QC judged the raw signal while features used the filtered one** | Quality control was approving a different signal from the one measured. |
| **A signal-quality threshold calibrated in the wrong band** | Measured over 0.5–8 Hz, applied over 0.58–3 Hz, where the classes did not separate at all. |
| **`dataset.py` contradicted its own docstring** | Intersecting feature names across rows let 34 windows delete 6 features from all 8,091 rows while reporting zero dropped. |
| **A fold with one class was scored anyway** | Balanced accuracy is the mean of per-class recall, and an absent class has none. It surfaced as a majority baseline reporting **0.533 instead of 0.500**. |
| **A test helper broke the no-dependency guarantee** | `conftest.py` is imported by every run, including the CI job that installs no scientific stack to prove the core needs none. |

## What is checked now that was not

| check | added after |
| --- | --- |
| EDF reader against an independent implementation on real files | the reviewer noted synthetic fixtures cannot establish correct handling of real ones |
| A constant-probability baseline beside every calibration claim | ECE turned out to be minimised by stating the base rate |
| Paired intervals over participants for every difference | two claims had been published on bare means |
| Per-subject AUC beside per-subject accuracy | a threshold failure had been reported as a model learning nothing |
| Coverage beside performance | excluding a participant had been reported without saying one was excluded |
| Dataset paths from the environment | every dataset test skipped silently on any machine but one |
| A manifest naming the folds behind a number | three tables have been rebuilt so far |

## Rules that came out of these

- A metric that cannot be checked by hand on a trivial input is not trusted. The majority
  baseline exists in every table for that reason, and it is what exposed the bin bug.
- A synthetic test is necessary and not sufficient. Two bugs passed clean synthetic
  signals and were found only on real recordings.
- A number that cannot move is a bug, not a measurement. Two constant columns were shipped
  before this became a rule.
- A difference without an interval is a description of a sample.

---

[← back to the README](../README.md)
