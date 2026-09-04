# Sleep staging on Sleep-EDF Expanded

Five-stage sleep scoring from a scalp montage, and the channel ablation that asks how
much of it survives on one EEG derivation.
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

96,137 epochs, 48 features, five stages — N2 36%, W 34%, REM 13%, N1 11%, N3 7%.

**These numbers replace an earlier result on 20 subjects, and they are worse.** That is
reported here rather than quietly swapped in, because the difference is the finding: a
cohort of twenty was not enough to estimate this, and it was not merely imprecise, it was
*unrepresentative*. Wake was 17% of the small cohort and is 34% of the full one; N3 was
14% and is 7%.

**The protocol, stated so the number can be compared with anything.** Every Sleep Cassette
subject with a first night — 76 of them — one night each, chosen by the file name rather
than by any property of the recording. Second nights are not downloaded.
Trimming uses the hypnogram itself to find the first and last non-wake epoch and keeps 30
minutes on each side, so **the evaluation is scored around a sleep interval located by the
reference annotation**. That is a valid benchmark scope and a disclosed one: it does not
measure finding sleep inside an unrestricted recording. Hyperparameters are the library
defaults, fixed before any run and never searched, so there is no inner selection loop to
group — and nothing was tuned against these folds. Every metric is the mean over the
seventy-six per-subject folds, not pooled predictions, except the confusion matrix, which
is summed.

Consecutive epochs share no signal, which removes the overlap problem the WESAD tables
carry. It does not make them independent: they come from one participant and one
continuously evolving sleep state, so a split made at random across epochs would still
leak. Splits are by subject here for that reason.

| model | bal. accuracy | Cohen's κ | accuracy | macro F1 | worst subject |
| --- | ---: | ---: | ---: | ---: | ---: |
| majority class | 0.206 ±0.016 | 0.000 | 0.377 | 0.110 | 0.200 |
| logistic regression | **0.695 ±0.111** | 0.607 ±0.135 | 0.721 | 0.613 | 0.227 |
| random forest | 0.674 ±0.102 | **0.656 ±0.124** | **0.765** | **0.633** | 0.310 |

Against the twenty-subject cohort, every column moved the same way:

| random forest | 20 subjects | 76 subjects |
| --- | ---: | ---: |
| Cohen's κ | 0.710 ±0.090 | **0.656 ±0.124** |
| accuracy | 0.793 | 0.765 |
| balanced accuracy | 0.725 | 0.674 |
| worst subject | 0.534 | **0.310** |

The majority row is 0.206 rather than exactly 0.200 because not every participant reaches
every stage; a fold missing one scores a constant answer at 1/4 rather than 1/5. It is
still zero on κ, which is the point of reporting κ.

Sleep staging is reported in Cohen's κ, not accuracy — the stages are unevenly distributed
enough that raw agreement flatters everything, which the majority row makes concrete at
38% accuracy and κ of exactly zero. **κ = 0.656 with 77% accuracy sits at the lower end of
the range published for feature-based automatic staging under subject-wise validation**,
which is the check that matters here: the pipeline is new, the task is not, and a number
far outside that range would mean something was wrong rather than something was found. At
twenty subjects it looked like 0.710, comfortably mid-range; the honest figure is lower.

### The features, in full

Twenty per electroencephalogram derivation, five from the electro-oculogram, three from
the chin electromyogram. Each derivation's features are prefixed with its own name
(`fpz_`, `pz_`), so a fused row holds both without one silently overwriting the other.

Spectra are estimated by Welch on 4-second segments — long enough to resolve delta, which
starts at 0.5 Hz and would otherwise be a single bin, and short enough that a 30-second
epoch holds several to average. Band power is integrated by the trapezium rule rather than
summed over bins, so the answer does not depend on the frequency resolution and a
30-second epoch agrees with a 20-second one about the same signal. No filtering is applied
before the spectrum; the bands do the selecting.

| feature | definition | unit |
| --- | --- | --- |
| `<ch>_delta` … `<ch>_beta` | integrated power in 0.5–4, 4–8, 8–12, 12–16, 16–30 Hz | µV²·Hz |
| `<ch>_delta_rel` … `<ch>_beta_rel` | the same, divided by power in 0.5–30 Hz | fraction |
| `<ch>_delta_theta_ratio` | delta ÷ theta — slow-wave dominance, which is what separates N3 | ratio |
| `<ch>_alpha_beta_ratio` | alpha ÷ beta — the balance that moves between REM and light sleep | ratio |
| `<ch>_hjorth_activity` | variance of the signal | µV² |
| `<ch>_hjorth_mobility` | sd of the first difference ÷ sd of the signal; for a sampled sine, exactly 2·sin(πf/rate) | dimensionless |
| `<ch>_hjorth_complexity` | mobility of the first difference ÷ mobility; 1.0 for a pure sine | dimensionless |
| `<ch>_entropy` | Shannon entropy of the normalised spectrum ÷ log(bins) | 0–1 |
| `<ch>_edge95` | frequency below which 95% of the power lies | Hz |
| `<ch>_total_power` | integrated power, 0.5–30 Hz | µV²·Hz |
| `<ch>_amplitude_p95` | 95th percentile of \|signal\| | µV |
| `<ch>_zero_crossings` | mean-crossings per second | Hz |
| `eog_slow_power` | integrated power, 0.3–2 Hz — where eye movements live | µV²·Hz |
| `eog_slow_rel` | the same ÷ power in 0.3–15 Hz | fraction |
| `eog_amplitude_sd`, `eog_amplitude_p95`, `eog_range` | deflection size | µV |
| `chin_emg_rms`, `chin_emg_p95`, `chin_emg_range` | muscle tone, mean removed first so an electrode offset is not tension | µV |

**Relative band power is emitted beside absolute** because absolute amplitude varies
several-fold between people for reasons unrelated to sleep — skull thickness, electrode
impedance, the amplifier's gain that night — and a model trained on one participant's
microvolts and tested on another's is being asked to generalise across the wrong thing.

Sigma is separated from beta because sleep spindles live at 12–16 Hz and are one of the
defining features of stage 2; folded into a wide beta band they are invisible.

Relative power is normalised over 0.5–30 Hz and not the whole spectrum: above 30 Hz a
scalp recording is largely muscle, and dividing by it would make every relative figure a
function of how tense the participant's jaw was.

### Which model is better depends on which stage you care about

| per-stage recall | N1 | N2 | N3 | REM | W |
| --- | ---: | ---: | ---: | ---: | ---: |
| logistic regression | **0.473** | 0.712 | 0.740 | 0.702 | 0.770 |
| random forest | 0.319 | 0.810 | 0.641 | 0.656 | **0.890** |

N3 is where the larger cohort hurt most: recall falls from 0.885 to 0.641 for random
forest. It was 14% of the twenty-subject cohort and is 7% of this one, so there is half as
much of it to learn from and it is being predicted away in favour of the classes that grew.

**Logistic regression finds N1 nearly twice as often as random forest and agrees with the
scorer less overall.** That is the whole disagreement between the two columns above: κ
rewards agreeing with a scorer whose night is 45% N2, and balanced accuracy rewards seeing
the stage that is 6% of it. Neither is the right answer in general; they are answers to
different questions, and reporting only one would hide that there was a choice.

N1 is the stage every automatic scorer fails on, and the confusion says why it is hard
rather than which model is bad. Random forest, summed over the twenty folds — rows are the
scorer's label, columns the model's:

| scored → | N1 | N2 | N3 | REM | W | recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **N1** | **3,190** | 2,895 | 25 | 1,740 | 2,244 | 0.316 |
| **N2** | 2,033 | **27,328** | 1,313 | 1,697 | 1,092 | 0.817 |
| **N3** | 2 | 1,127 | **5,172** | 10 | 147 | 0.801 |
| **REM** | 1,641 | 1,459 | 27 | **7,740** | 1,036 | 0.650 |
| **W** | 1,922 | 331 | 98 | 695 | **26,686** | 0.898 |

**These recalls are not the ones in the table above, and the difference is not an error.**
This matrix is summed across folds, so it is dominated by participants with many epochs of
a stage; the per-stage recalls reported earlier are the mean over the seventy-six
participant folds, so a participant with fifteen N3 epochs counts as much as one with a
thousand. N3 reads 0.801 pooled and 0.641 averaged, and the gap is exactly the population
of participants who have very little N3 and whom the model scores badly on it. The
averaged figure is the one that answers "how will this do for a person"; the pooled one
answers "how many epochs did it get right".

Of the N1 epochs it gets wrong, **29% go to N2, 22% to wake and 17% to REM**. N1 is the
transition into sleep, and it borders all three. On the twenty-subject cohort the largest
confusion was with REM; with 76 it is with N2 — which is another reason not to have drawn
conclusions from twenty. The other large cells are REM read as N2 (1,459) and N2 read as
REM (1,697), the same boundary seen from both sides.

### One EEG derivation gets most of the way

The montage question, answered on identical rows and folds with random forest:

What each channel contributes, paired participant by participant across all 76 and
resampled over participants:

| removed | mean κ change | 95% interval | improved |
| --- | ---: | ---: | ---: |
| **EEG Fpz-Cz** | **−0.040** | [−0.055, −0.025] | 15 of 76 |
| EEG Pz-Oz | −0.030 | [−0.047, −0.014] | 15 of 76 |
| EOG horizontal | −0.029 | [−0.037, −0.021] | 13 of 76 |
| chin EMG | +0.000 | [−0.002, +0.002] | **38 of 76** |

**Fpz-Cz resolves here and did not at twenty subjects**, where it was −0.046 with an
interval of [−0.095, +0.003] — the largest contributor by mean, and not distinguishable
from nothing. With 76 the interval narrows from 0.098 wide to 0.030 and excludes zero.

**The three real channels are far closer than the small cohort suggested.** Twenty
subjects gave 0.046 / 0.029 / 0.017, an apparently clear ordering. Seventy-six give
0.040 / 0.030 / 0.029: Pz-Oz and the electro-oculogram are indistinguishable from each
other, and the frontal derivation leads by less than the earlier means implied.

**And the chin electromyogram is as clean a null as this project has produced**: no change
at all to three decimal places, an interval of ±0.002, and exactly half the cohort — 38 of
76 — scoring better without it.

Fpz-Cz is a voltage difference between two electrode sites, not one electrode; a reduced
montage is one derivation, not one contact. It contributes most of the three, consistent
with being the derivation single-channel staging is usually built on — but *most* is
0.040 against 0.030, not the factor of two the twenty-subject means suggested.

The single-derivation scores are being recomputed on the full cohort; the figures
previously quoted here (κ 0.657 alone against 0.710 for the montage) were measured on
twenty subjects and both of those numbers have since moved.

**Chin electromyography contributes nothing**, and now with a bound: ±0.002 of kappa, with
half the cohort improving without it. Muscle atonia is half the textbook definition of
REM, so that deserves an explanation rather than a shrug — in these files the EMG is
stored at 1 Hz as an envelope, giving 30 samples per epoch and three crude amplitude
features. The finding is about this recording of that channel, not about chin tone.


---

[← back to the README](../README.md)
