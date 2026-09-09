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

98,023 epochs, 48 features, five stages — N2 36%, W 34%, REM 13%, N1 11%, N3 7%.

The full Sleep Cassette cohort. Class balance matters for reading the numbers below:
wake is a third of it and N3 under a tenth, and a subset of twenty subjects has a
noticeably different mix — 17% wake and 14% N3 — which is enough to move every score.

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

The majority row is 0.206 rather than exactly 0.200 because not every participant reaches
every stage; a fold missing one scores a constant answer at 1/4 rather than 1/5. It is
still zero on κ, which is the point of reporting κ.

Sleep staging is reported in Cohen's κ, not accuracy — the stages are unevenly distributed
enough that raw agreement flatters everything, which the majority row makes concrete at
38% accuracy and κ of exactly zero. **κ = 0.656 with 77% accuracy sits within
the range published for feature-based automatic staging under subject-wise validation**,
toward its lower end — which is the check that matters here: the pipeline is new, the task
is not, and a number far outside that range would mean something was wrong rather than
something was found.

### Reading the epochs either side

Every model above scores one epoch alone. A scorer does not: they read what came before and
after, and N1 is defined almost entirely by being a transition between the two things
around it. Handing each row its neighbours — the spectral balance, eye movement and muscle
tone from two epochs back and two forward, plus a centred rolling mean — turns 48 columns
into 143 and closes a good part of that gap.

| random forest | κ | accuracy | bal. accuracy | N1 recall |
| --- | ---: | ---: | ---: | ---: |
| one epoch alone | 0.656 ±0.123 | 0.762 | 0.673 | 0.319 |
| **with its neighbours** | **0.692 ±0.128** | **0.789** | **0.704** | **0.407** |

**+0.037 κ, 95% interval [+0.030, +0.044], improving 68 of 76 participants.** That is the
most clearly established improvement in this project, and it is the one that was predicted
in advance from what the model could not see rather than found by trying architectures.

The gain lands where it was expected to. **N1 recall rises from 0.319 to 0.407** — a
stage that is 11% of the epochs and was the worst-scored of the five. REM also improves,
0.657 to 0.707, which fits: REM and N1 are the two stages a scorer distinguishes partly by
what surrounds them. N3 does not move (0.634 to 0.626); slow-wave sleep looks like itself
regardless of its neighbours.

Context is added as ordinary columns, so the same classical models and the same evaluation
measure it. Participants are held out whole, so a row's neighbours always belong to the
same person — an epoch at the edge of a recording is padded with itself rather than with
somebody else's sleep.

There is also a [recurrent classifier](../src/physioml/models/sequence.py) that reads a
whole night as a sequence rather than taking neighbours as columns. It is built and tested;
its result on this cohort is not yet reported.

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

N3 is the hardest of the four common stages here at 0.641 averaged over participants: it
is under a tenth of the epochs, so there is comparatively little of it to learn from.

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

**This matrix is summed across folds; the per-stage recalls above are averaged over
participants.** The two answer different questions and give different numbers: N3 reads
0.801 pooled and 0.641 averaged, and the gap is the participants who have very little N3
and whom the model scores badly on it. The averaged figure says how this will do for a
person; the pooled one says how many epochs it got right.

Of the N1 epochs it gets wrong, **29% go to N2, 22% to wake and 17% to REM**. N1 is the
transition into sleep and it borders all three, which is why it is the stage every
automatic scorer struggles with and the one human scorers agree on least. The other large
cells are REM read as N2 (1,459) and N2 read as
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

**All three signal channels contribute, and by similar amounts.** The frontal derivation
leads at 0.040, with the occipital one and the electro-oculogram at 0.030 and 0.029 —
close enough that their intervals overlap, so the ordering between those two is not
something this cohort establishes.

**And the chin electromyogram is as clean a null as this project has produced**: no change
at all to three decimal places, an interval of ±0.002, and exactly half the cohort — 38 of
76 — scoring better without it.

Fpz-Cz is a voltage difference between two electrode sites, not one electrode; a reduced
montage is one derivation, not one contact. It contributes most of the three, consistent
with being the derivation single-channel staging is usually built on — but *most* is
0.040 against 0.030, not the factor of two the twenty-subject means suggested.

On the full cohort, each channel on its own:

| alone | features | bal. accuracy | κ | worst subject |
| --- | ---: | ---: | ---: | ---: |
| everything | 48 | 0.674 ±0.102 | **0.656** | 0.310 |
| **EEG Fpz-Cz** | 20 | 0.613 ±0.097 | **0.569** | 0.392 |
| EEG Pz-Oz | 20 | 0.593 ±0.113 | 0.552 | 0.122 |
| EOG horizontal | 5 | 0.569 ±0.094 | 0.473 | 0.280 |
| chin EMG | 3 | 0.276 ±0.056 | 0.118 | 0.177 |

**A single frontal derivation reaches κ 0.569 against 0.656 for the whole montage** —
87% of the agreement from one electrode pair, which is the finding a wearable would be
designed around.

Whether 0.569 is good enough is a question for a particular purpose, and this is a
promising reduced-montage result on one cohort rather than a demonstration of wearable
suitability. The two electroencephalogram derivations are close to each other alone
(0.569 against 0.552) — closer than the frontal one's larger removal cost implies, which
is what happens when two channels carry overlapping information: each is nearly sufficient
alone and neither is quite redundant.

One number in that table is worth pausing on. **Pz-Oz alone has a worst participant of
0.122** — well below the 0.200 a constant answer scores. A model can do worse than
chance for one person, and averaging it with seventy-five others hides that completely.

**Chin electromyography contributes nothing**, and now with a bound: ±0.002 of kappa, with
half the cohort improving without it. Muscle atonia is half the textbook definition of
REM, so that deserves an explanation rather than a shrug — in these files the EMG is
stored at 1 Hz as an envelope, giving 30 samples per epoch and three crude amplitude
features. The finding is about this recording of that channel, not about chin tone.


---

[← back to the README](../README.md)
