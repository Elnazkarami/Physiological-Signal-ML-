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
| one epoch alone | 0.656 ±0.122 | 0.762 | 0.672 | 0.309 |
| **with its neighbours** | **0.693 ±0.127** | **0.789** | **0.707** | **0.405** |

**+0.037 κ, 95% interval [+0.030, +0.044], improving 68 of 76 participants.** That is the
most clearly established improvement in this project, and it is the one that was predicted
in advance from what the model could not see rather than found by trying architectures.

The gain lands where it was expected to. **N1 recall rises from 0.309 to 0.405** — a
stage that is 11% of the epochs and was the worst-scored of the five. REM also improves,
0.651 to 0.706, which fits: REM and N1 are the two stages a scorer distinguishes partly by
what surrounds them. N3 does not move at all (0.642 either way); slow-wave sleep looks like
itself regardless of its neighbours.

Context is added as ordinary columns, so the same classical models and the same evaluation
measure it. Participants are held out whole, so a row's neighbours always belong to the
same person — an epoch at the edge of a recording is padded with itself rather than with
somebody else's sleep.

### Columns or a sequence: the same agreement, a different model

A [recurrent classifier](../src/physioml/models/sequence.py) reads a whole night as one
sequence rather than taking neighbours as columns. All three fitted on identical grouped
five-folds, so the comparison is not affected by the split:

| | κ | accuracy | bal. accuracy | fit time |
| --- | ---: | ---: | ---: | ---: |
| one epoch alone | 0.675 ±0.034 | 0.765 | 0.697 | 84 s |
| neighbours as columns | **0.708 ±0.036** | **0.790** | 0.725 | 154 s |
| recurrent, whole night | 0.698 ±0.029 | 0.774 | **0.779** | 740 s |

**On agreement they are indistinguishable.** Head to head the recurrent model is 0.013
lower with an interval of [−0.035, +0.008] and 32 of 76 participants improving. Both beat
the single-epoch model, the columns more convincingly: **+0.035 [+0.027, +0.043]** with 68
of 76 improving, against **+0.022 [+0.002, +0.041]** with 50 of 76.

**On the stages, they are not the same model at all.**

| recall | N1 | N2 | N3 | REM | wake |
| --- | ---: | ---: | ---: | ---: | ---: |
| one epoch alone | 0.320 | 0.817 | 0.811 | 0.646 | **0.891** |
| neighbours as columns | 0.381 | **0.845** | 0.803 | 0.696 | 0.899 |
| recurrent, whole night | **0.611** | 0.728 | **0.924** | **0.792** | 0.843 |

**The recurrent model finds N1 nearly twice as often as the context columns** — 0.611
against 0.381, on the stage every automatic scorer fails and human scorers agree on least.
It is better on N3 and REM too, and pays for it on N2 and wake, the two commonest stages.
Balanced accuracy, which weights the stages equally, reads 0.785 against 0.724; κ, which
accounts for how often each occurs, cannot see the trade at all.

So which is better depends on what the model is for. **For overall agreement with a
scorer, the columns win**: same κ, better accuracy, a third of the compute. **For finding
the transitions, the network wins clearly**, and by more than any feature change here
achieved.

Fit time is part of the answer. 740 seconds against 154 — and leave-one-subject-out was
abandoned for the recurrent model after fifteen CPU-hours without finishing, which is why
this table is five folds rather than seventy-six.

These figures were recomputed after two corrections: context and sequence runs are now cut
wherever epochs are not exactly adjacent, rather than treating the next surviving row as
the next epoch, and the recurrent model fits its normalisation and class weights on its
inner training participants rather than on all of them before choosing the split. The
first moved nothing measurable — it touched 0.08% of row pairs. The second took the
recurrent model from 0.708 to 0.698, which is what removing an optimistic early-stopping
signal looks like. Every command and manifest behind the table is in
[reproducing](reproducing.md).

## Carried to a different protocol

Everything above holds participants out of Sleep Cassette, which asks whether a model
transfers to a new person. Sleep Telemetry asks the harder question: a different protocol,
different participants, partly a medicated population, and a cohort holding a quarter of
Cassette's wake and more than twice its slow-wave sleep.

**One model, two test sets.** 19 Cassette participants are reserved and never trained on,
so the in-domain and cross-cohort scores come from the same fitted model, the same
training set of 57, and the same pooling.

| scored on | κ pooled | accuracy | κ averaged over participants | worst |
| --- | ---: | ---: | ---: | ---: |
| held-in Sleep Cassette (19) | **0.687** | 0.774 | 0.662 ±0.066 | 0.534 (SC77) |
| **Sleep Telemetry (22)** | **0.622** | 0.728 | 0.610 ±0.092 | 0.408 (ST18) |

**Crossing to a different protocol costs 0.065 κ pooled, or 0.052 averaged over
participants.** For a model that has never seen a Telemetry recording, a medicated
participant, or that recording setup, that is a real but modest loss — and it is the
strongest evidence here that these features describe sleep rather than describing one
dataset.

Two things keep it honest.

**This replaces an earlier figure of 0.019, which was not a controlled comparison.** That
one set a leave-one-subject-out mean over 76 participants against a single pooled score
from a different model trained on all 76. The difference between those mixes the change of
protocol with the training-set size, with participant-averaged against pooled κ, and with
the stage prevalences — three confounds and the effect of interest, in one subtraction.
The number here is about three times larger.

**Both figures are reported because they answer different questions.** Pooled κ counts
epochs; averaged κ counts participants and is the one that says how this behaves for a
person. The gap between them is wider on Telemetry (0.622 against 0.610) than in domain,
which is what a more variable cohort looks like.

