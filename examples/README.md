# Examples

## quickstart.py

The whole pipeline on signals generated in the script — no dataset, no
credentials, no services.

Python 3.11 or newer:

```bash
git clone https://github.com/Elnazkarami/Physiological-Signal-ML-.git
cd Physiological-Signal-ML-
pip install -e ".[signal,ml]"
python examples/quickstart.py
```

Runs in a few seconds and prints five stages:

| stage | what it exercises |
| --- | --- |
| 1 — generate | six synthetic participants, four wrist sensors each at the rates and units an Empatica E4 reports |
| 2 — window, judge, measure | `epochs()`, then `assess()`, then `extract()` — in that order, which `extract()` enforces by requiring the verdict |
| 3 — assemble | `assemble()`, the same function the WESAD builder ends in, so every row keeps its feature vector, its windows and its recordings |
| 4 — evaluate | `leave_one_subject_out()` and the real metrics; no participant on both sides of a split |
| 5 — export | `export_fold()` — a `ModelArtifact` and one `Prediction` per scored row, each naming what produced it |

### Why the data is synthetic

WESAD must be requested from its authors and Sleep-EDF Expanded is several
gigabytes. Neither can stand behind an example someone runs to find out what
this library does. Everything operating on the generated signals is the same
code the measured results came from; only the recording is replaced.

### The scores printed here are meaningless

They describe a difference the script writes into the data on purpose —
a 9 bpm faster pulse in the stress blocks, 0.7 µS more skin conductance, 0.2 °C
less skin temperature and three times the movement, each smaller than the
spread it puts between participants. The script says so at every stage and
again at the end.

**The measured results are in [the README](../README.md#results),
[docs/wesad-stress.md](../docs/wesad-stress.md) and
[docs/sleep.md](../docs/sleep.md).** Nothing in `examples/` belongs in a
summary of what this project achieves.

One thing in the output is worth reading, though, and it is not a score: one
participant usually lands at 0.500 balanced accuracy with an AUC near 1.000.
That is the finding in [docs/wesad-stress.md](../docs/wesad-stress.md) arriving
on its own — the model ranks that participant's windows perfectly and the
cohort's decision threshold sits to one side of all of them. A chance-level
score that is a threshold, not a failure to learn.

## expected/

The output of the commands above, produced by running them rather than written
to look plausible:

```
expected/
├── quickstart_output.txt      # python examples/quickstart.py
└── prediction_trace.json      # ... --write-trace examples/expected/prediction_trace.json
```

Regenerate both with:

```bash
python examples/quickstart.py --write-trace examples/expected/prediction_trace.json \
    > examples/expected/quickstart_output.txt
```

The run is seeded, so on any one machine it is byte-for-byte reproducible, and
`tests/test_examples.py` asserts that. If a change to windowing, quality
control, feature extraction or the export alters the chain, that test fails.

### The identifiers will not match on your machine, and that is correct

Run this yourself and **none** of the identifiers will match the ones committed
here, while every measurement will.

The identifiers are content hashes, and somewhere in each one's ancestry is a
floating-point measurement. A recording's covers a sha256 of its samples; a
window's covers its recording; a feature vector's covers the feature values;
a prediction's covers all of them. Different builds of NumPy, SciPy, a BLAS or
even libm's `sin` agree on those to about eleven significant figures and
disagree in the last bits — this project's CI produced
`2.2163459163476735e-05` where the machine that committed these files produced
`2.2163459163323433e-05`. A hash has no notion of nearly.

That is the cost of identifiers that fingerprint the signal instead of merely
describing it, and it is the right trade: an identifier that ignored the
samples would be the same for a recording and its correction, which is the one
thing provenance must never do.

So the guarantee is: **identical within an environment, equal to a tolerance
across environments.** The tests assert exactly that — the structure and the
records field by field, the numbers to a tolerance, the identifiers masked —
rather than a stronger claim that fails on the second machine to try it. For
tracing a prediction back to its inputs, which is what this is for, within-
environment identity is the property that matters: a prediction and the
features behind it are hashed by the same process on the same machine.

`prediction_trace.json` holds the full fold with every reference resolved:
one artifact, 62 predictions, the 62 feature vectors behind them, the 248 signal
windows behind those, the 4 recordings behind the windows, and the training run
that fitted the model. It contains the chain rather than naming it, and
`export.unresolved()` returns the references that lead nowhere — empty here, and
asserted by a test rather than assumed.

(`export.as_json()` is the other shape: predictions and references only, which
is what belongs beside a score in a results directory where the windows are
already on disk. The example prints one of those too, labelled.)

### What each identifier actually covers

Every identifier is a content hash, but they are hashes of different things, and
the distinction matters more than the slogan:

| identifier | covers |
| --- | --- |
| `rec-…` | the recording's metadata — subject, device, rate, interval — **and a sha256 of the samples** |
| `win-…` | the recording, the sample bounds, the interval and the preprocessing |
| `fvec-…` | the feature values, in a fixed column order, under a named feature set |
| `model-…` | the model, version, training run, expected columns **and a digest of the fitted coefficients** |
| `pred-…` | all of the above, plus the predicted class and its probability |

So recomputing any of them from the same inputs reproduces the identifier, and
changing an input it covers cannot. What they do **not** do is certify that the
samples on disk are unchanged since they were read: `source_hash` is a
fingerprint taken at read time and is only as good as that moment. This is a
provenance chain, not a tamper-evident log.

`source_fact_ids` is empty here because these observations did not come through
CDFS; with a CDFS deployment it carries the facts the features rest on, which is
what makes a correction cascade — see
[docs/architecture.md](../docs/architecture.md).

### Times

Predictions carry both. `window_start`/`window_end` are wall-clock UTC, read
from the recording's own start time — the example's signals begin at 09:00 on
2026-01-01, and that is what the published trace says. `window_start_seconds` is
the offset within the recording, which is what the splits are made on and what
stays true for a dataset carrying no acquisition timestamps at all (WESAD has
none; its reader places sessions at a documented fixed origin).
