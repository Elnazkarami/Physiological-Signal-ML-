# Examples

## quickstart.py

The whole pipeline on signals generated in the script — no dataset, no
credentials, no services.

```bash
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

Run this yourself and the window and recording identifiers will be the ones
committed here, while `prediction_id` and the feature-vector id will differ.

Those identifiers are content hashes of feature values, and feature values are
floating-point results of filtering and spectral estimation. Different builds
of NumPy, SciPy or a BLAS agree on them to about eleven significant figures and
disagree in the last bits — this project's CI produced
`2.2163459163476735e-05` where the machine that committed these files produced
`2.2163459163323433e-05`. A hash has no notion of nearly. Window and recording
identifiers are stable because they hash metadata — the subject, the device,
the interval, the sampling rate — and nothing measured.

So the guarantee is this: **identical within an environment, equal to a
tolerance across environments.** That is what content-hashed provenance over
floating-point measurements can offer, and the tests assert exactly it rather
than a stronger claim that would fail on the second machine to try it. For
tracing a prediction back to its inputs, which is what this is for, within-
environment identity is the property that matters: a prediction and the
features behind it are hashed by the same process on the same machine.

`prediction_trace.json` holds the full fold — one artifact and 62 predictions.
Each prediction names its own feature vector, the four signal windows behind it,
the feature-set version, and the training run and model that produced it. Every
identifier is a content hash, so the same signal through the same feature set
and the same model yields the same identifier, and a changed input cannot keep
the old one. `source_fact_ids` is empty here because these observations did not
come through CDFS; with a CDFS deployment it carries the facts the features rest
on, which is what makes a correction cascade — see
[docs/architecture.md](../docs/architecture.md).
