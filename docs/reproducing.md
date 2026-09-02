# Installing and reproducing every number

Every table in this repository comes from a command in this file.
## Install

```bash
pip install -e ".[dev]"          # core + tooling, no scientific stack
pip install -e ".[signal,ml]"    # signal processing and models
```

The core package has no runtime dependencies, and CI asserts that on every commit —
including that NumPy is absent from a `[dev]`-only install. To check the same thing
locally, block the scientific stack and run the core tests against it:

```bash
mkdir -p /tmp/nostack && cat > /tmp/nostack/sitecustomize.py <<'EOF'
import sys
from importlib.abc import MetaPathFinder
class Blocker(MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"numpy", "scipy", "sklearn"}:
            raise ImportError(f"{name} is blocked")
        return None
sys.meta_path.insert(0, Blocker())
EOF
PYTHONPATH=/tmp/nostack python -m pytest tests/test_core.py -q
```

This is worth having locally because the failure mode is invisible otherwise: pytest loads
`conftest.py` for every run, so a single numpy import placed there breaks the guarantee in
CI while passing everywhere else.

## Reproducing the result

```bash
pip install -e ".[signal,ml]"
python scripts/build_features.py ~/Downloads/WESAD.zip wesad_features.npz
python scripts/evaluate.py wesad_features.npz
python scripts/ablate.py wesad_features.npz --model logistic
python scripts/evaluate.py wesad_features.npz --calibrated

python scripts/build_features.py ~/Downloads/WESAD.zip wesad_fused.npz --device both
python scripts/ablate.py wesad_fused.npz --by device
python scripts/personalise.py wesad_features.npz

python scripts/build_features.py ~/Downloads/WESAD.zip wesad_qc.npz --quality
python scripts/compare.py wesad_features.npz
```

`compare.py` is the one to reach for before believing any difference in these tables. It
pairs two configurations participant by participant and bootstraps over **participants**,
which is the sample size — 8,057 rows look like a large sample and are fifteen people.

For the sleep recordings — the `sleep-cassette` files from
[Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/):

```bash
python scripts/build_sleep_features.py ~/sleep-edf sleep_features.npz
python scripts/evaluate.py sleep_features.npz --positive none
python scripts/ablate.py sleep_features.npz --by channel --positive none \
    --model random_forest
```

About 45 seconds to build 20 nights, and the archive is read in place — the reader
memory-maps each file and touches only the channels it is asked for.

Roughly 80 seconds to build the features and a couple of minutes to score all five
models. The archive is read as a stream and never extracted — 2.1 GB compressed against
about 17 GB unpacked.


---

[← back to the README](../README.md)
