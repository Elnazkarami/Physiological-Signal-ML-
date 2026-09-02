"""Where the datasets are, asked rather than assumed.

These tests need files nobody can redistribute: the WESAD archive, obtained
from its authors, and the Sleep-EDF recordings from PhysioNet. Hard-coding one
person's Downloads folder made every dataset-dependent test skip silently on
any other machine, which meant the reproducibility the README claims could only
be exercised by the person who wrote it.

Each location is an environment variable with a conventional default, so the
suite runs unchanged where the defaults happen to hold and is one variable away
from running anywhere else:

    PHYSIOML_WESAD=/data/WESAD.zip
    PHYSIOML_SLEEP_EDF=/data/sleep-cassette
    PHYSIOML_CDFS=/src/clinical-data-fabric-system
"""

from __future__ import annotations

import os
from pathlib import Path


def _from_env(variable: str, default: Path) -> Path:
    found = os.environ.get(variable)
    return Path(found).expanduser() if found else default


#: The WESAD archive, read in place and never unpacked.
WESAD_ARCHIVE = _from_env("PHYSIOML_WESAD", Path.home() / "Downloads" / "WESAD.zip")

#: A directory of Sleep-EDF Expanded ``SC4*.edf`` files.
SLEEP_EDF_DIR = _from_env("PHYSIOML_SLEEP_EDF", Path("/tmp/pm/sleep-edf"))

#: A CDFS checkout, for the integration tests that run against a real one.
CDFS_REPO = _from_env(
    "PHYSIOML_CDFS", Path.home() / "Downloads" / "clinical-data-fabric-system"
)


def missing(path: Path, what: str, variable: str) -> str:
    """A skip reason that says how to make the test run."""
    return f"{what} not found at {path}; set {variable} to point at it"


WESAD_MISSING = missing(WESAD_ARCHIVE, "the WESAD archive", "PHYSIOML_WESAD")
SLEEP_MISSING = missing(SLEEP_EDF_DIR, "the Sleep-EDF recordings", "PHYSIOML_SLEEP_EDF")
CDFS_MISSING = missing(CDFS_REPO, "a CDFS checkout", "PHYSIOML_CDFS")
