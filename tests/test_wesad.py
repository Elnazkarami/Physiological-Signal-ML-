"""The WESAD adapter and windowing, against the real archive where present.

Structural behaviour is tested on synthetic data so it runs anywhere; the tests
that need the dataset are skipped without it rather than mocked, since a mock of
a file format only asserts what this file already believes about it.
"""

from __future__ import annotations

import pickle
import zipfile
from datetime import UTC, datetime

import numpy as np
import pytest

from physioml.core.recording import Modality
from physioml.io.wesad import (
    CONDITIONS,
    WESAD,
    WESADError,
    condition_of,
)
from physioml.peripheral.windowing import epochs, label_counts
from tests.paths import WESAD_ARCHIVE as ARCHIVE
from tests.paths import WESAD_MISSING

needs_wesad = pytest.mark.skipif(not ARCHIVE.is_file(), reason=WESAD_MISSING)


# ── labelling ───────────────────────────────────────────────────────────────


def test_a_stretch_inside_one_condition_takes_its_label():
    assert condition_of(np.full(700, 2)) == "stress"
    assert condition_of(np.full(700, 1)) == "baseline"


def test_a_stretch_spanning_two_conditions_has_no_label():
    """Half stress is not stress; training on it teaches the transition."""
    straddling = np.concatenate([np.full(350, 1), np.full(350, 2)])
    assert condition_of(straddling) is None


def test_the_periods_wesad_says_to_ignore_are_unlabelled():
    for code in (0, 5, 6, 7):
        assert condition_of(np.full(10, code)) is None, code
    assert set(CONDITIONS) == {1, 2, 3, 4}


# ── the unpickler ───────────────────────────────────────────────────────────


def test_the_reader_refuses_to_execute_code_while_unpickling(tmp_path):
    """A pickle can call anything on load. This one may only rebuild arrays."""
    hostile = tmp_path / "WESAD.zip"
    with zipfile.ZipFile(hostile, "w") as bundle:
        bundle.writestr("WESAD/S2/S2.pkl", pickle.dumps({"cmd": print}))

    with pytest.raises(pickle.UnpicklingError, match="does not run code"):
        WESAD(hostile).read("S2")


def test_a_missing_archive_is_reported_at_construction(tmp_path):
    with pytest.raises(WESADError, match="no archive"):
        WESAD(tmp_path / "absent.zip")


def test_an_unknown_subject_names_the_ones_present(tmp_path):
    archive = tmp_path / "WESAD.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("WESAD/S2/S2.pkl", b"")
    with pytest.raises(WESADError, match="present: S2"):
        WESAD(archive).read("S99")


# ── windowing ───────────────────────────────────────────────────────────────


class FakeSubject:
    """A subject shaped like WESAD, small enough to reason about."""

    def __init__(self, seconds: int = 120):
        from physioml.core.recording import Recording
        from physioml.io.wesad import LABEL_HZ, WRIST_HZ

        self.subject_id = "S00"
        self.labels = np.concatenate(
            [
                np.full(round(seconds * LABEL_HZ / 2), 1),
                np.full(round(seconds * LABEL_HZ / 2), 2),
            ]
        )
        self.signals = {
            name: np.zeros((round(seconds * hz), 3 if name == "ACC" else 1))
            for name, hz in WRIST_HZ.items()
        }
        self.recordings = {
            name: Recording.create(
                study_id="WESAD",
                subject_id="S00",
                modality=Modality.ACC if name == "ACC" else Modality[name],
                sampling_rate_hz=hz,
                start_time=datetime(2017, 1, 1, tzinfo=UTC),
                duration_seconds=float(seconds),
                channels=("x", "y", "z") if name == "ACC" else (name,),
            )
            for name, hz in WRIST_HZ.items()
        }

    @property
    def duration_seconds(self) -> float:
        from physioml.io.wesad import LABEL_HZ

        return len(self.labels) / LABEL_HZ


def test_every_modality_covers_the_same_interval():
    """Rates differ; the interval must not."""
    made = epochs(FakeSubject(), length_seconds=60.0, stride_seconds=60.0)
    first = made[0]
    assert first.samples["BVP"].shape[0] == 3840  # 60s @ 64Hz
    assert first.samples["ACC"].shape[0] == 1920  # 60s @ 32Hz
    assert first.samples["EDA"].shape[0] == 240  # 60s @ 4Hz
    assert all(w.duration_seconds == 60.0 for w in first.windows.values())


def test_windows_share_a_start_time_across_modalities():
    made = epochs(FakeSubject(), length_seconds=30.0, stride_seconds=30.0)
    starts = {w.start_time for w in made[1].windows.values()}
    assert len(starts) == 1


def test_a_window_straddling_the_boundary_is_produced_but_unlabelled():
    """Kept rather than dropped, so how many were lost stays countable."""
    made = epochs(FakeSubject(120), length_seconds=60.0, stride_seconds=30.0)
    assert [e.label for e in made] == ["baseline", None, "stress"]
    assert label_counts(made) == {"baseline": 1, "unlabelled": 1, "stress": 1}


def test_stride_controls_overlap():
    subject = FakeSubject(120)
    assert len(epochs(subject, length_seconds=60.0, stride_seconds=60.0)) == 2
    assert len(epochs(subject, length_seconds=60.0, stride_seconds=10.0)) == 7


def test_a_stride_longer_than_the_window_is_refused():
    """It would step over signal that was never examined."""
    with pytest.raises(ValueError, match="skip signal"):
        epochs(FakeSubject(), length_seconds=10.0, stride_seconds=30.0)


@pytest.mark.parametrize(("length", "stride"), [(0, 5), (60, 0), (-1, 5)])
def test_nonsense_geometry_is_refused(length, stride):
    with pytest.raises(ValueError, match="must be positive"):
        epochs(FakeSubject(), length_seconds=length, stride_seconds=stride)


# ── against the real archive ────────────────────────────────────────────────


@needs_wesad
def test_the_archive_holds_the_published_cohort():
    assert WESAD(ARCHIVE).subjects() == [
        f"S{n}" for n in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17)
    ]


@needs_wesad
def test_a_real_subject_reads_with_the_rates_the_hardware_uses():
    data = WESAD(ARCHIVE).read("S2")
    rates = {n: r.sampling_rate_hz for n, r in data.recordings.items()}
    assert rates == {"ACC": 32.0, "BVP": 64.0, "EDA": 4.0, "TEMP": 4.0}
    assert data.duration_seconds > 3600, "WESAD sessions run about 100 minutes"
    for name, array in data.signals.items():
        expected = round(data.duration_seconds * rates[name])
        assert abs(array.shape[0] - expected) <= 1, name


@needs_wesad
def test_a_real_subject_yields_all_four_conditions():
    data = WESAD(ARCHIVE).read("S2")
    counts = label_counts(epochs(data))
    assert set(CONDITIONS.values()) <= set(counts)
    assert counts["unlabelled"] > 0, "transitions and recovery are excluded"
