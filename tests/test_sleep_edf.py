"""The Sleep-EDF adapter: pairing, scoring, and what gets trimmed away.

Nights are constructed here rather than read from the dataset, so the expected
stage of every epoch is known. The tests that need the real recordings are
skipped without them.
"""

from __future__ import annotations

import numpy as np
import pytest

from physioml.io.sleep_edf import (
    EPOCH_SECONDS,
    STAGES,
    SleepEDF,
    SleepEDFError,
    hypnogram,
)
from tests.edf_writer import tal, write_edf

RATE = 100.0


def night(directory, subject: str, night_number: int, stages: list[tuple[str, float]]):
    """One PSG and its hypnogram, with the stages given as (name, seconds)."""
    total = sum(seconds for _, seconds in stages)
    samples = int(total * RATE)
    rng = np.random.default_rng(0)
    write_edf(
        directory / f"SC4{subject}{night_number}E0-PSG.edf",
        {
            "EEG Fpz-Cz": rng.normal(0, 20, samples),
            "EEG Pz-Oz": rng.normal(0, 20, samples),
            "EOG horizontal": rng.normal(0, 30, samples),
            "EMG submental": rng.normal(0, 5, samples),
        },
        rates=dict.fromkeys(
            ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"], RATE
        ),
        record_seconds=EPOCH_SECONDS,
    )

    body = b""
    at = 0.0
    for name, seconds in stages:
        body += tal(at, seconds, name)
        at += seconds
    write_edf(
        directory / f"SC4{subject}{night_number}EC-Hypnogram.edf",
        {"A": np.zeros(int(total / EPOCH_SECONDS))},
        rates={"A": 1.0 / EPOCH_SECONDS},
        record_seconds=EPOCH_SECONDS,
        reserved="EDF+C",
        annotations=body,
    )


# ── the hypnogram ───────────────────────────────────────────────────────────


def test_a_run_is_expanded_to_the_epochs_it_covers(tmp_path):
    """The file stores an hour of stage 2 as one annotation, not 120 of them."""
    night(tmp_path, "00", 1, [("Sleep stage W", 60.0), ("Sleep stage 2", 300.0)])
    stages, unscored = hypnogram(tmp_path / "SC4001EC-Hypnogram.edf")
    assert list(stages) == ["W"] * 2 + ["N2"] * 10
    assert unscored == 0


def test_stages_three_and_four_are_one_stage(tmp_path):
    """Scored under Rechtschaffen and Kales, reported as modern practice does."""
    night(tmp_path, "00", 1, [("Sleep stage 3", 60.0), ("Sleep stage 4", 60.0)])
    stages, _ = hypnogram(tmp_path / "SC4001EC-Hypnogram.edf")
    assert set(stages) == {"N3"}
    assert STAGES["Sleep stage 3"] == STAGES["Sleep stage 4"] == "N3"


def test_an_unscored_epoch_is_not_a_sixth_stage(tmp_path):
    night(
        tmp_path,
        "00",
        1,
        [("Sleep stage W", 60.0), ("Sleep stage ?", 60.0), ("Sleep stage 2", 60.0)],
    )
    stages, unscored = hypnogram(tmp_path / "SC4001EC-Hypnogram.edf")
    assert list(stages) == ["W", "W", "", "", "N2", "N2"]
    assert unscored == 2


def test_a_file_with_no_annotations_is_refused(tmp_path):
    write_edf(tmp_path / "empty.edf", {"A": np.zeros(10)})
    with pytest.raises(SleepEDFError, match="carries no annotations"):
        hypnogram(tmp_path / "empty.edf")


# ── pairing and naming ──────────────────────────────────────────────────────


def test_nights_are_paired_with_their_scoring(tmp_path):
    night(tmp_path, "00", 1, [("Sleep stage W", 60.0), ("Sleep stage 2", 600.0)])
    night(tmp_path, "00", 2, [("Sleep stage W", 60.0), ("Sleep stage 2", 600.0)])
    night(tmp_path, "05", 1, [("Sleep stage W", 60.0), ("Sleep stage 2", 600.0)])
    source = SleepEDF(tmp_path)
    assert source.nights() == [("SC00", 1), ("SC00", 2), ("SC05", 1)]


def test_two_nights_of_one_person_are_one_subject(tmp_path):
    """Treating them as two would put the same person on both sides of every
    split -- the leak the whole evaluation exists to prevent, arriving through
    the file naming."""
    night(tmp_path, "00", 1, [("Sleep stage W", 60.0), ("Sleep stage 2", 600.0)])
    night(tmp_path, "00", 2, [("Sleep stage W", 60.0), ("Sleep stage 2", 600.0)])
    assert SleepEDF(tmp_path).subjects() == ["SC00"]


def test_a_recording_without_its_scoring_is_not_offered(tmp_path):
    night(tmp_path, "00", 1, [("Sleep stage W", 60.0), ("Sleep stage 2", 600.0)])
    (tmp_path / "SC4001EC-Hypnogram.edf").unlink()
    assert SleepEDF(tmp_path).nights() == []


def test_a_missing_night_says_what_is_there(tmp_path):
    night(tmp_path, "00", 1, [("Sleep stage W", 60.0), ("Sleep stage 2", 600.0)])
    with pytest.raises(SleepEDFError, match="no night 2"):
        SleepEDF(tmp_path).read("SC00", 2)


def test_a_directory_that_is_not_one_is_refused(tmp_path):
    with pytest.raises(SleepEDFError, match="not a directory"):
        SleepEDF(tmp_path / "nowhere")


# ── trimming ────────────────────────────────────────────────────────────────


def test_the_recorder_running_before_bed_is_trimmed_away(tmp_path):
    """Left alone, wake is three-quarters of the epochs and a classifier that
    answers "awake" scores extremely well."""
    night(
        tmp_path,
        "00",
        1,
        [
            ("Sleep stage W", 3600.0),
            ("Sleep stage 2", 1800.0),
            ("Sleep stage W", 3600.0),
        ],
    )
    record = SleepEDF(tmp_path).read("SC00", 1, margin_minutes=5.0)
    counts = record.counts()
    assert counts["N2"] == 60
    assert counts["W"] == 20, "five minutes each side, at 30 seconds an epoch"
    assert record.trimmed_epochs == 220


def test_the_margin_is_honoured(tmp_path):
    night(
        tmp_path,
        "00",
        1,
        [("Sleep stage W", 1800.0), ("Sleep stage 2", 600.0), ("Sleep stage W", 1800.0)],
    )
    source = SleepEDF(tmp_path)
    narrow = source.read("SC00", 1, margin_minutes=0.0)
    wide = source.read("SC00", 1, margin_minutes=10.0)
    assert narrow.counts().get("W", 0) == 0
    assert wide.counts()["W"] == 40
    assert wide.epochs > narrow.epochs


def test_a_night_with_no_sleep_is_kept_whole_rather_than_emptied(tmp_path):
    night(tmp_path, "00", 1, [("Sleep stage W", 600.0)])
    record = SleepEDF(tmp_path).read("SC00", 1)
    assert record.epochs == 20
    assert record.trimmed_epochs == 0


# ── the signals ─────────────────────────────────────────────────────────────


def test_the_signals_are_cut_to_the_same_span_as_the_stages(tmp_path):
    night(
        tmp_path,
        "00",
        1,
        [("Sleep stage W", 1800.0), ("Sleep stage 2", 600.0), ("Sleep stage W", 1800.0)],
    )
    record = SleepEDF(tmp_path).read("SC00", 1, margin_minutes=5.0)
    expected = record.epochs * EPOCH_SECONDS * RATE
    for label, samples in record.signals.items():
        assert samples.size == pytest.approx(expected, rel=0.01), label
    assert record.offset_seconds == pytest.approx(1500.0)


def test_a_recording_missing_every_requested_channel_is_refused(tmp_path):
    night(tmp_path, "00", 1, [("Sleep stage W", 60.0), ("Sleep stage 2", 600.0)])
    with pytest.raises(SleepEDFError, match="has none of"):
        SleepEDF(tmp_path).read("SC00", 1, channels=("EEG C3-A2",))


# ── a build survives one unreadable night ───────────────────────────────────


def test_a_truncated_recording_is_skipped_and_counted(tmp_path):
    """One bad night should not cost the other seventy-seven.

    A build over the full cohort takes minutes and reads gigabytes; losing all
    of it at the end because one file was half-downloaded is a bad trade for a
    guarantee nothing needed.
    """
    from physioml.neural.dataset import build_sleep

    night(tmp_path, "00", 1, [("Sleep stage W", 600.0), ("Sleep stage 2", 1200.0)])
    night(tmp_path, "01", 1, [("Sleep stage W", 600.0), ("Sleep stage 2", 1200.0)])
    whole = (tmp_path / "SC4011E0-PSG.edf").read_bytes()
    (tmp_path / "SC4011E0-PSG.edf").write_bytes(whole[: len(whole) // 2])

    table = build_sleep(tmp_path, margin_minutes=5.0)
    assert set(table.subjects.tolist()) == {"SC00"}
    assert table.qc_codes["unreadable_nights"] == 1


def test_a_build_where_nothing_is_readable_still_raises(tmp_path):
    """Skipping is for a night, not for the whole cohort."""
    from physioml.neural.dataset import build_sleep

    night(tmp_path, "00", 1, [("Sleep stage W", 600.0), ("Sleep stage 2", 1200.0)])
    whole = (tmp_path / "SC4001E0-PSG.edf").read_bytes()
    (tmp_path / "SC4001E0-PSG.edf").write_bytes(whole[: len(whole) // 2])

    with pytest.raises(ValueError, match="no scored epoch"):
        build_sleep(tmp_path, margin_minutes=5.0)


# ── verifying a directory of recordings ─────────────────────────────────────


def test_a_truncated_file_is_not_a_usable_night(tmp_path):
    """Counting files and checking sizes both miss this: a half-downloaded
    night is megabytes of perfectly good bytes with a header that says how
    many more there should have been."""
    from physioml.io.edf import EDF, EDFError

    night(tmp_path, "00", 1, [("Sleep stage W", 600.0), ("Sleep stage 2", 1200.0)])
    path = tmp_path / "SC4001E0-PSG.edf"
    whole = path.read_bytes()
    path.write_bytes(whole[: len(whole) // 2])

    assert path.stat().st_size > 0, "it is not empty, which is the trap"
    with pytest.raises(EDFError, match="truncated"):
        EDF(path)


def test_a_recording_without_its_scoring_is_not_a_usable_night(tmp_path):
    night(tmp_path, "00", 1, [("Sleep stage W", 600.0), ("Sleep stage 2", 1200.0)])
    (tmp_path / "SC4001EC-Hypnogram.edf").unlink()
    assert SleepEDF(tmp_path).nights() == []


# ── two cohorts, and the collision between them ─────────────────────────────


def telemetry_night(directory, subject: str, night_number: int, stages):
    """A Sleep Telemetry recording, which is named differently and is a
    different person from the Cassette subject with the same number."""
    total = sum(seconds for _, seconds in stages)
    rng = np.random.default_rng(1)
    write_edf(
        directory / f"ST7{subject}{night_number}J0-PSG.edf",
        {
            "EEG Fpz-Cz": rng.normal(0, 20, int(total * RATE)),
            "EEG Pz-Oz": rng.normal(0, 20, int(total * RATE)),
            "EOG horizontal": rng.normal(0, 30, int(total * RATE)),
            "EMG submental": rng.normal(0, 5, int(total * RATE)),
        },
        rates=dict.fromkeys(
            ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"], RATE
        ),
        record_seconds=EPOCH_SECONDS,
    )
    body, at = b"", 0.0
    for name, seconds in stages:
        body += tal(at, seconds, name)
        at += seconds
    write_edf(
        directory / f"ST7{subject}{night_number}JP-Hypnogram.edf",
        {"A": np.zeros(int(total / EPOCH_SECONDS))},
        rates={"A": 1.0 / EPOCH_SECONDS},
        record_seconds=EPOCH_SECONDS,
        reserved="EDF+C",
        annotations=body,
    )


def test_the_two_cohorts_are_both_read(tmp_path):
    stages = [("Sleep stage W", 600.0), ("Sleep stage 2", 1200.0)]
    night(tmp_path, "01", 1, stages)
    telemetry_night(tmp_path, "01", 1, stages)
    assert SleepEDF(tmp_path).subjects() == ["SC01", "ST01"]


def test_the_same_number_in_each_cohort_is_two_participants(tmp_path):
    """The leak this prevents: Sleep Cassette 01 and Sleep Telemetry 01 share
    a number and nothing else. Merged under one identifier,
    leave-one-subject-out would train on somebody it was about to score."""
    stages = [("Sleep stage W", 600.0), ("Sleep stage 2", 1200.0)]
    night(tmp_path, "01", 1, stages)
    telemetry_night(tmp_path, "01", 1, stages)

    source = SleepEDF(tmp_path)
    assert len(source.subjects()) == 2
    assert source.read("SC01", 1).subject_id == "SC01"
    assert source.read("ST01", 1).subject_id == "ST01"

    # And the recordings really are different, so the check has teeth.
    assert not np.array_equal(
        source.read("SC01", 1).signals["EEG Fpz-Cz"],
        source.read("ST01", 1).signals["EEG Fpz-Cz"],
    )


def test_a_table_built_across_cohorts_keeps_them_apart(tmp_path):
    from physioml.neural.dataset import build_sleep

    stages = [("Sleep stage W", 600.0), ("Sleep stage 2", 1200.0)]
    night(tmp_path, "01", 1, stages)
    telemetry_night(tmp_path, "01", 1, stages)

    table = build_sleep(tmp_path, margin_minutes=5.0)
    assert set(table.subject_ids) == {"SC01", "ST01"}
