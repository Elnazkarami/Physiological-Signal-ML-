"""The chest device: electrocardiogram, respiration, muscle activity.

The cardiac tests assert against signals whose answer is known by construction
-- a beat train built with a stated SDNN must come back at that SDNN -- which
is the check the wrist pulse failed and the reason its variability features
were removed. Two of the tests here exist because a bug survived that
construction and only real data exposed it; they are marked where they are.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from physioml.dataset import build
from physioml.peripheral.chest import (
    CHEST_CHECKS,
    CHEST_EXTRACTORS,
    CHEST_FEATURES_BY_SIGNAL,
    CHEST_POLICY,
    MALIK_TOLERANCE,
    check_ecg,
    check_resp,
    clean_intervals,
    detect_r_peaks,
    ecg_features,
    emg_features,
    resp_features,
    rr_intervals_ms,
)

RATE = 700.0


def beat_train(
    bpm: float = 70.0, sdnn_ms: float = 50.0, seconds: float = 60.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """An electrocardiogram with a stated mean rate and stated variability.

    R wave, S trough and T wave, so a detector cannot pass by finding any sharp
    thing in the window.
    """
    rng = np.random.default_rng(seed)
    mean_rr = 60000.0 / bpm
    intervals: list[float] = []
    while sum(intervals) < seconds * 1000:
        intervals.append(max(float(rng.normal(mean_rr, sdnn_ms)), 300.0))
    positions = np.cumsum(intervals)[:-1] / 1000.0

    samples = int(seconds * RATE)
    time = np.arange(samples) / RATE
    signal = np.zeros(samples)
    for at in positions:
        if at >= seconds:
            break
        signal += np.exp(-((time - at) ** 2) / (2 * 0.010**2))
        signal -= 0.25 * np.exp(-((time - (at + 0.03)) ** 2) / (2 * 0.015**2))
        signal += 0.15 * np.exp(-((time - (at + 0.25)) ** 2) / (2 * 0.040**2))
    return signal, np.array(intervals[: len(positions)])


def breathing(bpm: float = 15.0, seconds: float = 60.0, drift: float = 0.0):
    time = np.arange(int(seconds * RATE)) / RATE
    signal = np.sin(2 * np.pi * (bpm / 60.0) * time)
    return signal + drift * time


# ── the electrocardiogram, against known answers ────────────────────────────


@pytest.mark.parametrize(("bpm", "sdnn"), [(55.0, 30.0), (70.0, 50.0), (95.0, 20.0)])
def test_variability_is_recovered_from_a_train_built_to_have_it(bpm, sdnn):
    """What the wrist could not do: 236 ms against a true 65."""
    signal, intervals = beat_train(bpm, sdnn, seed=int(bpm))
    found = ecg_features(signal, RATE)
    assert found["chest_hr_mean"] == pytest.approx(bpm, abs=2.0)
    assert found["chest_sdnn"] == pytest.approx(float(np.std(intervals, ddof=1)), rel=0.2)


def test_the_beat_count_matches_the_beats_put_there():
    signal, intervals = beat_train(70.0, 40.0)
    peaks = detect_r_peaks(signal, RATE)
    assert abs(peaks.size - (len(intervals) + 1)) <= 2


def test_the_rate_is_estimated_from_the_envelope_not_the_filtered_signal():
    """A real bug: the band-pass is 5-15 Hz, so the dominant frequency of what
    comes out of it is the QRS band. On WESAD it reported 193 bpm for a 70 bpm
    heart, which set the refractory spacing to a third of what it should be."""
    from physioml.peripheral.chest import qrs_envelope
    from physioml.peripheral.preprocessing import dominant_rate_bpm

    signal, _ = beat_train(70.0, 40.0)
    filtered, envelope = qrs_envelope(signal, RATE)
    assert dominant_rate_bpm(envelope, RATE, 40.0, 200.0) == pytest.approx(70.0, abs=6.0)
    assert dominant_rate_bpm(filtered, RATE, 40.0, 200.0) > 150.0


def test_impossible_intervals_are_discarded():
    peaks = np.array([0, 700, 1400, 1410, 2100])  # one 14 ms "beat"
    intervals = rr_intervals_ms(peaks, RATE)
    assert intervals.size == 3
    assert intervals.min() >= 300.0


# ── interval cleaning ───────────────────────────────────────────────────────


def test_cleaning_removes_a_merged_interval_and_keeps_the_rest():
    intervals = np.array([850.0, 860.0, 1700.0, 855.0, 845.0])  # a missed beat
    keep = clean_intervals(intervals)
    assert not keep[2]
    assert keep.sum() == 4


def test_cleaning_compares_against_the_last_accepted_interval():
    """Otherwise one artifact drags its innocent successor out with it."""
    intervals = np.array([850.0, 1700.0, 860.0])
    keep = clean_intervals(intervals)
    assert list(keep) == [True, False, True]


def test_ordinary_variation_survives_cleaning():
    """The rule that deleted 61 of 200 good wrist windows must not do that here."""
    rng = np.random.default_rng(3)
    intervals = rng.normal(850.0, 45.0, 80)
    assert clean_intervals(intervals).mean() > 0.9


def test_successive_differences_do_not_span_a_removed_interval():
    """Differencing the retained series alone invents a difference exactly
    where an artifact was taken out."""
    signal, _ = beat_train(70.0, 30.0, seed=11)
    clean = ecg_features(signal, RATE)
    damaged = signal.copy()
    # Delete one beat's worth of signal, which merges two intervals.
    at = int(RATE * 12.0)
    damaged[at : at + int(RATE * 0.2)] = 0.0
    after = ecg_features(damaged, RATE)
    assert after["chest_rmssd"] == pytest.approx(clean["chest_rmssd"], rel=0.35)


@pytest.mark.parametrize("tolerance", [MALIK_TOLERANCE])
def test_the_cleaning_tolerance_is_the_conventional_one(tolerance):
    assert tolerance == pytest.approx(0.20)


# ── respiration ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bpm", [9.0, 15.0, 24.0])
def test_a_breathing_rate_is_recovered(bpm):
    found = resp_features(breathing(bpm), RATE)
    assert found["chest_resp_rate_bpm"] == pytest.approx(bpm, abs=1.5)


def test_a_drifting_baseline_does_not_become_the_breathing_rate():
    """A real bug, and it produced a constant column rather than a wrong one:
    every window of every subject read 6.00 breaths per minute, then 13.00
    after the band was narrowed. Both were the filter's own shape."""
    found = resp_features(breathing(18.0, drift=40.0), RATE)
    assert found["chest_resp_rate_bpm"] == pytest.approx(18.0, abs=2.0)


def test_a_signal_with_no_rhythm_reports_no_rate():
    rng = np.random.default_rng(0)
    flat = np.zeros(int(RATE * 60))
    assert "flatline" in check_resp(flat, RATE, CHEST_POLICY)
    assert "chest_resp_rate_bpm" not in resp_features(flat, RATE)
    # Amplitude is still reported for a signal that has one.
    assert "chest_resp_amplitude_sd" in resp_features(
        rng.normal(0, 1, int(RATE * 60)), RATE
    )


# ── muscle activity ─────────────────────────────────────────────────────────


def test_muscle_activity_rises_with_amplitude():
    rng = np.random.default_rng(0)
    quiet = emg_features(rng.normal(0, 0.01, int(RATE * 60)), RATE)
    tense = emg_features(rng.normal(0, 0.05, int(RATE * 60)), RATE)
    assert tense["chest_emg_rms"] > quiet["chest_emg_rms"] * 3
    assert tense["chest_emg_p95"] > quiet["chest_emg_p95"]


def test_muscle_activity_ignores_a_constant_offset():
    """Electrode offset is not tension."""
    rng = np.random.default_rng(1)
    signal = rng.normal(0, 0.02, int(RATE * 60))
    assert emg_features(signal, RATE)["chest_emg_rms"] == pytest.approx(
        emg_features(signal + 5.0, RATE)["chest_emg_rms"]
    )


# ── quality control ─────────────────────────────────────────────────────────


def test_a_flat_electrocardiogram_is_refused():
    assert check_ecg(np.zeros(int(RATE * 60)), RATE, CHEST_POLICY) == ["flatline"]


def test_a_good_electrocardiogram_passes():
    signal, _ = beat_train(70.0, 40.0)
    assert check_ecg(signal, RATE, CHEST_POLICY) == []


def test_noise_is_not_mistaken_for_a_heart():
    rng = np.random.default_rng(7)
    codes = check_ecg(rng.normal(0, 1, int(RATE * 60)), RATE, CHEST_POLICY)
    assert codes, "band-passed noise has peaks at plausible spacing"


def test_the_questionable_cardiac_codes_are_recorded_not_fatal():
    """Measured: rejecting on them costs a third of the cardiac windows and
    moves the median RMSSD by three milliseconds or by nothing at all."""
    assert "unstable_intervals" in CHEST_POLICY.warn_only
    assert "beat_count_disagrees" in CHEST_POLICY.warn_only


# ── the tables line up ──────────────────────────────────────────────────────


def test_every_chest_signal_has_an_extractor_a_check_and_declared_features():
    assert set(CHEST_EXTRACTORS) == set(CHEST_CHECKS) == set(CHEST_FEATURES_BY_SIGNAL)


def test_the_chest_temperature_is_checked_under_the_name_wesad_gives_it():
    """WESAD calls it ``Temp`` on the chest and ``TEMP`` on the wrist. Keying
    this table by the wrist spelling leaves chest temperature unchecked, and an
    unchecked signal still produces features."""
    assert "Temp" in CHEST_CHECKS
    assert "TEMP" not in CHEST_CHECKS


def test_chest_features_carry_the_device_in_their_names():
    """A fused row holds a wrist and a chest electrodermal level; two columns
    called eda_mean would be one column."""
    for names in CHEST_FEATURES_BY_SIGNAL.values():
        for name in names:
            assert name.startswith("chest_"), name


def test_chest_accelerometry_is_not_divided_by_the_wrist_scale():
    """The E4 reports raw counts scaled by 64; the RespiBAN reports g."""
    assert CHEST_POLICY.acc_g_scale == 1.0


# ── the fused table ─────────────────────────────────────────────────────────

ARCHIVE = Path.home() / "Downloads" / "WESAD.zip"
needs_wesad = pytest.mark.skipif(
    not ARCHIVE.is_file(), reason="the WESAD archive is not present"
)


@pytest.fixture(scope="module")
def fused():
    """Two subjects, both devices, read once.

    Each build reads a 700 Hz chest recording end to end; doing it per test
    turned this file into four minutes.
    """
    return build(ARCHIVE, subjects=["S2", "S3"], device="both", stride_seconds=30.0)


@pytest.fixture(scope="module")
def wrist_only():
    return build(ARCHIVE, subjects=["S2", "S3"], device="wrist", stride_seconds=30.0)


def test_an_unknown_device_is_refused():
    with pytest.raises(ValueError, match="wrist, chest or both"):
        build("nowhere.zip", device="ankle")


@needs_wesad
def test_fusing_both_devices_produces_one_row_per_window(fused, wrist_only):
    """Not one row per device. The join is on the window interval."""
    assert len(fused) == pytest.approx(len(wrist_only), rel=0.1)
    assert len(fused.feature_names) > len(wrist_only.feature_names)
    assert set(wrist_only.feature_names) <= set(fused.feature_names)


@needs_wesad
def test_a_fused_row_carries_both_devices(fused):
    chest = [n for n in fused.feature_names if n.startswith("chest_")]
    wrist = [n for n in fused.feature_names if not n.startswith("chest_")]
    assert chest and wrist
    assert "chest_hr_mean" in chest, "the cardiac features are the point of the strap"
    assert "pulse_rate_mean" in wrist


@needs_wesad
def test_a_fused_table_records_both_feature_sets_and_both_policies(fused):
    """A model artifact pinned to one of them would accept a table it has
    never seen."""
    assert "+" in fused.feature_set_version
    assert "chest" in fused.feature_set_version
    assert "wrist-e4" in fused.qc_policy_version
    assert "chest-respiban" in fused.qc_policy_version


@needs_wesad
def test_the_two_devices_agree_on_heart_rate(fused):
    """An optical wrist sensor against electrodes, on the same minutes.

    They should not agree exactly -- that was measured at about 7 bpm, which is
    what a wrist gives -- but a correlation near zero would mean one of the two
    pipelines is not reading a heart.
    """
    ecg = fused.values[:, fused.feature_names.index("chest_hr_mean")]
    ppg = fused.values[:, fused.feature_names.index("pulse_rate_mean")]
    assert 40.0 < ecg.mean() < 100.0
    assert float(np.abs(ecg - ppg).mean()) < 15.0
    assert float(np.corrcoef(ecg, ppg)[0, 1]) > 0.2


@needs_wesad
def test_the_breathing_rate_is_not_a_constant(fused):
    """It was, twice: 6.00 for every window, then 13.00 for every window."""
    breathing = fused.values[:, fused.feature_names.index("chest_resp_rate_bpm")]
    assert breathing.std() > 1.0
    assert 6.0 <= breathing.min() < breathing.max() <= 36.0
