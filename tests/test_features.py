"""Peripheral feature extraction.

Values are asserted against signals whose answer is known by construction — a
synthetic pulse at a stated rate must come back at that rate — rather than
against numbers recorded from a previous run of this same code, which would only
assert that it has not changed.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from physioml.core.recording import Modality, Recording
from physioml.core.window import QCStatus, SignalWindow
from physioml.io.wesad import WRIST_HZ
from physioml.peripheral.features import (
    FEATURE_SET_VERSION,
    acc_features,
    bvp_features,
    eda_features,
    extract,
    temp_features,
)
from physioml.peripheral.preprocessing import bandpass, detect_beats, dominant_rate_bpm
from physioml.peripheral.qc import assess
from physioml.peripheral.windowing import Epoch

T0 = datetime(2017, 1, 1, tzinfo=UTC)


def pulse(bpm: float = 72.0, seconds: float = 60.0, rate: float = 64.0) -> np.ndarray:
    t = np.arange(0, seconds, 1 / rate)
    return (100 * np.sin(2 * np.pi * (bpm / 60.0) * t)).reshape(-1, 1)


def epoch_of(**signals: np.ndarray) -> Epoch:
    windows, samples = {}, {}
    for name, array in signals.items():
        rate = WRIST_HZ[name]
        recording = Recording.create(
            study_id="T",
            subject_id="S00",
            modality=Modality.ACC if name == "ACC" else Modality[name],
            sampling_rate_hz=rate,
            start_time=T0,
            duration_seconds=len(array) / rate,
            channels=("x", "y", "z") if name == "ACC" else (name,),
        )
        windows[name] = SignalWindow.create(
            recording_id=recording.recording_id,
            subject_id="S00",
            start_sample=0,
            end_sample=len(array),
            start_time=T0,
            sampling_rate_hz=rate,
        )
        samples[name] = array
    return Epoch("S00", 0, 0.0, 60.0, "baseline", windows, samples)


# ── signal processing, against known answers ────────────────────────────────


@pytest.mark.parametrize("bpm", [48.0, 72.0, 110.0, 150.0])
def test_a_pulse_at_a_known_rate_is_recovered(bpm):
    found = bvp_features(pulse(bpm), 64.0)["pulse_rate_mean"]
    assert abs(found - bpm) < 2.0, f"{bpm} bpm came back as {found:.1f}"


@pytest.mark.parametrize("bpm", [50.0, 90.0, 140.0])
def test_the_spectral_estimate_finds_the_same_rate(bpm):
    assert abs(dominant_rate_bpm(pulse(bpm).ravel(), 64.0) - bpm) < 3.0


def test_beat_count_matches_the_rate():
    beats = detect_beats(pulse(60.0).ravel(), 64.0)
    assert 58 <= beats.size <= 62, "a minute at 60 bpm is about sixty beats"


def test_the_band_pass_removes_a_baseline_it_should_not_keep():
    t = np.arange(0, 60, 1 / 64)
    drift = 500 * np.sin(2 * np.pi * 0.02 * t)  # 0.02 Hz, well below the band
    filtered = bandpass(pulse().ravel() + drift, 64.0, 0.5, 8.0)
    assert np.std(filtered) < np.std(pulse().ravel() + drift)
    assert abs(dominant_rate_bpm(filtered, 64.0) - 72.0) < 3.0


def test_a_band_the_sampling_rate_cannot_support_is_refused():
    with pytest.raises(ValueError, match="not usable"):
        bandpass(np.zeros(1000), 8.0, 10.0, 20.0)


# ── the decision not to emit pulse variability ──────────────────────────────


def test_pulse_variability_is_not_emitted():
    """Validated against WESAD's chest ECG and wrong by a factor of 3.6.

    At 64 Hz one sample is 15.6 ms, a large fraction of the 20-60 ms these
    measures resolve, and every missed or doubled beat enters squared. They
    would have raised a model's score while being indefensible, which is worse
    than their absence. See the note in bvp_features.
    """
    emitted = set(bvp_features(pulse(), 64.0))
    assert not emitted & {"ppi_sdnn", "ppi_rmssd", "ppi_pnn50"}
    assert "pulse_rate_mean" in emitted, "rate is validated and kept"


def test_the_feature_set_version_records_that_removal():
    assert FEATURE_SET_VERSION == "1.3"


def test_changing_a_filter_changes_the_preprocessing_identity():
    """Windows produced under different settings must not share an identifier."""
    from physioml.peripheral.preprocessing import Preprocessing

    assert Preprocessing().run_id == Preprocessing().run_id
    assert Preprocessing().run_id != Preprocessing(bvp_high_hz=6.0).run_id


# ── the other modalities ────────────────────────────────────────────────────


def test_electrodermal_level_and_direction():
    rising = np.linspace(1.0, 2.0, 240).reshape(-1, 1)
    found = eda_features(rising, 4.0)
    assert found["eda_mean"] == pytest.approx(1.5, abs=0.01)
    assert found["eda_slope_per_min"] == pytest.approx(1.0, abs=0.01)
    assert found["eda_min"] < found["eda_max"]


def test_temperature_slope_is_per_minute():
    """A degree over a minute is a degree per minute."""
    falling = np.linspace(35.0, 34.0, 240).reshape(-1, 1)
    assert temp_features(falling, 4.0)["temp_slope_per_min"] == pytest.approx(
        -1.0, abs=0.01
    )


def test_accelerometry_reports_gravity_when_still():
    """A stationary wrist reads about 1 g, which is a useful sanity check."""
    still = np.zeros((1920, 3))
    still[:, 2] = 64.0  # one g on one axis, in E4 units
    found = acc_features(still, 32.0)
    assert found["acc_magnitude_mean"] == pytest.approx(1.0, abs=0.01)
    assert found["acc_magnitude_sd"] == pytest.approx(0.0, abs=1e-9)


def test_movement_raises_the_activity_measures():
    rng = np.random.default_rng(4)
    still = np.full((1920, 3), 20.0) + rng.normal(0, 0.3, (1920, 3))
    moving = np.full((1920, 3), 20.0) + rng.normal(0, 12.0, (1920, 3))
    assert acc_features(moving, 32.0)["acc_activity_count"] > (
        acc_features(still, 32.0)["acc_activity_count"] * 5
    )


# ── extraction honours quality control ──────────────────────────────────────


def test_a_rejected_signal_contributes_no_features():
    """A rate from a flatlined sensor is not missing, it is confidently wrong."""
    epoch = epoch_of(BVP=np.zeros((3840, 1)), TEMP=np.full((240, 1), 34.0))
    qc = assess(epoch)
    assert qc.statuses["BVP"] is QCStatus.REJECTED

    names = {f.name for f in extract(epoch, qc)}
    assert not any(n.startswith(("pulse_", "ppi_")) for n in names)


def test_a_warned_signal_still_contributes():
    """Motion makes the pulse less reliable; dropping it would lose the
    accelerometry that says so."""
    rng = np.random.default_rng(5)
    moving = np.full((1920, 3), 20.0) + rng.normal(0, 12.0, (1920, 3))
    epoch = epoch_of(BVP=pulse(), ACC=moving)
    qc = assess(epoch)
    assert qc.statuses["BVP"] is QCStatus.WARNING
    assert any(f.name == "pulse_rate_mean" for f in extract(epoch, qc))


def test_every_feature_carries_its_window_and_version():
    epoch = epoch_of(BVP=pulse(), TEMP=np.linspace(34, 34.2, 240).reshape(-1, 1))
    for feature in extract(epoch, assess(epoch)):
        assert feature.feature_set_version == FEATURE_SET_VERSION
        assert feature.source_window_ids
        assert feature.subject_id == "S00"
