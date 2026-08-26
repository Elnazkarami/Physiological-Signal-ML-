"""Quality control on wearable signal.

WESAD was collected in a laboratory with researchers present, and it is clean:
nothing in it is rejected. That makes it useless for proving the checks work, so
the faults are injected here — a flatlined sensor, a clipped converter, an
electrode losing contact — and QC is asserted to find each one.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from physioml.core.recording import Modality, Recording
from physioml.core.window import QCStatus, SignalWindow
from physioml.io.wesad import WRIST_HZ
from physioml.peripheral.qc import (
    CLIPPED,
    DISCONTINUITY,
    FLATLINE,
    IMPLAUSIBLE_RATE,
    MISSING_AXES,
    MOTION,
    NEGATIVE,
    NO_PULSE,
    OUT_OF_RANGE,
    SATURATED,
    QCPolicy,
    apply_to_windows,
    assess,
    check_acc,
    check_bvp,
    check_eda,
    check_temp,
)
from physioml.peripheral.windowing import Epoch

POLICY = QCPolicy()
T0 = datetime(2017, 1, 1, tzinfo=UTC)


def pulse(seconds: float = 60.0, bpm: float = 70.0, rate: float = 64.0) -> np.ndarray:
    """A plausible photoplethysmogram: one clean peak per beat."""
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


# ── faults each check must find ─────────────────────────────────────────────


def test_a_flatlined_pulse_sensor_is_caught():
    """A sensor that is not attached still reports numbers."""
    assert FLATLINE in check_bvp(np.zeros((3840, 1)), 64.0, POLICY)


def test_too_few_beats_to_judge_is_caught():
    rng = np.random.default_rng(9)
    barely = np.concatenate([pulse(seconds=5.0), rng.normal(0, 0.001, (3520, 1))])
    assert NO_PULSE in check_bvp(barely, 64.0, POLICY)


def test_a_clipped_converter_is_caught():
    signal = pulse()
    signal[signal > 50] = 50.0  # a fifth of the samples pinned at the limit
    assert CLIPPED in check_bvp(signal, 64.0, POLICY)


@pytest.mark.parametrize("seed", range(6))
def test_noise_is_not_mistaken_for_a_pulse(seed):
    """Counting beats cannot catch this; periodicity can.

    Band-passed noise has energy in the pulse band and yields peaks at a
    perfectly plausible spacing, so a rate check passes it. What separates them
    is that a heartbeat concentrates its power at one frequency and noise
    spreads it — measured on WESAD, real pulse never falls below 0.33 and noise
    never reaches 0.21.
    """
    noise = np.random.default_rng(seed).normal(0, 1, (3840, 1))
    assert NO_PULSE in check_bvp(noise, 64.0, POLICY)


def test_a_heart_rate_physiology_does_not_produce_is_caught():
    assert IMPLAUSIBLE_RATE in check_bvp(pulse(bpm=240), 64.0, POLICY)


def test_negative_conductance_is_caught():
    """Skin conductance cannot be below zero; a negative value is the device."""
    assert NEGATIVE in check_eda(np.full((240, 1), -0.5), 4.0, POLICY)


def test_an_electrode_losing_contact_is_caught():
    signal = np.full((240, 1), 0.5)
    signal[120:] = 3.0  # a step no skin makes
    assert DISCONTINUITY in check_eda(signal, 4.0, POLICY)


def test_conductance_outside_the_plausible_range_is_caught():
    assert OUT_OF_RANGE in check_eda(np.full((240, 1), 200.0), 4.0, POLICY)


def test_a_temperature_no_wrist_reaches_is_caught():
    assert OUT_OF_RANGE in check_temp(np.full((240, 1), 12.0), 4.0, POLICY)


def test_a_sensor_leaving_the_wrist_is_caught():
    """Skin does not cool three degrees in a minute; a strap coming off does."""
    signal = np.linspace(34.0, 31.0, 240).reshape(-1, 1)
    assert DISCONTINUITY in check_temp(signal, 4.0, POLICY)


def test_a_saturated_accelerometer_is_caught():
    array = np.full((1920, 3), 20.0)
    array[:100] = 128.0  # at the eight-bit rail: the true value is unknown
    assert SATURATED in check_acc(array, 32.0, POLICY)


def test_accelerometry_missing_an_axis_is_caught():
    assert MISSING_AXES in check_acc(np.zeros((1920, 2)), 32.0, POLICY)


# ── clean signal raises nothing ─────────────────────────────────────────────


def test_plausible_signal_passes_every_check():
    rng = np.random.default_rng(3)
    assert check_bvp(pulse(), 64.0, POLICY) == []
    assert check_eda(0.4 + rng.normal(0, 0.01, (240, 1)), 4.0, POLICY) == []
    assert check_temp(np.linspace(34.0, 34.2, 240).reshape(-1, 1), 4.0, POLICY) == []
    assert check_acc(20.0 + rng.normal(0, 0.3, (1920, 3)), 32.0, POLICY) == []


def test_a_perfectly_constant_signal_is_a_flatline_not_a_calm_one():
    """A sensor reporting the identical value 240 times is not measuring."""
    assert FLATLINE in check_eda(np.full((240, 1), 0.4), 4.0, POLICY)
    assert FLATLINE in check_acc(np.full((1920, 3), 20.0), 32.0, POLICY)


# ── how verdicts combine ────────────────────────────────────────────────────


def steady_acc(n: int = 1920) -> np.ndarray:
    """Accelerometry from someone sitting still: gravity, and almost nothing else."""
    rng = np.random.default_rng(1)
    return np.full((n, 3), 20.0) + rng.normal(0, 0.3, (n, 3))


def moving_acc(n: int = 1920) -> np.ndarray:
    rng = np.random.default_rng(2)
    return np.full((n, 3), 20.0) + rng.normal(0, 12.0, (n, 3))


def test_motion_warns_rather_than_rejects():
    """It makes the pulse unreliable; it does not make the interval wrong."""
    result = assess(epoch_of(BVP=pulse(), ACC=moving_acc()))
    assert result.statuses["BVP"] is QCStatus.WARNING
    assert MOTION in result.codes["BVP"]
    assert "BVP" in result.usable, "a warned window is still usable"


def test_motion_is_decided_from_accelerometry_not_from_the_pulse():
    """Checking a pulse against itself could never find that someone moved."""
    still = assess(epoch_of(BVP=pulse(), ACC=steady_acc()))
    assert still.statuses["BVP"] is QCStatus.VALID
    assert still.statuses["ACC"] is QCStatus.VALID

    moved = assess(epoch_of(BVP=pulse(), ACC=moving_acc()))
    assert moved.statuses["ACC"] is QCStatus.VALID, "the accelerometry itself is fine"
    assert moved.statuses["BVP"] is QCStatus.WARNING


def test_a_fatal_code_rejects_while_a_warning_does_not():
    result = assess(epoch_of(BVP=np.zeros((3840, 1)), ACC=steady_acc()))
    assert result.statuses["BVP"] is QCStatus.REJECTED
    assert result.rejected == {"BVP"}
    assert result.usable == {"ACC"}


def test_the_policy_version_travels_with_the_verdict():
    """A result obtained under one policy is not comparable to another's."""
    result = assess(epoch_of(BVP=pulse(), ACC=steady_acc()), QCPolicy(version="strict-9"))
    assert result.policy_version == "strict-9"


def test_rejection_labels_the_window_without_discarding_it():
    epoch = epoch_of(BVP=np.zeros((3840, 1)), ACC=steady_acc())
    marked = apply_to_windows(epoch, assess(epoch))

    assert set(marked) == {"BVP", "ACC"}, "nothing is dropped"
    assert marked["BVP"].qc_status is QCStatus.REJECTED
    assert FLATLINE in marked["BVP"].qc_reason_codes
    assert marked["BVP"].window_id == epoch.windows["BVP"].window_id


# ── against the real dataset ────────────────────────────────────────────────


ARCHIVE = Path.home() / "Downloads" / "WESAD.zip"


@pytest.mark.skipif(not ARCHIVE.is_file(), reason="the WESAD archive is not present")
def test_lab_collected_signal_is_not_rejected():
    """WESAD is clean, and QC saying so is the correct outcome.

    It flags a few percent of pulse windows for motion and rejects nothing. The
    checks are proved by the injected faults above; this asserts they do not
    fire on good signal, which is the other half of being useful.
    """
    from physioml.io.wesad import WESAD
    from physioml.peripheral.windowing import epochs

    data = WESAD(ARCHIVE).read("S2")
    labelled = [e for e in epochs(data) if e.labelled]
    results = [assess(e) for e in labelled[:200]]

    assert all(not r.rejected for r in results), "nothing in WESAD should be rejected"
    assert any(MOTION in r.codes.get("BVP", ()) for r in results), (
        "some windows should be flagged for movement"
    )
