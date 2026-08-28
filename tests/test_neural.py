"""Sleep electroencephalography features and quality control.

Every spectral claim is asserted against a signal whose spectrum is known by
construction -- a sine at 10 Hz must put its power in the alpha band and
nowhere else -- rather than against numbers recorded from a previous run.
"""

from __future__ import annotations

import numpy as np
import pytest

from physioml.neural.features import (
    BANDS,
    FEATURES_BY_CHANNEL,
    band_power,
    eeg_features,
    emg_features,
    eog_features,
    hjorth,
    spectral_edge,
    spectral_entropy,
    spectrum,
)
from physioml.neural.qc import (
    CLIPPED,
    DEFAULT_EEG_POLICY,
    FLATLINE,
    HIGH_AMPLITUDE,
    MUSCLE,
    check_eeg,
    check_emg,
)

RATE = 100.0
EPOCH = 30.0


def wave(hz: float, amplitude: float = 20.0, seconds: float = EPOCH) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return amplitude * np.sin(2 * np.pi * hz * t)


def noise(sd: float = 20.0, seconds: float = EPOCH, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, sd, int(seconds * RATE))


# ── band power ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("hz", "band"),
    [(2.0, "delta"), (6.0, "theta"), (10.0, "alpha"), (14.0, "sigma"), (22.0, "beta")],
)
def test_a_sine_puts_its_power_in_the_band_it_belongs_to(hz, band):
    found = eeg_features(wave(hz), RATE, "fpz")
    shares = {name: found[f"fpz_{name}_rel"] for name in BANDS}
    assert max(shares, key=shares.get) == band
    assert shares[band] > 0.8


def test_relative_powers_account_for_the_whole_band():
    found = eeg_features(noise(), RATE, "fpz")
    total = sum(found[f"fpz_{name}_rel"] for name in BANDS)
    assert total == pytest.approx(1.0, abs=0.02)


def test_relative_power_is_unchanged_by_the_amplifier_gain():
    """Absolute microvolts vary several-fold between people for reasons that
    have nothing to do with sleep; the relative figures are what should
    transfer."""
    quiet = eeg_features(wave(10.0, amplitude=5.0), RATE, "fpz")
    loud = eeg_features(wave(10.0, amplitude=50.0), RATE, "fpz")
    assert quiet["fpz_alpha_rel"] == pytest.approx(loud["fpz_alpha_rel"], abs=0.01)
    assert loud["fpz_alpha"] > quiet["fpz_alpha"] * 50


def test_band_power_does_not_depend_on_the_frequency_resolution():
    """Summing bins instead of integrating would make a 30-second epoch and a
    20-second one disagree about the same signal."""
    long_f, long_p = spectrum(wave(10.0, seconds=30.0), RATE)
    short_f, short_p = spectrum(wave(10.0, seconds=20.0), RATE)
    band = BANDS["alpha"]
    assert band_power(long_f, long_p, band) == pytest.approx(
        band_power(short_f, short_p, band), rel=0.15
    )


def test_a_band_with_no_room_in_it_is_zero_not_an_error():
    frequencies = np.array([0.0, 50.0])
    assert band_power(frequencies, np.array([1.0, 1.0]), (10.0, 11.0)) == 0.0


# ── spectral shape ──────────────────────────────────────────────────────────


def test_entropy_separates_a_rhythm_from_noise():
    """Deep sleep concentrates its power; wake and REM spread it."""
    _, rhythmic = spectrum(wave(2.0), RATE)
    _, broadband = spectrum(noise(), RATE)
    assert spectral_entropy(rhythmic) < 0.5
    assert spectral_entropy(broadband) > 0.8


def test_the_spectral_edge_sits_above_the_rhythm_and_below_the_ceiling():
    frequencies, power = spectrum(wave(10.0), RATE)
    assert 9.0 < spectral_edge(frequencies, power) < 15.0


def test_hjorth_mobility_is_the_frequency_of_a_sine():
    """Known in closed form, and the closed form is the discrete one.

    Mobility is the standard deviation of the first difference over that of the
    signal. For a sampled sine that is exactly 2*sin(pi*f/rate), not the
    2*pi*f/rate the derivative of a continuous sine would give -- the two agree
    only while f is small against the sampling rate, and at 20 Hz in 100 they
    differ by 7%.
    """
    for hz in (2.0, 10.0, 20.0):
        _, mobility, _ = hjorth(wave(hz))
        assert mobility == pytest.approx(2 * np.sin(np.pi * hz / RATE), rel=0.01)


def test_hjorth_complexity_is_lowest_for_a_pure_sine():
    """It measures departure from a single frequency, so a sine is its floor."""
    _, _, pure = hjorth(wave(10.0))
    _, _, mixed = hjorth(wave(10.0) + wave(2.0) + noise(5.0))
    assert pure == pytest.approx(1.0, abs=0.05)
    assert mixed > pure


def test_a_flat_signal_produces_no_infinities():
    assert hjorth(np.zeros(1000)) == (0.0, 0.0, 0.0)
    assert spectral_entropy(np.zeros(100)) == 0.0
    assert spectral_edge(np.array([1.0, 2.0]), np.zeros(2)) == 0.0


# ── the supporting channels ─────────────────────────────────────────────────


def test_eye_movement_shows_up_as_slow_power():
    """Rapid eye movements are large slow deflections, not fast ones."""
    rapid = eog_features(wave(0.8, amplitude=100.0), RATE)
    still = eog_features(wave(8.0, amplitude=100.0), RATE)
    assert rapid["eog_slow_rel"] > 0.8
    assert still["eog_slow_rel"] < 0.2


def test_chin_tone_ignores_a_constant_offset():
    """An electrode offset is not muscle tone."""
    signal = noise(3.0)
    assert emg_features(signal, RATE)["chin_emg_rms"] == pytest.approx(
        emg_features(signal + 50.0, RATE)["chin_emg_rms"]
    )


def test_chin_tone_falls_with_atonia():
    tense = emg_features(noise(10.0), RATE)
    atonic = emg_features(noise(0.5), RATE)
    assert atonic["chin_emg_rms"] < tense["chin_emg_rms"] / 10


# ── quality control ─────────────────────────────────────────────────────────


def test_a_disconnected_electrode_is_refused():
    assert check_eeg(np.zeros(3000), RATE, DEFAULT_EEG_POLICY) == [FLATLINE]


def test_ordinary_sleep_electroencephalography_passes():
    assert check_eeg(wave(2.0) + noise(5.0), RATE, DEFAULT_EEG_POLICY) == []


def test_a_saturated_amplifier_is_caught():
    clipped = np.clip(noise(200.0), -150.0, 150.0)
    assert CLIPPED in check_eeg(clipped, RATE, DEFAULT_EEG_POLICY)


def test_an_epoch_of_hundreds_of_microvolts_is_an_artifact():
    """A scalp recording is tens of microvolts; slow-wave sleep reaches perhaps
    two hundred peak to peak."""
    assert HIGH_AMPLITUDE in check_eeg(wave(1.0, amplitude=800.0), RATE, DEFAULT_EEG_POLICY)


def test_an_epoch_reading_muscle_is_marked_but_not_rejected():
    """Wake epochs legitimately carry muscle activity -- that is part of what
    makes them wake -- so rejecting on it would delete the class it describes."""
    jaw = wave(38.0, amplitude=40.0) + noise(2.0)
    assert MUSCLE in check_eeg(jaw, RATE, DEFAULT_EEG_POLICY)
    assert MUSCLE in DEFAULT_EEG_POLICY.warn_only


def test_atonia_is_a_finding_not_a_fault():
    """Chin electromyography in REM is genuinely almost flat."""
    assert check_emg(noise(0.05), RATE, DEFAULT_EEG_POLICY) == []
    assert check_emg(np.zeros(3000), RATE, DEFAULT_EEG_POLICY) == [FLATLINE]


# ── the channel map ─────────────────────────────────────────────────────────


def test_every_channel_declares_the_features_it_produces():
    assert set(FEATURES_BY_CHANNEL) == {
        "EEG Fpz-Cz",
        "EEG Pz-Oz",
        "EOG horizontal",
        "EMG submental",
    }
    for label, names in FEATURES_BY_CHANNEL.items():
        assert names, label


def test_no_feature_is_claimed_by_two_channels():
    seen: dict[str, str] = {}
    for channel, names in FEATURES_BY_CHANNEL.items():
        for name in names:
            assert name not in seen, f"{name} claimed by {seen.get(name)} and {channel}"
            seen[name] = channel


def test_the_two_electroencephalogram_channels_produce_distinct_names():
    """A table holds both derivations in one row; two columns called
    delta would be one column."""
    fpz = set(FEATURES_BY_CHANNEL["EEG Fpz-Cz"])
    pz = set(FEATURES_BY_CHANNEL["EEG Pz-Oz"])
    assert not fpz & pz
    assert len(fpz) == len(pz)
