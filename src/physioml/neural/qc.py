"""Quality control for scalp electrophysiology.

A night of sleep recording contains stretches where the electrode has come
loose, where the participant has moved, and where the amplifier has saturated.
None of them are stages, and all of them produce features.

The same discipline as the peripheral policy: this labels epochs, it does not
delete them, and the codes are carried so a model can be told why a window is
missing rather than finding a hole. What differs is the physiology. A scalp
electroencephalogram is tens of microvolts, so an epoch reaching hundreds is
not an unusually loud brain; and an epoch whose power is mostly above 30 Hz is
reading the participant's jaw, not their cortex.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from physioml.neural.features import band_power, spectrum

FLATLINE = "flatline"
CLIPPED = "clipped"
HIGH_AMPLITUDE = "high_amplitude"
MUSCLE = "muscle_dominant"


@dataclass(frozen=True, slots=True)
class EEGPolicy:
    """Thresholds for judging an epoch of scalp electrophysiology."""

    version: str = "sleep-eeg-1.0"

    flat_sd_uv: float = 0.1
    """Below this the electrode is not connected to anything. A real scalp
    signal at rest is a few microvolts of standard deviation even in the
    quietest stage."""

    clip_fraction: float = 0.02
    """Share of samples sitting at the extreme value before the amplifier is
    judged to have saturated."""

    max_amplitude_uv: float = 500.0
    """A scalp electroencephalogram is tens of microvolts. Slow-wave sleep
    reaches perhaps 200 peak to peak. Five hundred is an artifact -- a movement,
    a swallow, an electrode being touched."""

    max_muscle_share: float = 0.5
    """Share of 0.5-45 Hz power sitting above 30 Hz before the epoch is judged
    to be reading muscle. Set from what the band means rather than from the
    data: half the power above the electroencephalogram's own range is not a
    brain rhythm."""

    warn_only: frozenset[str] = field(default_factory=lambda: frozenset({MUSCLE}))
    """Muscle contamination marks an epoch questionable rather than unusable.
    Wake epochs legitimately carry muscle activity -- that is part of what makes
    them wake -- so rejecting on it would delete the class it describes."""


DEFAULT_EEG_POLICY = EEGPolicy()


def check_eeg(samples: np.ndarray, rate: float, policy: EEGPolicy) -> list[str]:
    data = np.asarray(samples, dtype=float).ravel()
    if data.size < int(rate):
        return ["too_short"]

    codes: list[str] = []
    if float(np.std(data)) < policy.flat_sd_uv:
        return [FLATLINE]

    extreme = float(np.max(np.abs(data)))
    if (
        extreme > 0
        and float(np.mean(np.abs(data) >= extreme * 0.999)) > policy.clip_fraction
    ):
        codes.append(CLIPPED)
    if extreme > policy.max_amplitude_uv:
        codes.append(HIGH_AMPLITUDE)

    frequencies, power = spectrum(data, rate)
    if frequencies.size:
        whole = band_power(frequencies, power, (0.5, 45.0))
        muscle = band_power(frequencies, power, (30.0, 45.0))
        if whole > 0 and muscle / whole > policy.max_muscle_share:
            codes.append(MUSCLE)
    return codes


def check_eog(samples: np.ndarray, rate: float, policy: EEGPolicy) -> list[str]:
    data = np.asarray(samples, dtype=float).ravel()
    if data.size < int(rate):
        return ["too_short"]
    if float(np.std(data)) < policy.flat_sd_uv:
        return [FLATLINE]
    return []


def check_emg(samples: np.ndarray, rate: float, policy: EEGPolicy) -> list[str]:
    data = np.asarray(samples, dtype=float).ravel()
    if data.size < 4:
        return ["too_short"]
    if float(np.std(data)) < policy.flat_sd_uv / 10.0:
        # Chin electromyography in REM is genuinely almost flat, so the
        # threshold here is far below the electroencephalogram's: atonia is a
        # finding, not a fault.
        return [FLATLINE]
    return []


EEGCheck = Callable[[np.ndarray, float, EEGPolicy], list[str]]

CHECKS: dict[str, EEGCheck] = {
    "EEG Fpz-Cz": check_eeg,
    "EEG Pz-Oz": check_eeg,
    "EOG horizontal": check_eog,
    "EMG submental": check_emg,
}
