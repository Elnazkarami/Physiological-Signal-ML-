"""Deciding whether a window of wearable signal is worth computing on.

Wearable physiology fails in ways laboratory physiology does not: the strap
loosens, the optical sensor lifts off the skin, the participant moves, the
electrode dries out. None of those announce themselves. They produce numbers,
and the numbers look like physiology until you check.

So this module exists between windowing and feature extraction, and it has one
rule: **it labels, it does not filter.** Every window gets a status and, when
something is wrong, codes saying what. A pipeline that dropped bad windows here
would report a clean dataset and a model trained on an unstated subset, and no
downstream number would reveal it. Keeping them means "how much of this subject
survived QC" is a question with an answer -- usually the first one worth asking
about a disappointing result.

**Thresholds are declared, not embedded.** They are judgements about a
particular device worn by particular people, not facts, and changing one changes
which windows are usable and therefore what the model learns. :class:`QCPolicy`
carries a version for that reason: a result obtained under one policy is not
comparable to a result obtained under another, and the version is what makes the
difference visible rather than a footnote nobody wrote down.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from physioml.core.window import QCStatus
from physioml.peripheral.preprocessing import (
    DEFAULT_PREPROCESSING,
    Preprocessing,
    detect_beats,
    prepare_bvp,
    pulse_concentration,
)

if TYPE_CHECKING:
    from physioml.peripheral.windowing import Epoch

# ── reason codes ─────────────────────────────────────────────────────────────
# Codes rather than prose so a cohort can be counted by failure mode.

FLATLINE = "flatline"
"""The signal does not vary. A sensor that is not attached still reports."""

CLIPPED = "clipped"
"""Samples sit at the converter's limit; the true value is unknown and larger."""

OUT_OF_RANGE = "out_of_range"
"""Values physiology does not produce -- a skin temperature of 12 °C."""

NEGATIVE = "negative"
"""Negative conductance. Electrodermal activity cannot be below zero."""

DISCONTINUITY = "discontinuity"
"""A step no physiological process makes; the sensor moved or dropped out."""

MOTION = "motion"
"""Accelerometry says the participant moved enough to corrupt an optical pulse."""

NO_PULSE = "no_pulse"
"""No plausible pulse could be found in a photoplethysmogram."""

IMPLAUSIBLE_RATE = "implausible_rate"
"""A pulse rate outside what a resting-to-mildly-stressed adult produces."""

SATURATED = "saturated"
"""Accelerometer magnitude beyond the range the device can represent."""

MISSING_AXES = "missing_axes"

MISSING = "missing"
"""Samples that are not numbers.

Every comparison against NaN is false, so a window of them passes each
threshold below in turn and is marked valid. It then produces NaN features,
which are dropped as non-finite during extraction, and the row loses that
signal with nothing on record saying why. Missing data has to be looked for
directly rather than caught by a range check."""


def _missing(signal: np.ndarray, allowed: float = 0.0) -> bool:
    """Whether a signal has more non-finite samples than can be tolerated."""
    if signal.size == 0:
        return True
    return float(np.mean(~np.isfinite(signal))) > allowed


"""Fewer than three axes of accelerometry."""


@dataclass(frozen=True, slots=True)
class QCPolicy:
    """Thresholds for judging wearable signal, and the version they form.

    Defaults suit the Empatica E4 worn on the wrist by healthy adults sitting or
    standing. They are a starting point that has not been tuned against
    anything, which is stated here rather than discovered later.
    """

    version: str = "wrist-e4-1.0"

    # BVP — a photoplethysmogram, in device units rather than a physical scale.
    bvp_flat_sd: float = 1e-6
    bvp_clip_fraction: float = 0.02
    """Reject when this share of samples sits at the observed extreme."""

    bvp_min_bpm: float = 35.0
    bvp_max_bpm: float = 180.0
    bvp_min_beats: int = 20
    """Fewer detectable beats than this in a window means no usable pulse."""

    bvp_min_concentration: float = 0.25
    """Least share of in-band power a real pulse concentrates at its own
    frequency and the harmonic above it.

    Sits in the gap between filtered white noise, which reaches 0.237 across 30
    draws, and the weakest of 300 real WESAD windows, at 0.327. Both measured
    exactly as this check calls it -- an earlier version of this threshold was
    calibrated over a different band from the one it was then applied in, and
    the separation it appeared to have did not exist under the conditions that
    mattered."""

    bvp_detect_ceiling_bpm: float = 300.0
    """Peaks are sought up to this rate, well above anything physiological, so
    that a rate above ``bvp_max_bpm`` is *detected and rejected* rather than
    quietly halved by the spacing constraint."""

    # A beat-interval regularity check lived here and was removed. It assumed
    # real intervals vary by a few percent; measured against WESAD they vary by
    # 30% at the median, because a raw photoplethysmogram carries a dicrotic
    # notch that a naive peak finder counts as a beat. At any threshold tight
    # enough to catch noise it rejected a third of good signal. The honest fix
    # is to band-pass the pulse before detecting peaks -- that is preprocessing,
    # it is not built yet, and QC should not quietly do it. Noise is still
    # caught, by the rate check: peaks found in noise imply hundreds of beats a
    # minute.

    # EDA — microsiemens.
    eda_flat_sd: float = 1e-4
    eda_min_us: float = 0.01
    eda_max_us: float = 60.0
    eda_max_jump_us: float = 1.0
    """A step larger than this between consecutive samples is not skin.

    Skin conductance rises over hundreds of milliseconds; a jump of a whole
    microsiemens between two samples is the electrode losing or regaining
    contact. Set at a physiological limit rather than at a percentile of any
    particular recording -- on WESAD the largest observed step is 0.78 µS, so
    this does not fire, which is the correct outcome for clean data."""

    # Skin temperature — Celsius.
    temp_min_c: float = 25.0
    temp_max_c: float = 40.0
    temp_max_slope_c_per_min: float = 2.0
    """Skin does not change temperature this fast. A drop at this rate is the
    sensor leaving the wrist, not the wrist cooling."""

    # Accelerometry — device units, 1g = 64 on the E4.
    acc_g_scale: float = 64.0
    acc_rail: float = 128.0
    """The converter's limit. The E4 reports eight-bit axes, so a sample at
    ±128 means the true acceleration was larger and is unknown -- which is what
    saturation actually looks like, rather than a magnitude in g."""

    acc_rail_fraction: float = 0.01
    acc_motion_g: float = 0.08
    """Standard deviation of magnitude above which an optical pulse is suspect.

    Calibrated against WESAD, where a seated protocol puts the 95th percentile
    at 0.078 g and the maximum at 0.135. It marks the most active few percent of
    windows as questionable; it does not reject them, because motion invalidates
    a pulse feature while leaving the accelerometry features it is derived from
    perfectly good. A free-living deployment would need this re-set, and that is
    what the policy version is for."""

    warn_only: frozenset[str] = field(
        default_factory=lambda: frozenset({MOTION, DISCONTINUITY})
    )
    """Codes that mark a window questionable rather than unusable.

    Motion does not make a window wrong, it makes the pulse features unreliable
    while leaving accelerometry features perfectly valid -- so it is recorded
    and left to the model, not used to discard the interval."""


DEFAULT_POLICY = QCPolicy()


# ── per-modality checks ──────────────────────────────────────────────────────


def check_bvp(
    samples: np.ndarray,
    rate: float,
    policy: QCPolicy,
    preprocessing: Preprocessing = DEFAULT_PREPROCESSING,
) -> list[str]:
    """Blood-volume pulse: is there a usable pulse in here?

    Clipping and flatline are judged on the raw signal, because they are
    properties of what the converter recorded and a filter would hide both.
    Everything after that is judged on the *filtered* signal, because that is
    what the features are computed from — quality control approving a signal
    the extractor never sees would be checking the wrong thing.
    """
    raw = np.asarray(samples, dtype=float).ravel()
    codes: list[str] = []
    if _missing(raw):
        return [MISSING]
    if raw.size == 0:
        return [FLATLINE]

    if float(np.std(raw)) <= policy.bvp_flat_sd:
        return [FLATLINE]

    signal = raw

    # Each rail is counted separately: a converter pinned at its positive limit
    # while the signal still swings well past that negatively is clipped, and
    # comparing against the absolute extreme would never see it.
    for rail in (float(signal.max()), float(signal.min())):
        if np.isclose(signal, rail, rtol=1e-9).mean() > policy.bvp_clip_fraction:
            codes.append(CLIPPED)
            break

    # From here on, the signal as the feature extractor will see it.
    try:
        signal = prepare_bvp(raw, rate, preprocessing)
    except ValueError:
        signal = raw

    # Periodicity before rate. Counting beats cannot separate a pulse from
    # filtered noise, because noise produces peaks at a plausible spacing too.
    if (
        pulse_concentration(
            signal,
            rate,
            min_bpm=policy.bvp_min_bpm,
            max_bpm=policy.bvp_detect_ceiling_bpm,
        )
        < policy.bvp_min_concentration
    ):
        codes.append(NO_PULSE)
        return codes

    beats = _pulse_peaks(signal, rate, policy)
    if beats.size < policy.bvp_min_beats:
        codes.append(NO_PULSE)
        return codes

    intervals = np.diff(beats) / rate
    bpm = 60.0 / float(np.median(intervals))
    if not policy.bvp_min_bpm <= bpm <= policy.bvp_max_bpm:
        codes.append(IMPLAUSIBLE_RATE)
    return codes


def check_eda(
    samples: np.ndarray,
    rate: float,
    policy: QCPolicy,
    preprocessing: Preprocessing = DEFAULT_PREPROCESSING,
) -> list[str]:
    """Electrodermal activity: is the electrode in contact with skin?"""
    signal = np.asarray(samples, dtype=float).ravel()
    codes: list[str] = []
    if signal.size == 0:
        return [FLATLINE]
    if _missing(signal):
        return [MISSING]

    if float(signal.min()) < 0:
        codes.append(NEGATIVE)
    if float(np.std(signal)) <= policy.eda_flat_sd:
        codes.append(FLATLINE)
    if float(signal.min()) < policy.eda_min_us or float(signal.max()) > policy.eda_max_us:
        codes.append(OUT_OF_RANGE)
    if signal.size > 1 and float(np.abs(np.diff(signal)).max()) > policy.eda_max_jump_us:
        codes.append(DISCONTINUITY)
    return codes


def check_temp(
    samples: np.ndarray,
    rate: float,
    policy: QCPolicy,
    preprocessing: Preprocessing = DEFAULT_PREPROCESSING,
) -> list[str]:
    """Skin temperature: is the sensor against a person?"""
    signal = np.asarray(samples, dtype=float).ravel()
    codes: list[str] = []
    if _missing(signal):
        return [MISSING]
    if signal.size == 0:
        return [FLATLINE]

    if float(signal.min()) < policy.temp_min_c or float(signal.max()) > policy.temp_max_c:
        codes.append(OUT_OF_RANGE)
    if signal.size > 1:
        minutes = signal.size / rate / 60.0
        slope = abs(float(signal[-1] - signal[0])) / max(minutes, 1e-9)
        if slope > policy.temp_max_slope_c_per_min:
            codes.append(DISCONTINUITY)
    return codes


def check_acc(
    samples: np.ndarray,
    rate: float,
    policy: QCPolicy,
    preprocessing: Preprocessing = DEFAULT_PREPROCESSING,
) -> list[str]:
    """Accelerometry: three axes, and within what the device can represent."""
    array = np.asarray(samples, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        return [MISSING_AXES]
    if array.size == 0:
        return [FLATLINE]
    if _missing(array):
        return [MISSING]

    codes: list[str] = []
    at_rail = np.isclose(np.abs(array), policy.acc_rail, atol=0.5)
    if at_rail.mean() > policy.acc_rail_fraction:
        codes.append(SATURATED)
    magnitude = np.linalg.norm(array, axis=1) / policy.acc_g_scale
    if float(np.std(magnitude)) <= 1e-9:
        codes.append(FLATLINE)
    return codes


Check = Callable[[np.ndarray, float, "QCPolicy", "Preprocessing"], list[str]]
"""Quality control judges the *filtered* pulse, so it needs the same settings
the features are computed under. Reaching for the module default here would
let a window be approved on one signal and measured on another."""

CHECKS: dict[str, Check] = {
    "BVP": check_bvp,
    "EDA": check_eda,
    "TEMP": check_temp,
    "ACC": check_acc,
}


# ── assessing a whole epoch ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class QCResult:
    """What quality control made of one epoch, per signal."""

    statuses: dict[str, QCStatus]
    codes: dict[str, tuple[str, ...]]
    policy_version: str

    @property
    def usable(self) -> frozenset[str]:
        """Signals a feature extractor may compute on."""
        return frozenset(n for n, s in self.statuses.items() if s.usable)

    @property
    def rejected(self) -> frozenset[str]:
        return frozenset(n for n, s in self.statuses.items() if not s.usable)

    def all_codes(self) -> tuple[str, ...]:
        return tuple(sorted({c for codes in self.codes.values() for c in codes}))


def assess(
    epoch: Epoch,
    policy: QCPolicy = DEFAULT_POLICY,
    *,
    checks: dict[str, Check] | None = None,
    motion_affects: str | None = "BVP",
) -> QCResult:
    """Judge every signal in one epoch.

    Motion is decided across modalities rather than within one: accelerometry is
    what says whether the participant moved, and the consequence falls on the
    optical pulse. Checking BVP against itself could never find it.

    ``checks`` and ``motion_affects`` are what make this usable for a second
    device. The chest band has no optical sensor, so movement has no channel to
    be charged to there and ``motion_affects`` is None; its signals are named
    differently and carry different checks.
    """
    statuses: dict[str, QCStatus] = {}
    codes: dict[str, tuple[str, ...]] = {}
    table = CHECKS if checks is None else checks

    moved = motion_affects is not None and _in_motion(epoch, policy)
    for name, samples in epoch.samples.items():
        check = table.get(name)
        if check is None:
            continue
        rate = epoch.windows[name].sampling_rate_hz
        found = check(samples, rate, policy, epoch.preprocessing)
        if moved and name == motion_affects:
            found.append(MOTION)

        fatal = [c for c in found if c not in policy.warn_only]
        if fatal:
            status = QCStatus.REJECTED
        elif found:
            status = QCStatus.WARNING
        else:
            status = QCStatus.VALID
        statuses[name] = status
        codes[name] = tuple(found)

    return QCResult(statuses, codes, policy.version)


def apply_to_windows(epoch: Epoch, result: QCResult) -> dict[str, object]:
    """The epoch's windows, carrying their verdicts.

    Rejection produces a new window object rather than mutating one, and leaves
    the identifier alone -- identity is the slice, not the judgement about it.
    """
    return {
        name: window.rejected(*result.codes[name])
        if result.statuses.get(name) is QCStatus.REJECTED
        else window
        for name, window in epoch.windows.items()
    }


def _in_motion(epoch: Epoch, policy: QCPolicy) -> bool:
    samples = epoch.samples.get("ACC")
    if samples is None:
        return False
    array = np.asarray(samples, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or array.size == 0:
        return False
    magnitude = np.linalg.norm(array, axis=1) / policy.acc_g_scale
    return bool(np.std(magnitude) > policy.acc_motion_g)


def _pulse_peaks(signal: np.ndarray, rate: float, policy: QCPolicy) -> np.ndarray:
    """Systolic peaks, spaced by the rate the spectrum says is there."""
    return detect_beats(
        signal,
        rate,
        min_bpm=policy.bvp_min_bpm,
        max_bpm=policy.bvp_detect_ceiling_bpm,
    )
