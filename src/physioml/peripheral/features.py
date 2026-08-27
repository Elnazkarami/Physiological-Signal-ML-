"""Turning a window of wearable signal into named numbers a model can use.

Four modalities, each answering a different question about the same minute:
the pulse says how fast and how variably the heart is beating, electrodermal
activity says how much the sympathetic nervous system is doing, temperature
says which way peripheral blood flow is going, and accelerometry says whether
the person was moving while the other three were measured.

Three rules shape what is here.

**Every feature is a named scalar.** A feature set that wants a spectrum emits
one feature per band rather than one feature holding a vector, so that a model
selecting "alpha power" selects a named thing rather than an index. It also
means a single wrong feature is identifiable later rather than buried in an
array.

**Nothing is computed on a rejected window.** Quality control runs first and its
verdict is honoured: a heart rate derived from a flatlined sensor is not a
missing value, it is a confident wrong one, and it will be believed.

**The set carries a version.** Change a band definition or a filter and the
version changes with it, which is what makes it possible later to find every
feature computed the old way and every prediction that used one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from physioml.core.feature import Feature
from physioml.core.window import QCStatus
from physioml.peripheral.preprocessing import DEFAULT_PREPROCESSING, prepare_bvp
from physioml.peripheral.qc import DEFAULT_POLICY, QCPolicy, QCResult, _pulse_peaks

if TYPE_CHECKING:
    from physioml.peripheral.windowing import Epoch

#: Bumped whenever any computation below changes. See the module note.
FEATURE_SET = "peripheral-wrist"
FEATURE_SET_VERSION = "1.3"
"""1.1 band-passed the pulse before detecting beats. 1.2 spaced peaks by the
rate estimated from the spectrum. 1.3 removed the pulse-variability measures
altogether after validating them against the chest electrocardiogram: they were
wrong by a factor of 3.6 and no filtering fixed it. Any feature carrying an
earlier version should be recomputed, and any ppi_sdnn, ppi_rmssd or ppi_pnn50
value from one should be discarded rather than recomputed."""


def _f(
    name: str,
    value: float,
    window_id: str,
    subject_id: str,
    unit: str | None = None,
    *,
    feature_set: str = FEATURE_SET,
    feature_set_version: str = FEATURE_SET_VERSION,
):
    return Feature.create(
        subject_id=subject_id,
        name=name,
        value=float(value),
        unit=unit,
        feature_set=feature_set,
        feature_set_version=feature_set_version,
        source_window_ids=(window_id,),
        transform_id=f"{feature_set}@{feature_set_version}",
    )


# ── blood-volume pulse ───────────────────────────────────────────────────────


def bvp_features(
    samples: np.ndarray, rate: float, policy: QCPolicy = DEFAULT_POLICY
) -> dict[str, float]:
    """Rate and variability from a photoplethysmogram.

    **Rate only. The variability measures were removed after being measured.**

    SDNN, RMSSD and pNN50 were computed here and validated against the chest
    electrocardiogram WESAD records alongside the wrist, on the same windows of
    the same subject. Heart rate agreed to within about 7 bpm, which is in line
    with what an optical wrist sensor is expected to give. The variability did
    not agree at all: against a true SDNN of 65 ms the wrist produced 236 ms,
    **3.6 times the real value**.

    Band-passing the signal and spacing peaks by the spectrally-estimated rate
    both improved it and neither fixed it. The cause is not a threshold: at
    64 Hz one sample is 15.6 ms, a meaningful fraction of the 20-60 ms the
    measure is trying to resolve, and every missed or doubled beat enters the
    calculation squared. Getting this right needs artifact correction and
    ectopic-beat handling, which is a project rather than a parameter.

    So the features are not emitted. They would have carried some signal —
    enough to raise a model's score, being correlated with artifact rate and so
    with movement — while being scientifically indefensible. A number that is
    wrong by a factor of four and *looks* useful is worse than an absent one.

    The chest electrocardiogram supports them, and a later feature set for that
    device may provide them. This one is a wrist, and a wrist gives rate.
    """
    # Filtered first. On the raw signal the dicrotic notch is counted as a beat
    # and every variability measure below comes out several times what a human
    # produces -- plausibly enough to be believed.
    signal = prepare_bvp(samples, rate, DEFAULT_PREPROCESSING)
    peaks = _pulse_peaks(signal, rate, policy)
    if peaks.size < 3:
        return {}

    intervals_ms = np.diff(peaks) / rate * 1000.0
    return {
        "pulse_rate_mean": 60000.0 / float(np.mean(intervals_ms)),
        "pulse_rate_median": 60000.0 / float(np.median(intervals_ms)),
        "ppi_mean": float(np.mean(intervals_ms)),
        "pulse_amplitude_mean": float(np.mean(signal[peaks])),
        "pulse_amplitude_sd": float(np.std(signal[peaks])),
        "beat_count": float(peaks.size),
    }


# ── electrodermal activity ───────────────────────────────────────────────────


def eda_features(
    samples: np.ndarray, rate: float, policy: QCPolicy = DEFAULT_POLICY
) -> dict[str, float]:
    """Tonic level, phasic response, and the shape of the minute.

    Tonic and phasic are separated with a moving average rather than a proper
    decomposition (cvxEDA, ledalab). That is a deliberate first pass: the moving
    average is transparent, has no parameters to fit, and does not require the
    signal to obey a model. It is also cruder, and a later feature-set version
    replacing it is exactly the change the version number exists to record.
    """
    signal = np.asarray(samples, dtype=float).ravel()
    if signal.size < 4:
        return {}

    seconds = signal.size / rate
    window = max(int(rate * 4), 1)
    kernel = np.ones(window) / window
    tonic = np.convolve(signal, kernel, mode="same")
    phasic = signal - tonic

    # A skin-conductance response: a rise above a threshold, counted once.
    threshold = 0.01
    rising = phasic > threshold
    onsets = int(np.sum(np.diff(rising.astype(int)) == 1))

    return {
        "eda_mean": float(np.mean(signal)),
        "eda_sd": float(np.std(signal)),
        "eda_min": float(np.min(signal)),
        "eda_max": float(np.max(signal)),
        "eda_slope_per_min": float((signal[-1] - signal[0]) / (seconds / 60.0)),
        "scl_mean": float(np.mean(tonic)),
        "scr_count_per_min": onsets / (seconds / 60.0),
        "scr_amplitude_mean": float(np.mean(phasic[phasic > threshold]))
        if np.any(phasic > threshold)
        else 0.0,
        # Trapezoidal integral, written out. numpy renamed trapz to trapezoid
        # in 2.0 and removed the old name, so neither spelling works across the
        # range this package supports; the arithmetic is two lines and does.
        "eda_auc": float(np.sum(signal[:-1] + signal[1:]) / 2.0 / rate),
    }


# ── skin temperature ─────────────────────────────────────────────────────────


def temp_features(
    samples: np.ndarray, rate: float, policy: QCPolicy = DEFAULT_POLICY
) -> dict[str, float]:
    """Level and direction.

    Peripheral skin temperature falls under sympathetic activation as blood is
    shunted away from the extremities, so the slope over a window carries more
    than the level does — a cold room and a stressed participant produce the
    same mean and opposite trends.
    """
    signal = np.asarray(samples, dtype=float).ravel()
    if signal.size < 2:
        return {}
    seconds = signal.size / rate
    return {
        "temp_mean": float(np.mean(signal)),
        "temp_sd": float(np.std(signal)),
        "temp_min": float(np.min(signal)),
        "temp_max": float(np.max(signal)),
        "temp_slope_per_min": float((signal[-1] - signal[0]) / (seconds / 60.0)),
    }


# ── accelerometry ────────────────────────────────────────────────────────────


def acc_features(
    samples: np.ndarray, rate: float, policy: QCPolicy = DEFAULT_POLICY
) -> dict[str, float]:
    """How much the person moved.

    Included as features in their own right, not only as a quality signal.
    Movement is part of the physiological state being inferred — someone who has
    just been asked to give a speech is not sitting the way they were a minute
    earlier — and a model given the accelerometry can use it rather than being
    confounded by it.
    """
    array = np.asarray(samples, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < 2:
        return {}

    magnitude = np.linalg.norm(array, axis=1) / policy.acc_g_scale
    jerk = np.diff(magnitude) * rate
    return {
        "acc_magnitude_mean": float(np.mean(magnitude)),
        "acc_magnitude_sd": float(np.std(magnitude)),
        "acc_magnitude_max": float(np.max(magnitude)),
        "acc_jerk_mean": float(np.mean(np.abs(jerk))),
        "acc_activity_count": float(np.sum(np.abs(magnitude - np.mean(magnitude)))),
        "acc_x_sd": float(np.std(array[:, 0]) / policy.acc_g_scale),
        "acc_y_sd": float(np.std(array[:, 1]) / policy.acc_g_scale),
        "acc_z_sd": float(np.std(array[:, 2]) / policy.acc_g_scale),
    }


#: Every extractor takes the same arguments, whether or not it needs the policy.
#: A dispatch table whose entries disagree about their signature forces the
#: caller to know which is which, and to say so in casts.
Extractor = Callable[[np.ndarray, float, QCPolicy], dict[str, float]]

EXTRACTORS: dict[str, Extractor] = {
    "BVP": bvp_features,
    "EDA": eda_features,
    "TEMP": temp_features,
    "ACC": acc_features,
}

#: Which signal each feature comes from, for ablation.
#:
#: Declared rather than inferred from the name, because ``beat_count`` and
#: ``scl_mean`` do not carry their sensor in their spelling and a prefix rule
#: would quietly put them in the wrong group. A test asserts every extractor
#: emits only names listed here, and that no name appears under two signals --
#: an ablation built on a stale mapping reports the wrong sensor as the useful
#: one, and nothing about the numbers would look wrong.
FEATURES_BY_SIGNAL: dict[str, tuple[str, ...]] = {
    "BVP": (
        "pulse_rate_mean",
        "pulse_rate_median",
        "ppi_mean",
        "pulse_amplitude_mean",
        "pulse_amplitude_sd",
        "beat_count",
    ),
    "EDA": (
        "eda_mean",
        "eda_sd",
        "eda_min",
        "eda_max",
        "eda_slope_per_min",
        "eda_auc",
        "scl_mean",
        "scr_count_per_min",
        "scr_amplitude_mean",
    ),
    "TEMP": (
        "temp_mean",
        "temp_sd",
        "temp_min",
        "temp_max",
        "temp_slope_per_min",
    ),
    "ACC": (
        "acc_magnitude_mean",
        "acc_magnitude_sd",
        "acc_magnitude_max",
        "acc_jerk_mean",
        "acc_activity_count",
        "acc_x_sd",
        "acc_y_sd",
        "acc_z_sd",
    ),
}


def signal_of(feature_name: str) -> str | None:
    """The signal a feature came from, or ``None`` if it is not one of ours."""
    for signal, names in FEATURES_BY_SIGNAL.items():
        if feature_name in names:
            return signal
    return None


UNITS = {
    "pulse_rate_mean": "bpm",
    "pulse_rate_median": "bpm",
    "ppi_mean": "ms",
    "eda_mean": "uS",
    "eda_sd": "uS",
    "eda_min": "uS",
    "eda_max": "uS",
    "scl_mean": "uS",
    "scr_amplitude_mean": "uS",
    "eda_slope_per_min": "uS/min",
    "temp_mean": "Cel",
    "temp_sd": "Cel",
    "temp_min": "Cel",
    "temp_max": "Cel",
    "temp_slope_per_min": "Cel/min",
}


def extract(
    epoch: Epoch,
    qc: QCResult | None = None,
    policy: QCPolicy = DEFAULT_POLICY,
    *,
    extractors: dict[str, Extractor] | None = None,
    feature_set: str = FEATURE_SET,
    feature_set_version: str = FEATURE_SET_VERSION,
) -> list[Feature]:
    """Every feature computable from one epoch, honouring quality control.

    Signals quality control rejected are skipped, so their features are absent
    rather than wrong. A window flagged only with a warning is computed: motion
    makes a pulse feature less reliable, and dropping it would discard the
    accelerometry that says so.
    """
    made: list[Feature] = []
    table = EXTRACTORS if extractors is None else extractors
    for name, samples in epoch.samples.items():
        extractor = table.get(name)
        if extractor is None:
            continue
        if qc is not None and qc.statuses.get(name, QCStatus.VALID) is QCStatus.REJECTED:
            continue

        window = epoch.windows[name]
        rate = window.sampling_rate_hz
        values = extractor(samples, rate, policy)
        made.extend(
            _f(
                key,
                value,
                window.window_id,
                epoch.subject_id,
                UNITS.get(key),
                feature_set=feature_set,
                feature_set_version=feature_set_version,
            )
            for key, value in values.items()
            if np.isfinite(value)
        )
    return made
