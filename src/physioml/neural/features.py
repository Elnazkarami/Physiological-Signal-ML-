"""What a 30-second epoch of sleep electroencephalography is worth measuring by.

Sleep scoring is a spectral judgement made by eye. A scorer looks for slow
high-amplitude waves, for spindles, for the flattening and eye movement of REM,
and the features here are the arithmetic of those same observations: power in
the bands the stages are defined by, the shape of the spectrum, and the two
supporting channels -- eye movement and chin muscle tone -- that separate REM
from light sleep when the electroencephalogram alone cannot.

Relative band power is emitted alongside absolute. Absolute amplitude varies
several-fold between people for reasons that have nothing to do with sleep --
skull thickness, electrode impedance, the amplifier's gain that night -- and a
model trained on one participant's absolute microvolts and tested on another's
is being asked to generalise across the wrong thing. The relative figures are
what should transfer; both are emitted so the ablation can say whether they do.
"""

from __future__ import annotations

import numpy as np

FEATURE_SET = "sleep-eeg"
FEATURE_SET_VERSION = "1.0"

#: The bands sleep is scored in. Sigma is separated from beta because sleep
#: spindles live there and are one of the defining features of stage 2; folded
#: into a wide beta band they are invisible.
BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}

TOTAL_BAND = (0.5, 30.0)
"""What relative power is relative to. Not the whole spectrum: above 30 Hz a
scalp recording is mostly muscle, and dividing by it would make every relative
figure a function of how tense the participant's jaw was."""


def spectrum(samples: np.ndarray, rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Power spectral density, by Welch, at about a quarter-hertz.

    Four-second segments: long enough to resolve the delta band, which starts
    at 0.5 Hz and would otherwise be a single bin, and short enough that a
    30-second epoch holds several of them to average over.
    """
    from scipy.signal import welch

    data = np.asarray(samples, dtype=float).ravel()
    segment = min(int(rate * 4), data.size)
    if segment < 8:
        return np.empty(0), np.empty(0)
    return welch(data, fs=rate, nperseg=segment)


def band_power(
    frequencies: np.ndarray, power: np.ndarray, band: tuple[float, float]
) -> float:
    """Integrated power in one band, by the trapezium rule.

    Summing the bins instead would make the answer depend on the frequency
    resolution, which changes with the epoch length -- so a 30-second epoch and
    a 20-second one would disagree about the same signal.
    """
    inside = (frequencies >= band[0]) & (frequencies <= band[1])
    if inside.sum() < 2:
        return 0.0
    x, y = frequencies[inside], power[inside]
    return float(np.sum((y[:-1] + y[1:]) / 2.0 * np.diff(x)))


def hjorth(samples: np.ndarray) -> tuple[float, float, float]:
    """Activity, mobility, complexity: the shape of a signal in three numbers.

    Older than the spectral measures and still used in sleep scoring, because
    they are computed in the time domain and say something the band powers do
    not: mobility is a mean frequency and complexity is how far the signal
    departs from a pure sine at that frequency.
    """
    data = np.asarray(samples, dtype=float).ravel()
    if data.size < 3:
        return 0.0, 0.0, 0.0
    variance = float(np.var(data))
    if variance == 0.0:
        return 0.0, 0.0, 0.0

    first = np.diff(data)
    second = np.diff(first)
    var_first = float(np.var(first))
    if var_first == 0.0:
        return variance, 0.0, 0.0

    mobility = np.sqrt(var_first / variance)
    complexity = np.sqrt(float(np.var(second)) / var_first) / mobility
    return variance, float(mobility), float(complexity)


def spectral_entropy(power: np.ndarray) -> float:
    """How evenly the power is spread, normalised to lie in [0, 1].

    Deep sleep concentrates its power at the bottom of the spectrum and scores
    low; wake and REM spread it and score high.
    """
    total = float(np.sum(power))
    if total <= 0 or power.size < 2:
        return 0.0
    share = power / total
    share = share[share > 0]
    return float(-np.sum(share * np.log(share)) / np.log(power.size))


def spectral_edge(frequencies: np.ndarray, power: np.ndarray, share: float = 0.95) -> float:
    """The frequency below which ``share`` of the power lies."""
    if frequencies.size < 2:
        return 0.0
    cumulative = np.cumsum(power)
    if cumulative[-1] <= 0:
        return 0.0
    return float(frequencies[int(np.searchsorted(cumulative / cumulative[-1], share))])


def eeg_features(samples: np.ndarray, rate: float, prefix: str) -> dict[str, float]:
    """Every measure from one electroencephalogram channel."""
    frequencies, power = spectrum(samples, rate)
    if frequencies.size == 0:
        return {}

    total = band_power(frequencies, power, TOTAL_BAND)
    found: dict[str, float] = {}
    for name, band in BANDS.items():
        absolute = band_power(frequencies, power, band)
        found[f"{prefix}_{name}"] = absolute
        found[f"{prefix}_{name}_rel"] = absolute / total if total > 0 else 0.0

    # Ratios two stages are told apart by: slow-wave dominance separates N3,
    # and the balance of the fast bands moves between REM and light sleep.
    delta = found[f"{prefix}_delta"]
    theta = found[f"{prefix}_theta"]
    alpha = found[f"{prefix}_alpha"]
    beta = found[f"{prefix}_beta"]
    found[f"{prefix}_delta_theta_ratio"] = delta / theta if theta > 0 else 0.0
    found[f"{prefix}_alpha_beta_ratio"] = alpha / beta if beta > 0 else 0.0

    activity, mobility, complexity = hjorth(samples)
    found[f"{prefix}_hjorth_activity"] = activity
    found[f"{prefix}_hjorth_mobility"] = mobility
    found[f"{prefix}_hjorth_complexity"] = complexity

    found[f"{prefix}_entropy"] = spectral_entropy(power)
    found[f"{prefix}_edge95"] = spectral_edge(frequencies, power)
    found[f"{prefix}_total_power"] = total

    data = np.asarray(samples, dtype=float).ravel()
    found[f"{prefix}_amplitude_p95"] = float(np.percentile(np.abs(data), 95))
    found[f"{prefix}_zero_crossings"] = float(
        np.count_nonzero(np.diff(np.signbit(data - np.mean(data)))) / (data.size / rate)
    )
    return found


def eog_features(samples: np.ndarray, rate: float) -> dict[str, float]:
    """Eye movement: the other half of the REM definition.

    Rapid eye movements are large, slow deflections. Their power sits below
    2 Hz, so the measure that matters is how much of the signal's energy is
    there and how far it swings, not its spectrum in detail.
    """
    frequencies, power = spectrum(samples, rate)
    data = np.asarray(samples, dtype=float).ravel()
    if frequencies.size == 0 or data.size == 0:
        return {}

    slow = band_power(frequencies, power, (0.3, 2.0))
    total = band_power(frequencies, power, (0.3, 15.0))
    return {
        "eog_slow_power": slow,
        "eog_slow_rel": slow / total if total > 0 else 0.0,
        "eog_amplitude_sd": float(np.std(data)),
        "eog_amplitude_p95": float(np.percentile(np.abs(data), 95)),
        "eog_range": float(np.ptp(data)),
    }


def emg_features(samples: np.ndarray, rate: float) -> dict[str, float]:
    """Chin muscle tone. REM sleep is defined partly by its absence."""
    data = np.asarray(samples, dtype=float).ravel()
    if data.size < 4:
        return {}
    centred = data - float(np.mean(data))
    return {
        "chin_emg_rms": float(np.sqrt(np.mean(centred**2))),
        "chin_emg_p95": float(np.percentile(np.abs(centred), 95)),
        "chin_emg_range": float(np.ptp(centred)),
    }


#: Which channel each family of features comes from, for the ablation. Declared
#: rather than inferred from the name, for the reason the peripheral table is:
#: a prefix rule would misfile anything whose name does not carry its channel.
FEATURES_BY_CHANNEL: dict[str, tuple[str, ...]] = {}


def _register() -> None:
    """Build the channel map from the extractors, once, at import.

    Generated rather than typed out because there are 22 features per
    electroencephalogram channel and a hand-written list of them is a list that
    goes stale silently.
    """
    rate = 100.0
    probe = np.sin(2 * np.pi * 10.0 * np.arange(int(rate * 30)) / rate)
    for label, prefix in (("EEG Fpz-Cz", "fpz"), ("EEG Pz-Oz", "pz")):
        FEATURES_BY_CHANNEL[label] = tuple(eeg_features(probe, rate, prefix))
    FEATURES_BY_CHANNEL["EOG horizontal"] = tuple(eog_features(probe, rate))
    FEATURES_BY_CHANNEL["EMG submental"] = tuple(emg_features(probe, rate))


_register()

CHANNEL_PREFIX = {"EEG Fpz-Cz": "fpz", "EEG Pz-Oz": "pz"}
