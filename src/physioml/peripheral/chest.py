"""The chest device: electrocardiogram, respiration, muscle activity.

WESAD records a RespiBAN on the chest alongside the wrist band, at 700 Hz. Two
things follow from that rate and that placement.

**Heart-rate variability becomes defensible.** It was removed from the wrist
feature set after measurement: against a true SDNN of 65 ms the optical wrist
sensor produced 236 ms, and at 64 Hz one sample is 15.6 ms, a large fraction of
the 20-60 ms the measure resolves. At 700 Hz one sample is 1.43 ms, and the
electrocardiogram's R wave is a sharp deflection rather than a broad optical
pulse. The same features that were indefensible on the wrist are ordinary here,
and they are the reference the wrist was judged against in the first place.

**Respiration and muscle activity have no wrist equivalent.** Breathing rate is
one of the few autonomic measures that moves under stress for a reason a
reviewer can state, and trapezius EMG reads a physical response -- tension --
that no other channel here sees.

This module is the second device, not a replacement. Whether it earns its place
next to a wrist band, which is what somebody would actually wear, is a question
for the fusion ablation rather than for this docstring.
"""

from __future__ import annotations

import numpy as np

from physioml.peripheral.features import (
    Extractor,
    acc_features,
    eda_features,
    temp_features,
)
from physioml.peripheral.preprocessing import bandpass, dominant_rate_bpm
from physioml.peripheral.qc import (
    DISCONTINUITY,
    MOTION,
    Check,
    QCPolicy,
    QCResult,
    assess,
    check_acc,
    check_eda,
    check_temp,
)
from physioml.peripheral.windowing import Epoch

CHEST_FEATURE_SET = "chest-respiban"
CHEST_FEATURE_SET_VERSION = "1.0"

#: Chest accelerometry arrives in g. The wrist band's does not -- the E4 reports
#: raw counts scaled by 64 -- so the policy that reads one cannot read the other.
CHEST_POLICY = QCPolicy(
    version="chest-respiban-1.0",
    acc_g_scale=1.0,
    acc_rail=16.0,
    acc_motion_g=0.08,
    warn_only=frozenset(
        {MOTION, DISCONTINUITY, "unstable_intervals", "beat_count_disagrees"}
    ),
)
"""Chest quality control, and two codes that are recorded rather than fatal.

Both were fatal first, and both were measured. Across 359 windows of two
subjects with enough clean beats to score:

===============================  ======  ==========  =========
policy                           kept    RMSSD med   SDNN med
===============================  ======  ==========  =========
interval cleaning only           100%    54.9 ms     82.0 ms
also reject >20% cleaned         79.1%   51.6 ms     73.2 ms
also reject beat-count disagree  90.8%   54.9 ms     77.7 ms
both                             73.3%   51.5 ms     70.9 ms
===============================  ======  ==========  =========

The beat-count check moves the median not at all and costs nine per cent of
the data. The cleaning-volume check moves it three milliseconds and costs
twenty-one per cent. Neither pays for itself once the intervals themselves are
cleaned, and rejecting a third of the cardiac windows puts the feature below
any sensible coverage threshold, at which point the table loses heart-rate
variability altogether rather than losing the questionable windows.

So they mark the window and the features are still emitted. Moving either out
of this set makes it fatal again, which is what a study that wants the stricter
policy should do -- with a new version string, because it is a different
feature set."""

#: Physiologically possible R-R intervals. Anything outside is a detection
#: failure rather than a heartbeat, and every variability measure below squares
#: its errors, so one 4-second "interval" would dominate a whole window.
RR_MIN_MS = 300.0
RR_MAX_MS = 2000.0

MALIK_TOLERANCE = 0.20
"""How far one R-R interval may sit from the one before it and still be a
heartbeat. The conventional value, and the one the measurements here support:
applied to intervals rather than to whole windows it keeps 87% of them and
brings the median RMSSD from 76 ms to 50 ms."""

MAX_DROPPED_INTERVALS = 0.20
"""Above this the window is refused rather than cleaned. Cleaning a fifth of a
minute's beats is repair; cleaning more is deciding what the signal should have
said."""

BEAT_COUNT_TOLERANCE = 0.10
"""How far the counted beats may sit from the spectral rate before the window
is refused. Ten per cent of a 70 bpm minute is seven beats."""

MIN_BEATS = 20
"""Below this a 60-second window has not seen enough cardiac cycles to say
anything about variability. At 40 bpm a minute holds 40 beats, so this admits
windows where a third of the beats were missed and refuses the rest."""


# ── electrocardiogram ────────────────────────────────────────────────────────


def qrs_envelope(samples: np.ndarray, rate: float) -> tuple[np.ndarray, np.ndarray]:
    """The band-passed signal and its energy envelope.

    Shared by detection and quality control so both judge the same thing. The
    envelope pulses once per beat whatever the polarity of the lead, which is
    what makes a spectral estimate of the heart rate possible at all.
    """
    filtered = bandpass(np.asarray(samples, dtype=float).ravel(), rate, 5.0, 15.0)
    energy = np.diff(filtered, prepend=filtered[0]) ** 2
    span = max(int(0.15 * rate), 1)
    return filtered, np.convolve(energy, np.ones(span) / span, mode="same")


def detect_r_peaks(
    samples: np.ndarray,
    rate: float,
    *,
    min_bpm: float = 40.0,
    max_bpm: float = 200.0,
    refractory: float = 0.6,
) -> np.ndarray:
    """R-wave indices, by band-pass, energy integration, and refinement.

    The classical shape: band-pass to the QRS band, differentiate, square, and
    integrate over a window about as long as a QRS complex, which turns each
    beat into one broad hill regardless of the polarity of the lead.

    The refinement at the end is what makes the result usable for variability.
    A 150 ms integration window smears the peak, so an interval measured
    between two integrated maxima can be tens of milliseconds off -- the same
    order as the quantity being measured. Each peak is therefore moved to the
    largest absolute deflection of the band-passed signal within 60 ms, which
    puts it on the R wave itself.
    """
    from scipy.signal import find_peaks

    data = np.asarray(samples, dtype=float).ravel()
    if data.size < int(rate):
        return np.empty(0, dtype=int)

    filtered, integrated = qrs_envelope(data, rate)

    # The rate comes from the envelope, not from the band-passed signal. The
    # band-pass is 5-15 Hz, so the dominant frequency of what comes out of it
    # is the QRS band -- measured on WESAD it reported about 193 bpm for a
    # 70 bpm heart. The envelope pulses once per beat, which is the quantity
    # the refractory spacing below is meant to be a fraction of.
    estimated = dominant_rate_bpm(integrated, rate, min_bpm, max_bpm)
    if estimated is None or estimated <= 0:
        spacing = max(int(rate * 60.0 / max_bpm), 1)
    else:
        spacing = max(int(refractory * rate * 60.0 / estimated), 1)

    peaks, _ = find_peaks(integrated, distance=spacing, height=float(np.mean(integrated)))
    if peaks.size == 0:
        return peaks.astype(int)

    reach = max(int(0.06 * rate), 1)
    refined = []
    magnitude = np.abs(filtered)
    for peak in peaks:
        low = max(int(peak) - reach, 0)
        high = min(int(peak) + reach + 1, magnitude.size)
        refined.append(low + int(np.argmax(magnitude[low:high])))
    return np.unique(np.asarray(refined, dtype=int))


def rr_intervals_ms(peaks: np.ndarray, rate: float) -> np.ndarray:
    """Beat-to-beat intervals, with impossible ones discarded.

    Only an absolute range is applied. A relative filter -- rejecting an
    interval that differs too much from its neighbour -- was tried on the wrist
    and threw away a fifth of perfectly good windows, because real beat-to-beat
    variation is larger than the intuition behind such a rule. What is removed
    here could not have come from a heart.
    """
    if peaks.size < 2:
        return np.empty(0, dtype=float)
    intervals = np.diff(np.sort(peaks)) / rate * 1000.0
    return intervals[(intervals >= RR_MIN_MS) & (intervals <= RR_MAX_MS)]


def clean_intervals(
    intervals: np.ndarray, tolerance: float = MALIK_TOLERANCE
) -> np.ndarray:
    """Which intervals are physiologically continuous with the one before.

    An interval that differs from its predecessor by more than ``tolerance``
    did not come from the next heartbeat: it came from a missed beat, which
    merges two intervals, or a spurious one, which splits one. Both enter every
    variability measure squared.

    **This is the filter that was rejected for the wrist**, and the difference
    is worth stating because the rule looks identical. There the intervals were
    wrong systematically -- 236 ms of SDNN against a true 65 -- and a relative
    filter deleted 61 of 200 perfectly good windows, because real beat-to-beat
    variation is larger than the intuition behind such a rule. Here the
    intervals are accurate: on synthetic beats at this sampling rate the
    recovered SDNN is within about 1 ms of the constructed value. What the
    filter removes is occasional detection failure on an otherwise sound
    signal, which is what it is for.

    The comparison is against the last *accepted* interval, not the last one,
    so a single artifact does not drag its successor out with it.
    """
    if intervals.size == 0:
        return np.zeros(0, dtype=bool)
    keep = np.ones(intervals.size, dtype=bool)
    previous = float(intervals[0])
    for i in range(1, intervals.size):
        current = float(intervals[i])
        if abs(current - previous) > tolerance * previous:
            keep[i] = False
        else:
            previous = current
    return keep


def ecg_features(
    samples: np.ndarray, rate: float, policy: QCPolicy = CHEST_POLICY
) -> dict[str, float]:
    """Rate and variability from the electrocardiogram.

    Time-domain measures only. Frequency-domain variability -- the LF/HF ratio
    a stress paper usually reaches for -- needs several minutes to resolve the
    low-frequency band, and these windows are one minute long. Reporting it
    from 60 seconds would produce a number with a respectable name and no
    support underneath it.
    """
    peaks = detect_r_peaks(samples, rate)
    intervals = rr_intervals_ms(peaks, rate)
    keep = clean_intervals(intervals)
    retained = intervals[keep]
    if retained.size < MIN_BEATS - 1:
        return {}

    # Successive differences are taken only between intervals that are both
    # retained *and* adjacent. Differencing the retained series alone would
    # measure across every gap the cleaning made, inventing a large successive
    # difference exactly where an artifact was removed.
    adjacent = keep[:-1] & keep[1:]
    successive = np.diff(intervals)[adjacent]
    if successive.size == 0:
        return {}

    return {
        "chest_hr_mean": 60000.0 / float(np.mean(retained)),
        "chest_hr_median": 60000.0 / float(np.median(retained)),
        "chest_rr_mean": float(np.mean(retained)),
        "chest_sdnn": float(np.std(retained, ddof=1)),
        "chest_rmssd": float(np.sqrt(np.mean(successive**2))),
        "chest_pnn50": float(np.mean(np.abs(successive) > 50.0)),
        "chest_beat_count": float(retained.size + 1),
    }


# ── respiration ──────────────────────────────────────────────────────────────


def resp_features(
    samples: np.ndarray, rate: float, policy: QCPolicy = CHEST_POLICY
) -> dict[str, float]:
    """Breathing rate and depth from the inductive band.

    Rate is taken from the spectrum rather than by counting zero crossings: the
    signal drifts with posture, and a drifting baseline crosses zero for
    reasons that have nothing to do with breathing.
    """
    data = np.asarray(samples, dtype=float).ravel()
    if data.size < int(rate * 10):
        return {}

    filtered = bandpass(data, rate, RESP_BAND_HZ[0], RESP_BAND_HZ[1])
    found = {
        "chest_resp_amplitude_sd": float(np.std(filtered)),
        "chest_resp_range": float(np.ptp(filtered)),
    }
    rate_bpm = _breathing_rate(data, rate)
    if rate_bpm is not None:
        found["chest_resp_rate_bpm"] = rate_bpm
    return found


#: The band amplitude is measured in, and the range a breathing rate is sought
#: in -- 6 to 36 breaths per minute, expressed in the units a heart rate uses.
RESP_BAND_HZ = (0.1, 0.6)
RESP_MIN_BPM = 6.0
RESP_MAX_BPM = 36.0


def _breathing_rate(samples: np.ndarray, rate: float) -> float | None:
    """Breaths per minute, from the spectrum of the detrended raw signal.

    Not from the band-passed one. A Butterworth band-pass has its own shape,
    and when the input is dominated by postural drift the output peaks where
    the *filter* peaks rather than where the participant breathes. Measured on
    WESAD that produced a column reading 6.00 breaths per minute for every
    window of every subject, and after the band was narrowed, 13.00 for every
    window -- a constant that looks like a measurement. Welch on the detrended
    signal gives 8, 12, 20 and 22 breaths per minute for four windows of one
    subject, which is what breathing does.
    """
    from scipy.signal import detrend, welch

    data = np.asarray(samples, dtype=float).ravel()
    if data.size < int(rate * 20):
        return None
    segment = max(int(rate * 30), 256)
    frequencies, power = welch(detrend(data), fs=rate, nperseg=min(segment, data.size))
    band = (frequencies >= RESP_MIN_BPM / 60.0) & (frequencies <= RESP_MAX_BPM / 60.0)
    if not band.any():
        return None
    in_band = power[band]
    if not np.any(in_band > 0):
        # A flat or absent signal. argmax over zeros returns the first index,
        # which would report the bottom of the search range as a breathing
        # rate -- a number where there is no breathing.
        return None
    peak = float(frequencies[band][int(np.argmax(in_band))])
    if peak <= 0:
        return None
    return peak * 60.0


# ── muscle activity ──────────────────────────────────────────────────────────


def emg_features(
    samples: np.ndarray, rate: float, policy: QCPolicy = CHEST_POLICY
) -> dict[str, float]:
    """Trapezius muscle activity: amplitude, not spectrum.

    Surface EMG amplitude is what tension shows up in. The median frequency,
    which is the other thing usually reported, tracks fatigue over sustained
    contraction and has no clear meaning in a minute of sitting.
    """
    data = np.asarray(samples, dtype=float).ravel()
    if data.size < 16:
        return {}
    centred = data - float(np.mean(data))
    absolute = np.abs(centred)
    return {
        "chest_emg_rms": float(np.sqrt(np.mean(centred**2))),
        "chest_emg_mav": float(np.mean(absolute)),
        "chest_emg_p95": float(np.percentile(absolute, 95)),
    }


# ── the shared channels, under chest names ───────────────────────────────────


def _prefixed(found: dict[str, float]) -> dict[str, float]:
    """Chest features carry the device in the name.

    A fused table holds a wrist and a chest electrodermal level in the same
    row, and two columns called ``eda_mean`` would be one column with the
    second silently overwriting the first.
    """
    return {f"chest_{name}": value for name, value in found.items()}


def chest_eda_features(
    samples: np.ndarray, rate: float, policy: QCPolicy = CHEST_POLICY
) -> dict[str, float]:
    return _prefixed(eda_features(samples, rate, policy))


def chest_temp_features(
    samples: np.ndarray, rate: float, policy: QCPolicy = CHEST_POLICY
) -> dict[str, float]:
    return _prefixed(temp_features(samples, rate, policy))


def chest_acc_features(
    samples: np.ndarray, rate: float, policy: QCPolicy = CHEST_POLICY
) -> dict[str, float]:
    return _prefixed(acc_features(samples, rate, policy))


#: Keyed by the names WESAD gives the chest signals.
CHEST_EXTRACTORS: dict[str, Extractor] = {
    "ECG": ecg_features,
    "Resp": resp_features,
    "EMG": emg_features,
    "EDA": chest_eda_features,
    "Temp": chest_temp_features,
    "ACC": chest_acc_features,
}

CHEST_FEATURES_BY_SIGNAL: dict[str, tuple[str, ...]] = {
    "ECG": (
        "chest_hr_mean",
        "chest_hr_median",
        "chest_rr_mean",
        "chest_sdnn",
        "chest_rmssd",
        "chest_pnn50",
        "chest_beat_count",
    ),
    "Resp": (
        "chest_resp_rate_bpm",
        "chest_resp_amplitude_sd",
        "chest_resp_range",
    ),
    "EMG": ("chest_emg_rms", "chest_emg_mav", "chest_emg_p95"),
    "EDA": (
        "chest_eda_mean",
        "chest_eda_sd",
        "chest_eda_min",
        "chest_eda_max",
        "chest_eda_slope_per_min",
        "chest_eda_auc",
        "chest_scl_mean",
        "chest_scr_count_per_min",
        "chest_scr_amplitude_mean",
    ),
    "Temp": (
        "chest_temp_mean",
        "chest_temp_sd",
        "chest_temp_min",
        "chest_temp_max",
        "chest_temp_slope_per_min",
    ),
    "ACC": (
        "chest_acc_magnitude_mean",
        "chest_acc_magnitude_sd",
        "chest_acc_magnitude_max",
        "chest_acc_jerk_mean",
        "chest_acc_activity_count",
        "chest_acc_x_sd",
        "chest_acc_y_sd",
        "chest_acc_z_sd",
    ),
}


# ── quality control ──────────────────────────────────────────────────────────


def check_ecg(samples: np.ndarray, rate: float, policy: QCPolicy) -> list[str]:
    """Whether this electrocardiogram can be measured from.

    A rate check is not enough on its own and is not enough here either: the
    band-passed derivative of noise has peaks at plausible spacing. What
    separates them is beat count against the rate the spectrum reports -- a
    detector finding half the beats it should is finding something other than
    this heart.
    """
    data = np.asarray(samples, dtype=float).ravel()
    codes: list[str] = []
    if data.size < int(rate):
        return ["too_short"]

    if float(np.std(data)) < 1e-6:
        return ["flatline"]

    rails = float(np.max(np.abs(data)))
    if rails > 0 and float(np.mean(np.abs(data) >= rails * 0.999)) > 0.02:
        codes.append("clipped")

    peaks = detect_r_peaks(data, rate)
    intervals = rr_intervals_ms(peaks, rate)
    if intervals.size < MIN_BEATS - 1:
        codes.append("too_few_beats")
        return codes

    bpm = 60000.0 / float(np.mean(intervals))
    if not 40.0 <= bpm <= 180.0:
        codes.append("implausible_rate")

    # Counted beats against the rate the spectrum reports. A detector that
    # misses one beat in ten still produces a plausible-looking heart rate --
    # about 10% low -- and a variability that is nonsense, because the missed
    # beat merges two intervals into one and every measure below squares that
    # difference. Measured across three subjects, roughly 8% of windows
    # disagree by more than a tenth, and no detection threshold tried moved
    # that: median disagreement stayed at 2-3% and the tail stayed where it
    # was. So the tail is refused rather than tuned away.
    expected = dominant_rate_bpm(qrs_envelope(data, rate)[1], rate, 40.0, 200.0)
    if expected is None:
        codes.append("no_cardiac_rhythm")
    elif abs(peaks.size - expected) / expected > BEAT_COUNT_TOLERANCE:
        codes.append("beat_count_disagrees")

    # More than a fifth of intervals thrown out as physiologically impossible
    # means the detector is not tracking this signal, whatever the survivors
    # average to.
    kept = intervals.size / max(peaks.size - 1, 1)
    if kept < 0.8:
        codes.append("unstable_detection")

    dropped = 1.0 - float(np.mean(clean_intervals(intervals)))
    if dropped > MAX_DROPPED_INTERVALS:
        codes.append("unstable_intervals")
    return codes


def check_resp(samples: np.ndarray, rate: float, policy: QCPolicy) -> list[str]:
    data = np.asarray(samples, dtype=float).ravel()
    if data.size < int(rate * 10):
        return ["too_short"]
    if float(np.std(data)) < 1e-6:
        return ["flatline"]
    if _breathing_rate(data, rate) is None:
        return ["no_breathing_rhythm"]
    return []


def check_emg(samples: np.ndarray, rate: float, policy: QCPolicy) -> list[str]:
    data = np.asarray(samples, dtype=float).ravel()
    if data.size < 16:
        return ["too_short"]
    if float(np.std(data)) < 1e-9:
        return ["flatline"]
    return []


#: Keyed by the names WESAD gives the chest signals, which are not the wrist
#: names -- the chest thermometer is ``Temp`` where the wrist one is ``TEMP``.
#: Keying this table by the wrist spelling would silently leave chest
#: temperature unchecked, and an unchecked signal still produces features.
CHEST_CHECKS: dict[str, Check] = {
    "ECG": check_ecg,
    "Resp": check_resp,
    "EMG": check_emg,
    "EDA": check_eda,
    "Temp": check_temp,
    "ACC": check_acc,
}


def assess_chest(epoch: Epoch, policy: QCPolicy = CHEST_POLICY) -> QCResult:
    """Quality control for the chest band.

    No signal is charged for movement. On the wrist, motion is charged to the
    optical pulse because that is what an optical sensor loses to it; the chest
    electrodes measure a potential difference and a moving participant is a
    participant, not an artifact.
    """
    return assess(epoch, policy, checks=CHEST_CHECKS, motion_affects=None)
