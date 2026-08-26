"""Filtering, and recording that it happened.

A photoplethysmogram is not a clean pulse waveform. It carries a slow baseline
from vasomotion and respiration, high-frequency sensor noise, and — most
awkwardly — a dicrotic notch, the secondary bump as the aortic valve closes. A
peak finder run over the raw signal counts that notch as a beat, and the
resulting intervals are not beat-to-beat intervals at all.

The consequence is not subtle and it is not obvious: pulse rate comes out
roughly plausible, while the variability measures derived from the same
intervals come out three to ten times what any human produces. A pipeline that
skips this step still reports SDNN. It reports 300 ms.

**Preprocessing is versioned and recorded.** A window carries the identifier of
the run that produced it, so changing a cutoff produces new windows rather than
the same windows quietly meaning something else. The filter is stated here in
one place for the same reason a study's derivations are: it is a decision, it
affects every number downstream, and it should be reviewable without reading
the code that applies it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physioml.core.provenance import content_id


@dataclass(frozen=True, slots=True)
class Preprocessing:
    """Filter settings for wrist physiology, and the identity they form."""

    version: str = "wrist-e4-1.0"

    bvp_low_hz: float = 0.5
    """Below a plausible resting heart rate, so the baseline goes and the
    slowest real beats stay."""

    bvp_high_hz: float = 8.0
    """Above the harmonics that give a pulse its shape, below the sensor noise.
    The standard band for photoplethysmography."""

    bvp_order: int = 3

    eda_low_pass_hz: float = 1.0
    """Electrodermal activity is slow; anything faster is not skin."""

    @property
    def run_id(self) -> str:
        """Identifier for windows produced under these settings."""
        return content_id(
            "prep",
            {
                "version": self.version,
                "bvp_low_hz": self.bvp_low_hz,
                "bvp_high_hz": self.bvp_high_hz,
                "bvp_order": self.bvp_order,
                "eda_low_pass_hz": self.eda_low_pass_hz,
            },
        )


DEFAULT_PREPROCESSING = Preprocessing()


def bandpass(
    signal: np.ndarray, rate: float, low_hz: float, high_hz: float, order: int = 3
) -> np.ndarray:
    """Zero-phase Butterworth band-pass.

    Filtered forwards and backwards (``filtfilt``) so the filter introduces no
    phase shift. That matters here more than usual: a one-directional filter
    delays the signal by an amount that varies with frequency, which moves the
    peaks, which moves the intervals — the exact quantity being measured.
    """
    from scipy.signal import butter, filtfilt

    data = np.asarray(signal, dtype=float).ravel()
    nyquist = rate / 2.0
    high = min(high_hz, nyquist * 0.99)
    if not 0 < low_hz < high:
        raise ValueError(
            f"band {low_hz}-{high_hz} Hz is not usable at {rate} Hz "
            f"(Nyquist is {nyquist} Hz)"
        )
    # filtfilt needs a few times the filter length to settle.
    if data.size <= order * 6:
        return data

    b, a = butter(order, [low_hz / nyquist, high / nyquist], btype="band")
    return np.asarray(filtfilt(b, a, data), dtype=float)


def lowpass(signal: np.ndarray, rate: float, cutoff_hz: float, order: int = 3):
    """Zero-phase Butterworth low-pass."""
    from scipy.signal import butter, filtfilt

    data = np.asarray(signal, dtype=float).ravel()
    nyquist = rate / 2.0
    cutoff = min(cutoff_hz, nyquist * 0.99)
    if data.size <= order * 6:
        return data
    b, a = butter(order, cutoff / nyquist, btype="low")
    return np.asarray(filtfilt(b, a, data), dtype=float)


def prepare_bvp(
    signal: np.ndarray, rate: float, settings: Preprocessing = DEFAULT_PREPROCESSING
) -> np.ndarray:
    """A photoplethysmogram with the baseline and the noise taken out."""
    return bandpass(
        signal, rate, settings.bvp_low_hz, settings.bvp_high_hz, settings.bvp_order
    )


def dominant_rate_bpm(
    signal: np.ndarray, rate: float, min_bpm: float = 40.0, max_bpm: float = 180.0
) -> float | None:
    """The pulse rate, estimated from the spectrum rather than from peaks.

    Used to *constrain* peak detection, so it must not itself depend on peaks.
    The dominant frequency inside the plausible band is robust to the occasional
    spurious bump that ruins a peak-counting estimate, because one extra bump in
    sixty seconds barely moves the spectrum.
    """
    data = np.asarray(signal, dtype=float).ravel()
    if data.size < 16:
        return None
    spectrum = np.abs(np.fft.rfft(data * np.hanning(data.size)))
    frequencies = np.fft.rfftfreq(data.size, 1.0 / rate)
    band = (frequencies >= min_bpm / 60.0) & (frequencies <= max_bpm / 60.0)
    if not band.any():
        return None
    return float(frequencies[band][np.argmax(spectrum[band])] * 60.0)


def detect_beats(
    signal: np.ndarray,
    rate: float,
    *,
    min_bpm: float = 40.0,
    max_bpm: float = 180.0,
    prominence_sd: float = 0.4,
    refractory: float = 0.6,
) -> np.ndarray:
    """Systolic peak indices, spaced by what the estimated rate allows.

    A fixed minimum spacing cannot work. Set to a physiological ceiling it
    admits spurious peaks between real beats at low heart rates; set tight
    enough to exclude them it deletes real beats at high ones. Measured on
    WESAD, the fixed-ceiling version produced the right *number* of peaks while
    splitting individual beats into a short interval followed by a long one --
    which leaves the mean rate about right and inflates every variability
    measure by a factor of five.

    So the rate is estimated from the spectrum first, and peaks are then
    required to be at least ``refractory`` of one beat apart. 0.6 leaves room
    for genuine beat-to-beat variation while excluding a second peak inside one
    cardiac cycle.
    """
    from scipy.signal import find_peaks

    data = np.asarray(signal, dtype=float).ravel()
    if data.size < 16:
        return np.empty(0, dtype=int)

    estimated = dominant_rate_bpm(data, rate, min_bpm, max_bpm)
    if estimated is None or estimated <= 0:
        spacing = max(int(rate * 60.0 / max_bpm), 1)
    else:
        spacing = max(int(refractory * rate * 60.0 / estimated), 1)

    peaks, _ = find_peaks(
        data, distance=spacing, prominence=float(np.std(data)) * prominence_sd
    )
    return np.asarray(peaks, dtype=int)
