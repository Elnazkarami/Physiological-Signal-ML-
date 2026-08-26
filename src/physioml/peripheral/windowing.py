"""Cutting continuous signals into the intervals a model is trained on.

Two things make this less trivial than slicing an array.

**Modalities are sampled at different rates.** On the Empatica E4, BVP arrives
at 64 Hz, accelerometry at 32 Hz and EDA and temperature at 4 Hz. A window is
therefore defined by a *time interval* and each signal contributes however many
samples that interval holds. Windowing by sample count instead would silently
give the modalities different durations, and a feature comparing them would be
comparing different stretches of the recording.

**A window that straddles a protocol boundary has no label.** WESAD's stretch
of stress does not end where the participant stopped being stressed, but a
window half in one condition and half in the next is not sixty percent of the
first: it is unusable, and training on it teaches the transition. Such windows
are produced and marked unlabelled rather than quietly dropped, so how many were
lost stays countable.

On overlap: a stride shorter than the window means consecutive windows share
samples and are not independent. That inflates the apparent size of a dataset
without adding information, which is worth remembering when a per-window metric
looks reassuring. It does not create leakage *across* subjects, which is the
error that matters, and subject-wise splits are what prevent that.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np

from physioml.core.window import SignalWindow
from physioml.io.wesad import LABEL_HZ, SubjectData, condition_of


@dataclass(frozen=True, slots=True)
class Epoch:
    """One time interval, across every modality recorded in it."""

    subject_id: str
    index: int
    start_seconds: float
    duration_seconds: float
    label: str | None
    """The protocol condition, or ``None`` when the interval spans more than one."""

    windows: dict[str, SignalWindow]
    samples: dict[str, np.ndarray]
    """Views into the subject's arrays — slicing copies nothing."""

    @property
    def labelled(self) -> bool:
        return self.label is not None

    def __str__(self) -> str:
        state = self.label or "unlabelled"
        return f"{self.subject_id}#{self.index} {self.start_seconds:.0f}s {state}"


def epochs(
    data: SubjectData,
    *,
    length_seconds: float = 60.0,
    stride_seconds: float = 5.0,
    preprocessing_run_id: str = "",
) -> list[Epoch]:
    """Cut one subject into overlapping intervals.

    Defaults follow the WESAD literature: a sixty-second window, long enough for
    the heart-rate variability measures to mean anything, stepped every five
    seconds.
    """
    if length_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("window length and stride must be positive")
    if stride_seconds > length_seconds:
        raise ValueError(
            f"stride {stride_seconds}s exceeds window {length_seconds}s, which would "
            "skip signal between windows"
        )

    total = data.duration_seconds
    made: list[Epoch] = []
    index = 0
    start = 0.0
    while start + length_seconds <= total:
        made.append(_epoch(data, index, start, length_seconds, preprocessing_run_id))
        index += 1
        start += stride_seconds
    return made


def _epoch(
    data: SubjectData,
    index: int,
    start: float,
    length: float,
    preprocessing_run_id: str,
) -> Epoch:
    label_slice = data.labels[round(start * LABEL_HZ) : round((start + length) * LABEL_HZ)]
    label = condition_of(label_slice)

    windows: dict[str, SignalWindow] = {}
    samples: dict[str, np.ndarray] = {}
    for name, array in data.signals.items():
        recording = data.recordings[name]
        rate = recording.sampling_rate_hz
        first = round(start * rate)
        last = first + round(length * rate)
        if last > array.shape[0]:
            continue
        windows[name] = SignalWindow.create(
            recording_id=recording.recording_id,
            subject_id=data.subject_id,
            start_sample=first,
            end_sample=last,
            start_time=recording.start_time + timedelta(seconds=start),
            sampling_rate_hz=rate,
            channel_ids=recording.channels,
            preprocessing_run_id=preprocessing_run_id,
            label=label,
        )
        samples[name] = array[first:last]

    return Epoch(
        subject_id=data.subject_id,
        index=index,
        start_seconds=start,
        duration_seconds=length,
        label=label,
        windows=windows,
        samples=samples,
    )


def label_counts(made: list[Epoch]) -> dict[str, int]:
    """How many epochs fell in each condition, and how many in none."""
    counts: dict[str, int] = {}
    for epoch in made:
        key = epoch.label or "unlabelled"
        counts[key] = counts.get(key, 0) + 1
    return counts
