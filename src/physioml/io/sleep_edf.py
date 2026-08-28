"""Sleep-EDF Expanded: whole nights of polysomnography, scored by hand.

Two electroencephalogram derivations at 100 Hz, an electro-oculogram, submental
electromyography, and a hypnogram an expert scored in 30-second epochs. It is
the dataset that makes the neural half of this project answerable: WESAD has no
EEG, and a pipeline validated against nothing is the thing this repository
exists to argue against.

**The recordings are much longer than the sleep.** A Sleep Cassette night runs
about twenty hours because the recorder was started in the afternoon and
stopped the next morning; the first file read here opens with eight and a half
hours of a single annotation saying the subject is awake. Left alone, wake
becomes three-quarters of the epochs and a classifier that answers "awake"
scores extremely well. So the record is trimmed to the sleep period plus a
margin, the amount of trimming is reported rather than assumed, and the
untrimmed distribution is available for anyone who wants to see what the
trimming did.

**Stages are scored under Rechtschaffen and Kales**, which separates stages 3
and 4. Modern practice merges them into N3, and that is done here, because
keeping them apart invites a five-way comparison against published numbers that
were computed a different way. Movement time and unscored epochs are dropped
rather than assigned to anything.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from physioml.core.recording import Modality, Recording
from physioml.io.edf import EDF

EPOCH_SECONDS = 30.0
"""What a hypnogram epoch is, everywhere in sleep scoring."""

#: Rechtschaffen and Kales as written, mapped to the stages people report.
STAGES: dict[str, str] = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}

#: Present in the files and deliberately not mapped: an epoch nobody scored is
#: not an epoch in some sixth stage.
UNSCORED = {"Sleep stage ?", "Movement time"}

#: What each channel is, for the recordings this produces.
MODALITIES: dict[str, Modality] = {
    "EEG Fpz-Cz": Modality.EEG,
    "EEG Pz-Oz": Modality.EEG,
    "EOG horizontal": Modality.EOG,
    "EMG submental": Modality.EMG,
    "Resp oro-nasal": Modality.RESP,
    "Temp rectal": Modality.TEMP,
}

_NAME = re.compile(r"^SC4(?P<subject>\d\d)(?P<night>\d)")


class SleepEDFError(RuntimeError):
    """Raised when a directory is not the dataset it is expected to be."""


@dataclass(frozen=True, slots=True)
class Night:
    """One night: the signals, the scored stages, and what was trimmed away."""

    subject_id: str
    night: int
    signals: dict[str, np.ndarray]
    rates: dict[str, float]
    stages: np.ndarray
    """One label per 30-second epoch, over the retained span."""

    offset_seconds: float
    """Where the retained span starts in the original recording."""

    recordings: dict[str, Recording]
    trimmed_epochs: int
    """Epochs discarded from the head and tail, nearly all of them wake."""

    unscored_epochs: int

    @property
    def epochs(self) -> int:
        return int(self.stages.size)

    @property
    def duration_seconds(self) -> float:
        return self.epochs * EPOCH_SECONDS

    def counts(self) -> dict[str, int]:
        unique, counts = np.unique(self.stages, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist(), strict=True))


def hypnogram(path: Path | str) -> tuple[np.ndarray, int]:
    """Stage per 30-second epoch, and how many epochs nobody scored.

    The file stores runs -- one annotation covering an hour of stage 2 -- so
    each is expanded to the epochs it spans. Unscored epochs become an empty
    string, which is dropped later rather than treated as a stage.
    """
    found = EDF(path).annotations()
    if not found:
        raise SleepEDFError(f"{Path(path).name} carries no annotations")

    end = max(a.onset_seconds + a.duration_seconds for a in found)
    labels = np.full(round(end / EPOCH_SECONDS), "", dtype="<U4")
    unscored = 0
    for note in found:
        stage = STAGES.get(note.text)
        first = round(note.onset_seconds / EPOCH_SECONDS)
        last = first + round(note.duration_seconds / EPOCH_SECONDS)
        if stage is None:
            if note.text in UNSCORED:
                unscored += max(last - first, 0)
            continue
        labels[first:last] = stage
    return labels, unscored


def _sleep_span(stages: np.ndarray, margin_epochs: int) -> tuple[int, int]:
    """The scored night, plus a margin, in epoch indices.

    Everything outside is the recorder running before the participant went to
    bed and after they got up. Keeping it makes wake the answer to almost every
    epoch.
    """
    asleep = np.flatnonzero((stages != "W") & (stages != ""))
    if asleep.size == 0:
        return 0, stages.size
    first = max(int(asleep[0]) - margin_epochs, 0)
    last = min(int(asleep[-1]) + margin_epochs + 1, stages.size)
    return first, last


class SleepEDF:
    """A directory of Sleep-EDF recordings, addressed by subject."""

    def __init__(self, directory: Path | str, *, study_id: str = "SLEEP-EDF") -> None:
        self.directory = Path(directory)
        self.study_id = study_id
        if not self.directory.is_dir():
            raise SleepEDFError(f"{self.directory} is not a directory")

    def _pairs(self) -> dict[tuple[str, int], tuple[Path, Path]]:
        psg: dict[tuple[str, int], Path] = {}
        hypnograms: dict[tuple[str, int], Path] = {}
        for path in sorted(self.directory.glob("SC4*.edf")):
            found = _NAME.match(path.name)
            if found is None:
                continue
            key = (found["subject"], int(found["night"]))
            if path.name.endswith("-PSG.edf"):
                psg[key] = path
            elif path.name.endswith("-Hypnogram.edf"):
                hypnograms[key] = path
        # A recording without its scoring is unusable, and a scoring without
        # its recording has nothing to score; neither is an error worth
        # stopping for, so both are simply not offered.
        return {k: (psg[k], hypnograms[k]) for k in sorted(psg) if k in hypnograms}

    def nights(self) -> list[tuple[str, int]]:
        return list(self._pairs())

    def subjects(self) -> list[str]:
        return sorted({subject for subject, _ in self._pairs()})

    def read(
        self,
        subject_id: str,
        night: int = 1,
        *,
        channels: tuple[str, ...] = (
            "EEG Fpz-Cz",
            "EEG Pz-Oz",
            "EOG horizontal",
            "EMG submental",
        ),
        margin_minutes: float = 30.0,
    ) -> Night:
        """One night, trimmed to the sleep period plus ``margin_minutes``."""
        pairs = self._pairs()
        key = (subject_id, night)
        if key not in pairs:
            raise SleepEDFError(
                f"no night {night} for subject {subject_id!r}; have {self.nights()}"
            )
        psg_path, hypnogram_path = pairs[key]

        stages, unscored = hypnogram(hypnogram_path)
        margin = round(margin_minutes * 60.0 / EPOCH_SECONDS)
        first, last = _sleep_span(stages, margin)

        edf = EDF(psg_path)
        # EDF records a wall clock with no zone, which is a property of the
        # format rather than an omission here. UTC is attached because the
        # provenance layer will not accept a naive timestamp, and it is a
        # stand-in: every calculation in this project is relative to the start
        # of the recording, so the absolute offset is never used. What must not
        # happen is a timestamp that silently claims to know something the file
        # does not say.
        started = edf.started.replace(tzinfo=UTC) if edf.started else datetime.now(UTC)
        available = int(edf.duration_seconds // EPOCH_SECONDS)
        last = min(last, available)
        if last <= first:
            raise SleepEDFError(f"{psg_path.name} and its scoring do not overlap")

        offset = first * EPOCH_SECONDS
        signals: dict[str, np.ndarray] = {}
        rates: dict[str, float] = {}
        recordings: dict[str, Recording] = {}
        for label in channels:
            if label not in edf.labels:
                continue
            rate = edf.sampling_rate(label)
            samples = edf.read(label)
            begin = round(offset * rate)
            finish = round(last * EPOCH_SECONDS * rate)
            signals[label] = samples[begin:finish]
            rates[label] = rate
            recordings[label] = Recording.create(
                study_id=self.study_id,
                subject_id=subject_id,
                modality=MODALITIES.get(label, Modality.EEG),
                sampling_rate_hz=rate,
                start_time=started,
                duration_seconds=signals[label].size / rate,
                channels=(label,),
                device_name="Sleep Cassette",
            )

        if not signals:
            raise SleepEDFError(
                f"{psg_path.name} has none of {channels}; it holds {list(edf.labels)}"
            )

        return Night(
            subject_id=subject_id,
            night=night,
            signals=signals,
            rates=rates,
            stages=stages[first:last],
            offset_seconds=offset,
            recordings=recordings,
            trimmed_epochs=int(stages.size - (last - first)),
            unscored_epochs=unscored,
        )

    def __iter__(self) -> Iterator[tuple[str, int]]:
        return iter(self.nights())
