"""Reading European Data Format, which is what sleep recordings arrive in.

EDF is a header of fixed-width ASCII fields followed by interleaved 16-bit
records, and EDF+ adds a reserved signal carrying time-stamped annotations.
Both are documented and small enough to read directly, which is why this is
here rather than a dependency: the alternative pulls in a large toolbox to
parse a few hundred bytes of header, and the parsing is the part that has to be
right.

Two things this does not do, because doing them silently is how a reader
becomes a source of quiet error. It does not resample -- signals recorded at
different rates are returned at the rates they were recorded at. And it does
not load a whole recording to hand back one channel: a night of polysomnography
is a few hundred megabytes, the records interleave every signal, and the file
is memory-mapped so that reading one channel touches only its own columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

HEADER_BYTES = 256
ANNOTATION_LABEL = "EDF Annotations"

#: The three separators of a time-stamped annotation list.
_ONSET_END = b"\x15"
_TAL_END = b"\x14"
_PADDING = b"\x00"


class EDFError(RuntimeError):
    """Raised when a file is not the EDF it claims to be."""


@dataclass(frozen=True, slots=True)
class Channel:
    """One signal's header. Everything needed to turn integers into volts."""

    label: str
    transducer: str
    unit: str
    physical_min: float
    physical_max: float
    digital_min: float
    digital_max: float
    prefilter: str
    samples_per_record: int

    @property
    def scale(self) -> float:
        """Physical units per digital step.

        A recording whose digital range is a single point cannot be scaled;
        rather than dividing by zero and returning infinities that look like
        readings, such a channel is passed through unscaled.
        """
        span = self.digital_max - self.digital_min
        if span == 0:
            return 1.0
        return (self.physical_max - self.physical_min) / span


@dataclass(frozen=True, slots=True)
class Annotation:
    """One EDF+ event: when it started, how long it lasted, what it was."""

    onset_seconds: float
    duration_seconds: float
    text: str


def _ascii(raw: bytes) -> str:
    return raw.decode("ascii", errors="replace").strip()


def _number(raw: bytes, what: str) -> float:
    text = _ascii(raw)
    try:
        return float(text)
    except ValueError as exc:
        raise EDFError(f"{what} is not a number: {text!r}") from exc


class EDF:
    """An EDF or EDF+ recording, addressed by channel label."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        with self.path.open("rb") as handle:
            head = handle.read(HEADER_BYTES)
            if len(head) < HEADER_BYTES:
                raise EDFError(f"{self.path.name} is too short to hold an EDF header")

            self.version = _ascii(head[0:8])
            self.patient = _ascii(head[8:88])
            self.recording = _ascii(head[88:168])
            self.started = self._start_time(head[168:176], head[176:184])
            self.header_bytes = int(_number(head[184:192], "header length"))
            self.reserved = _ascii(head[192:236])
            declared_records = int(_number(head[236:244], "number of records"))
            self.record_seconds = _number(head[244:252], "record duration")
            count = int(_number(head[252:256], "number of signals"))
            if count <= 0:
                raise EDFError(f"{self.path.name} declares {count} signals")

            self.channels = tuple(self._channels(handle, count))

        self.samples_per_record = sum(c.samples_per_record for c in self.channels)
        self.records = self._record_count(declared_records)

    # ── header ───────────────────────────────────────────────────────────────

    @staticmethod
    def _start_time(date: bytes, time: bytes) -> datetime | None:
        """The recording's start, or None if the file does not say.

        EDF writes a two-digit year and the specification clips it: 85 to 99
        mean 1985 to 1999, 00 to 84 mean 2000 to 2084. Guessing the century any
        other way would put a 1989 sleep study in 2089.
        """
        try:
            day, month, year = (int(p) for p in _ascii(date).split("."))
            hour, minute, second = (int(p) for p in _ascii(time).split("."))
        except ValueError:
            return None
        century = 1900 if year >= 85 else 2000
        try:
            return datetime(century + year, month, day, hour, minute, second)
        except ValueError:
            return None

    def _channels(self, handle: Any, count: int) -> list[Channel]:
        def block(width: int) -> list[bytes]:
            raw = handle.read(width * count)
            return [raw[i * width : (i + 1) * width] for i in range(count)]

        labels = block(16)
        transducers = block(80)
        units = block(8)
        physical_min = block(8)
        physical_max = block(8)
        digital_min = block(8)
        digital_max = block(8)
        prefilters = block(80)
        samples = block(8)
        block(32)  # reserved, per signal

        made = []
        for i in range(count):
            made.append(
                Channel(
                    label=_ascii(labels[i]),
                    transducer=_ascii(transducers[i]),
                    unit=_ascii(units[i]),
                    physical_min=_number(physical_min[i], "physical minimum"),
                    physical_max=_number(physical_max[i], "physical maximum"),
                    digital_min=_number(digital_min[i], "digital minimum"),
                    digital_max=_number(digital_max[i], "digital maximum"),
                    prefilter=_ascii(prefilters[i]),
                    samples_per_record=int(_number(samples[i], "samples per record")),
                )
            )
        return made

    def _record_count(self, declared: int) -> int:
        """How many records there are, believing the file over the header.

        EDF allows -1 for "not known", written by anything recording to a file
        it has not finished. The size on disk knows either way.
        """
        available = self.path.stat().st_size - self.header_bytes
        possible = available // (self.samples_per_record * 2)
        if declared < 0:
            return int(possible)
        if declared > possible:
            raise EDFError(
                f"{self.path.name} declares {declared} records but holds {possible}; "
                "the file is truncated"
            )
        return declared

    # ── signals ──────────────────────────────────────────────────────────────

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(c.label for c in self.channels)

    @property
    def duration_seconds(self) -> float:
        return self.records * self.record_seconds

    def channel(self, label: str) -> Channel:
        for found in self.channels:
            if found.label == label:
                return found
        raise KeyError(f"{label!r} is not in {self.path.name}: {list(self.labels)}")

    def sampling_rate(self, label: str) -> float:
        return self.channel(label).samples_per_record / self.record_seconds

    def _columns(self, label: str) -> tuple[int, int]:
        start = 0
        for found in self.channels:
            if found.label == label:
                return start, start + found.samples_per_record
            start += found.samples_per_record
        raise KeyError(f"{label!r} is not in {self.path.name}")

    def _mapped(self) -> np.ndarray:
        return np.memmap(
            self.path,
            dtype="<i2",
            mode="r",
            offset=self.header_bytes,
            shape=(self.records, self.samples_per_record),
        )

    def read(self, label: str, *, physical: bool = True) -> np.ndarray:
        """One channel, in its physical unit unless asked for raw integers."""
        found = self.channel(label)
        start, end = self._columns(label)
        raw = np.array(self._mapped()[:, start:end].reshape(-1), dtype=np.float64)
        if not physical:
            return raw
        return (raw - found.digital_min) * found.scale + found.physical_min

    # ── EDF+ annotations ─────────────────────────────────────────────────────

    def annotations(self) -> list[Annotation]:
        """Every time-stamped annotation, in file order.

        The first annotation of each record is the record's own timestamp and
        carries no text; it is skipped rather than returned as an event with an
        empty name.
        """
        labels = [c.label for c in self.channels if c.label == ANNOTATION_LABEL]
        if not labels:
            return []

        start, end = self._columns(ANNOTATION_LABEL)
        raw = np.ascontiguousarray(self._mapped()[:, start:end]).tobytes()
        found: list[Annotation] = []
        for block in raw.split(_PADDING):
            if not block.strip():
                continue
            found.extend(_parse_tal(block))
        return found


def _parse_tal(block: bytes) -> list[Annotation]:
    """One time-stamped annotation list: an onset, a duration, and its texts."""
    parts = block.split(_TAL_END)
    if not parts:
        return []

    head = parts[0]
    if _ONSET_END in head:
        onset_raw, duration_raw = head.split(_ONSET_END, 1)
    else:
        onset_raw, duration_raw = head, b""

    try:
        onset = float(onset_raw.decode("ascii", errors="replace"))
    except ValueError:
        return []
    try:
        duration = float(duration_raw.decode("ascii", errors="replace"))
    except ValueError:
        duration = 0.0

    made = []
    for text in parts[1:]:
        name = text.decode("ascii", errors="replace").strip()
        if name:
            made.append(Annotation(onset, duration, name))
    return made
