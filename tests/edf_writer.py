"""A literal EDF writer, for tests that need files with known contents.

Deliberately literal about the format -- fixed-width ASCII fields, then
interleaved 16-bit records -- and it shares no code with the reader: a pair
built from the same helper would agree with each other about a format neither
had got right.

**Not in conftest.py.** pytest imports conftest for every run, including the
continuous-integration job that installs no scientific stack in order to prove
the provenance core needs none. Putting a numpy import there broke exactly the
guarantee that job exists to assert, and it broke it silently everywhere else,
because every other job has numpy installed.
"""

from __future__ import annotations

import numpy as np

from physioml.io.edf import ANNOTATION_LABEL


def field(text: str, width: int) -> bytes:
    return f"{text:<{width}}".encode("ascii")[:width]


def write_edf(
    path,
    signals: dict[str, np.ndarray],
    *,
    rates: dict[str, float] | None = None,
    record_seconds: float = 1.0,
    physical: tuple[float, float] = (-100.0, 100.0),
    digital: tuple[int, int] = (-2048, 2047),
    started: str = "01.01.89",
    at: str = "23.30.00",
    reserved: str = "",
    declared_records: int | None = None,
    annotations: bytes | None = None,
) -> None:
    """A minimal EDF/EDF+ writer, for producing files with known contents."""
    rates = rates or dict.fromkeys(signals, 1.0)
    per_record = {n: int(rates[n] * record_seconds) for n in signals}
    records = min(len(v) // per_record[n] for n, v in signals.items())

    labels = list(signals)
    if annotations is not None:
        labels.append(ANNOTATION_LABEL)
        per_record[ANNOTATION_LABEL] = max(
            1, (len(annotations) + records - 1) // records // 2 + 1
        )

    count = len(labels)
    head = b"".join(
        [
            field("0", 8),
            field("X X X X", 80),
            field("Startdate", 80),
            field(started, 8),
            field(at, 8),
            field(str(256 + 256 * count), 8),
            field(reserved, 44),
            field(str(records if declared_records is None else declared_records), 8),
            field(str(record_seconds), 8),
            field(str(count), 4),
        ]
    )

    def per_signal(make) -> bytes:
        return b"".join(make(name) for name in labels)

    head += per_signal(lambda n: field(n, 16))
    head += per_signal(lambda n: field("transducer", 80))
    head += per_signal(lambda n: field("uV" if n != ANNOTATION_LABEL else "", 8))
    head += per_signal(lambda n: field(str(physical[0]), 8))
    head += per_signal(lambda n: field(str(physical[1]), 8))
    head += per_signal(lambda n: field(str(digital[0]), 8))
    head += per_signal(lambda n: field(str(digital[1]), 8))
    head += per_signal(lambda n: field("HP:0.1Hz", 80))
    head += per_signal(lambda n: field(str(per_record[n]), 8))
    head += per_signal(lambda n: field("", 32))

    span = digital[1] - digital[0]
    scale = (physical[1] - physical[0]) / span if span else 1.0
    body = bytearray()
    room = per_record.get(ANNOTATION_LABEL, 0) * 2
    for r in range(records):
        for name in labels:
            if name == ANNOTATION_LABEL:
                chunk = annotations[r * room : (r + 1) * room] if annotations else b""
                body += chunk.ljust(room, b"\x00")
                continue
            n = per_record[name]
            block = signals[name][r * n : (r + 1) * n]
            digitised = np.round((block - physical[0]) / scale + digital[0])
            body += np.asarray(digitised, dtype="<i2").tobytes()

    path.write_bytes(head + bytes(body))


def tal(onset: float, duration: float | None, *texts: str) -> bytes:
    """One time-stamped annotation list, in the shape EDF+ specifies."""
    head = f"{onset:+}".encode("ascii")
    if duration is not None:
        head += b"\x15" + str(duration).encode("ascii")
    for text in texts:
        head += b"\x14" + text.encode("ascii")
    return head + b"\x14\x00"
