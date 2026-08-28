"""The European Data Format reader.

Files are constructed here byte by byte, so every expected value is known
rather than recorded from a previous run of this same code. The writer below is
deliberately literal about the format -- fixed-width ASCII fields, then
interleaved 16-bit records -- because if it shared code with the reader the
pair would agree with each other about a format neither had got right.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from physioml.io.edf import ANNOTATION_LABEL, EDF, Annotation, EDFError


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


# ── the header ──────────────────────────────────────────────────────────────


def test_the_header_is_read(tmp_path):
    path = tmp_path / "one.edf"
    write_edf(path, {"EEG Fpz-Cz": np.zeros(300)}, rates={"EEG Fpz-Cz": 100.0})
    edf = EDF(path)

    assert edf.labels == ("EEG Fpz-Cz",)
    assert edf.sampling_rate("EEG Fpz-Cz") == pytest.approx(100.0)
    assert edf.records == 3
    assert edf.duration_seconds == pytest.approx(3.0)
    assert edf.channel("EEG Fpz-Cz").unit == "uV"


def test_a_two_digit_year_is_read_the_way_the_format_says(tmp_path):
    """85 to 99 are the 1900s, 00 to 84 the 2000s. Guessing otherwise puts a
    1989 sleep study in 2089."""
    path = tmp_path / "old.edf"
    write_edf(path, {"A": np.zeros(10)}, started="24.04.89", at="22.15.30")
    assert EDF(path).started == datetime(1989, 4, 24, 22, 15, 30)

    write_edf(path, {"A": np.zeros(10)}, started="24.04.05")
    assert EDF(path).started.year == 2005


def test_an_unreadable_date_is_none_rather_than_a_guess(tmp_path):
    path = tmp_path / "undated.edf"
    write_edf(path, {"A": np.zeros(10)}, started="ah.ah.ah")
    assert EDF(path).started is None


def test_a_file_too_short_to_hold_a_header_is_refused(tmp_path):
    path = tmp_path / "stub.edf"
    path.write_bytes(b"0" * 50)
    with pytest.raises(EDFError, match="too short"):
        EDF(path)


def test_a_truncated_file_is_refused_rather_than_read_short(tmp_path):
    """A half-downloaded recording should not silently become a short night."""
    path = tmp_path / "cut.edf"
    write_edf(path, {"A": np.arange(100, dtype=float)}, rates={"A": 10.0})
    whole = path.read_bytes()
    path.write_bytes(whole[: len(whole) - 40])
    with pytest.raises(EDFError, match="truncated"):
        EDF(path)


def test_a_file_that_does_not_know_its_own_length_is_measured(tmp_path):
    """EDF allows -1 for "not known"; the size on disk knows either way."""
    path = tmp_path / "open.edf"
    write_edf(path, {"A": np.zeros(50)}, rates={"A": 10.0}, declared_records=-1)
    assert EDF(path).records == 5


# ── the signals ─────────────────────────────────────────────────────────────


def test_a_signal_comes_back_in_its_physical_unit(tmp_path):
    path = tmp_path / "wave.edf"
    wanted = np.linspace(-80.0, 80.0, 400)
    write_edf(path, {"EEG": wanted}, rates={"EEG": 100.0})
    got = EDF(path).read("EEG")
    assert got.shape == wanted.shape
    # One digital step is 200/4095 uV, so agreement is to about 0.05 uV.
    assert np.abs(got - wanted).max() < 0.1


def test_raw_integers_are_available_unscaled(tmp_path):
    path = tmp_path / "raw.edf"
    write_edf(path, {"EEG": np.zeros(100)}, rates={"EEG": 100.0})
    raw = EDF(path).read("EEG", physical=False)
    # Zero microvolts sits at the middle of the digital range, not at zero.
    assert raw[0] == pytest.approx(-0.5, abs=1.0)


def test_channels_are_not_confused_with_each_other(tmp_path):
    """The records interleave every signal, so an off-by-one in the column
    arithmetic returns another channel's samples at the right length."""
    path = tmp_path / "many.edf"
    write_edf(
        path,
        {
            "EEG Fpz-Cz": np.full(200, 10.0),
            "EEG Pz-Oz": np.full(200, -20.0),
            "EMG submental": np.full(100, 40.0),
        },
        rates={"EEG Fpz-Cz": 100.0, "EEG Pz-Oz": 100.0, "EMG submental": 50.0},
        record_seconds=2.0,
    )
    edf = EDF(path)
    assert np.abs(edf.read("EEG Fpz-Cz") - 10.0).max() < 0.1
    assert np.abs(edf.read("EEG Pz-Oz") + 20.0).max() < 0.1
    assert np.abs(edf.read("EMG submental") - 40.0).max() < 0.1
    assert edf.read("EMG submental").size == 100


def test_channels_may_have_different_rates(tmp_path):
    path = tmp_path / "mixed.edf"
    write_edf(
        path,
        {"EEG": np.zeros(200), "Temp": np.zeros(2)},
        rates={"EEG": 100.0, "Temp": 1.0},
        record_seconds=1.0,
    )
    edf = EDF(path)
    assert edf.sampling_rate("EEG") == pytest.approx(100.0)
    assert edf.sampling_rate("Temp") == pytest.approx(1.0)
    assert edf.read("EEG").size == 200
    assert edf.read("Temp").size == 2


def test_an_absent_channel_says_what_is_there(tmp_path):
    path = tmp_path / "one.edf"
    write_edf(path, {"EEG Fpz-Cz": np.zeros(10)})
    with pytest.raises(KeyError, match="EEG Pz-Oz"):
        EDF(path).read("EEG Pz-Oz")


def test_a_channel_with_no_digital_range_is_not_divided_by_zero(tmp_path):
    path = tmp_path / "flat.edf"
    write_edf(path, {"A": np.zeros(10)}, digital=(0, 0), physical=(0.0, 0.0))
    assert np.isfinite(EDF(path).read("A")).all()


# ── EDF+ annotations ────────────────────────────────────────────────────────


def test_annotations_are_read_with_their_onsets_and_durations(tmp_path):
    path = tmp_path / "hypnogram.edf"
    body = (
        tal(0.0, 30.0, "Sleep stage W")
        + tal(30.0, 60.0, "Sleep stage 1")
        + tal(90.0, 120.0, "Sleep stage 2")
    )
    write_edf(
        path,
        {"A": np.zeros(3)},
        record_seconds=1.0,
        reserved="EDF+C",
        annotations=body,
    )
    found = EDF(path).annotations()
    assert found == [
        Annotation(0.0, 30.0, "Sleep stage W"),
        Annotation(30.0, 60.0, "Sleep stage 1"),
        Annotation(90.0, 120.0, "Sleep stage 2"),
    ]


def test_a_timestamp_carrying_no_text_is_not_an_event(tmp_path):
    """The first annotation of each record is the record's own clock."""
    path = tmp_path / "stamped.edf"
    body = tal(0.0, None) + tal(0.0, 30.0, "Sleep stage W")
    write_edf(path, {"A": np.zeros(2)}, reserved="EDF+C", annotations=body)
    found = EDF(path).annotations()
    assert [a.text for a in found] == ["Sleep stage W"]


def test_a_file_without_an_annotation_channel_has_no_annotations(tmp_path):
    path = tmp_path / "plain.edf"
    write_edf(path, {"EEG": np.zeros(100)}, rates={"EEG": 100.0})
    assert EDF(path).annotations() == []
