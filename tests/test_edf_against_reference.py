"""The EDF reader against an independent implementation, on real files.

Everything else testing this reader constructs its own files. That is worth
doing and it is not sufficient: a reader and a writer built by one person agree
with each other about a format neither of them may have got right. The only
evidence that these files are being read correctly is agreement with somebody
else's reader on the actual recordings the results come from.

pyedflib is a wrapper around the reference C library. It is a test dependency,
never a runtime one, and these tests skip when it or the recordings are absent.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from physioml.io.edf import ANNOTATION_LABEL, EDF
from tests.paths import SLEEP_EDF_DIR as SLEEP_EDF
from tests.paths import SLEEP_MISSING

pyedflib = pytest.importorskip("pyedflib", reason="pyedflib is not installed")

RECORDINGS = sorted(SLEEP_EDF.glob("*PSG.edf")) if SLEEP_EDF.is_dir() else []
HYPNOGRAMS = sorted(SLEEP_EDF.glob("*Hypnogram.edf")) if SLEEP_EDF.is_dir() else []

needs_files = pytest.mark.skipif(not RECORDINGS, reason=SLEEP_MISSING)


@pytest.fixture
def reference():
    """A pyedflib reader, closed however the test ends."""
    open_readers = []

    def make(path: Path):
        found = pyedflib.EdfReader(str(path))
        open_readers.append(found)
        return found

    yield make
    for found in open_readers:
        found.close()


@needs_files
@pytest.mark.parametrize("path", RECORDINGS[:3], ids=lambda p: p.name)
def test_the_same_number_of_records_is_read(path, reference):
    assert EDF(path).records == reference(path).datarecords_in_file


@needs_files
@pytest.mark.parametrize("path", RECORDINGS[:3], ids=lambda p: p.name)
def test_every_channel_has_the_same_length_and_rate(path, reference):
    """EDF allows a different sample count per channel within one record, and
    these files use it: the electroencephalogram is 100 Hz and the respiration
    belt is 1 Hz in the same file. Getting the column arithmetic wrong returns
    another channel's samples at a plausible length."""
    mine, theirs = EDF(path), reference(path)
    for index, label in enumerate(theirs.getSignalLabels()):
        if label == ANNOTATION_LABEL:
            continue
        assert mine.read(label).size == theirs.readSignal(index).size, label
        assert mine.sampling_rate(label) == pytest.approx(
            theirs.getSampleFrequency(index)
        ), label


@needs_files
@pytest.mark.parametrize("path", RECORDINGS[:3], ids=lambda p: p.name)
def test_the_physical_values_agree(path, reference):
    """Digital-to-physical scaling, which is where a reader silently invents
    microvolts. Agreement is to floating-point noise, not to a tolerance."""
    mine, theirs = EDF(path), reference(path)
    for index, label in enumerate(theirs.getSignalLabels()):
        if label == ANNOTATION_LABEL:
            continue
        got = mine.read(label)
        want = theirs.readSignal(index)
        assert np.max(np.abs(got - want)) < 1e-9, label


@needs_files
def test_a_channel_read_by_name_is_the_channel_the_reference_reads_by_index():
    """The two readers address channels differently, so this checks the map
    rather than the arithmetic: reading 'EEG Pz-Oz' must not return Fpz-Cz."""
    path = RECORDINGS[0]
    mine = EDF(path)
    theirs = pyedflib.EdfReader(str(path))
    try:
        labels = theirs.getSignalLabels()
        first = labels.index("EEG Fpz-Cz")
        second = labels.index("EEG Pz-Oz")
        assert np.max(np.abs(mine.read("EEG Fpz-Cz") - theirs.readSignal(first))) < 1e-9
        assert np.max(np.abs(mine.read("EEG Pz-Oz") - theirs.readSignal(second))) < 1e-9
        # And the two are genuinely different signals, so the check has teeth.
        assert np.max(np.abs(theirs.readSignal(first) - theirs.readSignal(second))) > 1.0
    finally:
        theirs.close()


@pytest.mark.skipif(not HYPNOGRAMS, reason=SLEEP_MISSING)
@pytest.mark.parametrize("path", HYPNOGRAMS[:3], ids=lambda p: p.name)
def test_annotations_agree_in_count_onset_duration_and_text(path, reference):
    """The EDF+ time-stamped annotation list, which the whole sleep result
    rests on: every stage label and every boundary comes from here."""
    mine = EDF(path).annotations()
    onsets, durations, texts = reference(path).readAnnotations()

    assert len(mine) == len(texts)
    for found, onset, duration, text in zip(mine, onsets, durations, texts, strict=True):
        assert found.onset_seconds == pytest.approx(onset)
        assert found.duration_seconds == pytest.approx(duration)
        assert found.text == text


@pytest.mark.skipif(not HYPNOGRAMS, reason=SLEEP_MISSING)
def test_the_scored_interval_covers_the_whole_recording():
    """A gap would mean epochs silently unscored rather than marked."""
    for path in HYPNOGRAMS[:3]:
        found = EDF(path).annotations()
        edges = [(a.onset_seconds, a.onset_seconds + a.duration_seconds) for a in found]
        for (_, ends), (starts, _) in itertools.pairwise(edges):
            assert starts == pytest.approx(ends), f"gap in {path.name}"
