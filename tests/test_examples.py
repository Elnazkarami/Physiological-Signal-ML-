"""The public entry point, held to the output it publishes.

``examples/quickstart.py`` is what someone runs who has neither dataset, and
``examples/expected/`` is its committed output. A published expected output
that no longer matches the code is worse than none: it reads as evidence and
is a record of what the pipeline used to do.

What "matches" means here needs care, and the care is the interesting part.
The identifiers in the chain are content hashes of feature values, and feature
values are floating-point results of filtering and spectral estimation. Two
machines running different builds of NumPy, SciPy or a BLAS agree on those to
about eleven significant figures and disagree in the last bits -- measured,
not supposed: this suite's own CI produced ``2.2163459163476735e-05`` where a
developer machine produced ``2.2163459163323433e-05``. A hash has no notion of
nearly, so the identifiers differ.

That is a true property of hashing floating-point arithmetic, and the tests
say so rather than hiding it:

* within one environment the run is byte-for-byte identical, which is the
  guarantee the design actually offers and the one provenance depends on;
* against the committed output, the structure must match exactly and the
  numbers to a tolerance, with the identifiers masked.

This reaches further than it first appears. A recording identifier covers a
sha256 of its samples, so it moves with them; a window identifier covers its
recording, so it moves too; and a feature vector and a prediction cover both.
*Every* identifier in the chain is therefore measurement-dependent, which is
the price of having them fingerprint the signal rather than only describe it.
The records behind them are compared field by field with the identifiers
masked, which is what actually establishes that the chain has not changed.

The example is also the only place the whole chain runs in one process, so a
break anywhere between the recording and the prediction shows up here even
when every unit test still passes.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parent.parent
QUICKSTART = ROOT / "examples" / "quickstart.py"
EXPECTED = ROOT / "examples" / "expected"
REGENERATE = (
    "regenerate with: python examples/quickstart.py "
    "--write-trace examples/expected/prediction_trace.json "
    "> examples/expected/quickstart_output.txt"
)

#: Every identifier in the chain. All of them hash measured values somewhere in
#: their ancestry -- a recording covers a digest of its samples, and everything
#: downstream covers the recording -- so all of them move with the last bits of
#: the arithmetic.
MEASURED_ID = re.compile(r"\b(pred|fvec|model|trun|win|rec)-[0-9a-f]{32}\b")

#: Any number, so that text can be compared with the arithmetic held apart.
NUMBER = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?|-?\d+")

TOLERANCE = 1e-6


def run(trace: Path | None = None) -> tuple[str, dict[str, Any] | None]:
    """One run of the example, as a visitor would run it."""
    command = [sys.executable, str(QUICKSTART)]
    if trace is not None:
        command += ["--write-trace", str(trace)]
    finished = subprocess.run(
        command, capture_output=True, text=True, cwd=ROOT, timeout=600
    )
    assert finished.returncode == 0, finished.stderr
    return finished.stdout, json.loads(trace.read_text()) if trace else None


@pytest.fixture(scope="module")
def produced(tmp_path_factory) -> tuple[str, dict[str, Any]]:
    trace = tmp_path_factory.mktemp("trace") / "prediction_trace.json"
    output, parsed = run(trace)
    assert parsed is not None
    return output, parsed


def shape_of(text: str) -> str:
    """The text with identifiers masked and numbers removed."""
    return NUMBER.sub("#", MEASURED_ID.sub(r"\1-<id>", text))


def numbers_in(text: str) -> list[float]:
    return [float(n) for n in NUMBER.findall(MEASURED_ID.sub("", text))]


# ── the guarantee that actually holds ───────────────────────────────────────


def test_the_run_is_reproducible_in_one_environment():
    """Two runs on one machine are identical, hashes included.

    This is what the provenance claim rests on: the same signal through the
    same feature set and the same model yields the same identifier. If this
    fails, something in the pipeline is reading uninitialised memory, a dict
    order, or a clock.
    """
    first, _ = run()
    second, _ = run()
    assert first == second


# ── and the one that holds to a tolerance ───────────────────────────────────


def test_the_committed_output_still_has_the_same_shape(produced):
    output, _trace = produced
    committed = (EXPECTED / "quickstart_output.txt").read_text()
    assert shape_of(output) == shape_of(committed), (
        f"the example's output has changed structurally since it was "
        f"published. {REGENERATE}"
    )


def test_the_committed_output_still_has_the_same_numbers(produced):
    output, _trace = produced
    committed = (EXPECTED / "quickstart_output.txt").read_text()
    assert numbers_in(output) == pytest.approx(
        numbers_in(committed), rel=TOLERANCE, abs=TOLERANCE
    ), f"the example's numbers have moved since it was published. {REGENERATE}"


def test_the_committed_trace_still_matches(produced):
    """Field by field, with the identifiers masked.

    Comparing identifiers would only re-assert that two machines disagree about
    floating point. What has to hold is that the same records are present,
    describing the same windows of the same recordings, with the same
    measurements in them to a tolerance.
    """
    _output, trace = produced
    committed = json.loads((EXPECTED / "prediction_trace.json").read_text())

    assert len(trace["predictions"]) == len(committed["predictions"]), REGENERATE
    for mine, published in zip(trace["predictions"], committed["predictions"], strict=True):
        for field in (
            "subject_id",
            "window_start",
            "window_end",
            "predicted_class",
            "source_fact_ids",
        ):
            assert mine[field] == published[field], f"{field} changed. {REGENERATE}"
        for field in ("window_start_seconds", "window_seconds"):
            # Construction constants rather than measurements, so these should
            # be exact -- compared loosely anyway, because being strict about a
            # float buys nothing and costs a false failure.
            assert mine[field] == pytest.approx(published[field]), (
                f"{field} changed. {REGENERATE}"
            )
        for field in ("feature_ids", "source_window_ids"):
            assert len(mine[field]) == len(published[field]), (
                f"{field} changed. {REGENERATE}"
            )
        assert mine["probability"] == pytest.approx(
            published["probability"], rel=TOLERANCE, abs=TOLERANCE
        ), REGENERATE

    assert (
        trace["artifact"]["expected_features"]
        == (committed["artifact"]["expected_features"])
    ), REGENERATE
    for field in ("model_name", "model_version", "task", "feature_schema_version"):
        assert trace["artifact"][field] == committed["artifact"][field], REGENERATE

    records, published_records = trace["records"], committed["records"]
    for kind in records:
        assert len(records[kind]) == len(published_records[kind]), (
            f"the number of {kind} records changed. {REGENERATE}"
        )

    # Windows describe a slice of signal. Compared as a set of descriptions,
    # since their identifiers no longer survive a change of machine.
    def described(window: dict[str, Any]) -> tuple:
        return (
            window["subject_id"],
            window["start_sample"],
            window["end_sample"],
            window["start_time"],
            window["sampling_rate_hz"],
            tuple(window["channel_ids"]),
            window["qc_status"],
            window["label"],
        )

    assert sorted(map(described, records["windows"].values())) == sorted(
        map(described, published_records["windows"].values())
    ), f"the windows behind the predictions changed. {REGENERATE}"

    def sourced(recording: dict[str, Any]) -> tuple:
        return (
            recording["study_id"],
            recording["subject_id"],
            recording["modality"],
            recording["sampling_rate_hz"],
            recording["start_time"],
            recording["duration_seconds"],
            tuple(recording["channels"]),
            recording["device_name"],
            recording["source_uri"],
            # Not source_hash: it is a digest of the samples, and the samples
            # differ in their last bits between machines. That it is present
            # and the right length is asserted separately.
        )

    assert sorted(map(sourced, records["recordings"].values())) == sorted(
        map(sourced, published_records["recordings"].values())
    ), f"the recordings behind the windows changed. {REGENERATE}"

    for mine, published in zip(
        records["feature_vectors"].values(),
        published_records["feature_vectors"].values(),
        strict=True,
    ):
        assert mine["names"] == published["names"], REGENERATE
        assert mine["subject_id"] == published["subject_id"], REGENERATE
        assert mine["values"] == pytest.approx(
            published["values"], rel=TOLERANCE, abs=TOLERANCE
        ), REGENERATE

    one = next(iter(records["training_runs"].values()))
    other = next(iter(published_records["training_runs"].values()))
    for field in ("task", "split_strategy", "train_subjects", "test_subjects"):
        assert one[field] == other[field], f"{field} changed. {REGENERATE}"


# ── what the published chain has to contain ─────────────────────────────────


def test_the_example_runs(produced):
    output, _trace = produced
    assert "[5/5] exporting one fold as traceable predictions" in output


def test_every_reference_in_the_published_chain_resolves(produced):
    """The published file contains the chain, it does not merely name it.

    Checking that an identifier starts with ``win-`` says it is the right
    shape, not that anything is behind it. This follows every reference --
    prediction to feature vector, vector to windows, window to recording,
    artifact to training run -- and fails on the first one that leads nowhere.
    """
    from physioml.evaluation.export import unresolved

    _output, trace = produced
    assert unresolved(trace) == []
    records = trace["records"]
    assert records["feature_vectors"] and records["windows"]
    assert records["recordings"] and records["training_runs"]


def test_the_published_timestamps_are_when_the_signal_was_recorded(produced):
    """Not the Unix epoch, which is what they used to be.

    The example's recordings begin at 09:00 on 2026-01-01, and the export
    counted the within-recording offset from 1970 instead of reading the
    recording's own start time -- so every prediction this project emitted was
    dated 1970 however the signal was actually recorded.
    """
    from datetime import datetime

    _output, trace = produced
    for prediction in trace["predictions"]:
        start = datetime.fromisoformat(prediction["window_start"])
        assert start.year == 2026, "the recording's start time was discarded"
        # And the offset within the recording is carried separately, because it
        # is what the splits are made on, and it stays true for a dataset that
        # has no acquisition timestamps at all.
        assert prediction["window_start_seconds"] >= 0.0
        assert prediction["window_seconds"] == 60.0


def test_a_published_recording_identifies_its_own_samples(produced):
    """A recording identifier otherwise covers metadata that can stay the same
    while the signal changes."""
    _output, trace = produced
    for recording in trace["records"]["recordings"].values():
        assert len(recording["source_hash"]) == 64, (
            "the recording does not fingerprint its samples, so a corrected "
            "export of the same session would carry the same identifier"
        )


def test_the_published_artifact_identifies_its_fitted_parameters(produced):
    """Otherwise two models fitted by the same run on the same columns are the
    same artifact however different their coefficients."""
    _output, trace = produced
    assert len(trace["artifact"]["artifact_hash"]) == 64
    assert trace["artifact"]["model_id"].startswith("model-")


def test_every_published_prediction_names_what_produced_it(produced):
    _output, trace = produced
    assert trace["artifact"]["model_id"], "the artifact must name its own identity"
    assert trace["predictions"], "the trace is empty"
    for prediction in trace["predictions"]:
        assert prediction["prediction_id"].startswith("pred-")
        assert len(prediction["feature_ids"]) == 1
        assert prediction["feature_ids"][0].startswith("fvec-")
        # Four wrist sensors, so four windows. One would mean the row was
        # traceable to a single signal and silent about the other three.
        assert len(prediction["source_window_ids"]) == 4
        assert all(w.startswith("win-") for w in prediction["source_window_ids"])
        assert prediction["source_fact_ids"] == []


def test_a_prediction_identity_is_not_shared_between_rows(produced):
    """Two rows scored differently must not land on one identifier."""
    _output, trace = produced
    identities = {p["prediction_id"] for p in trace["predictions"]}
    assert len(identities) == len(trace["predictions"])


def test_the_published_window_is_the_one_that_was_measured(produced):
    """Sixty seconds, because that is what the example windows at.

    The export used to assume thirty regardless, so every peripheral prediction
    named an interval ending halfway through the signal it was computed from.
    """
    from datetime import datetime

    _output, trace = produced
    for prediction in trace["predictions"]:
        start = datetime.fromisoformat(prediction["window_start"])
        end = datetime.fromisoformat(prediction["window_end"])
        assert (end - start).total_seconds() == 60.0


def test_the_example_does_not_reach_for_cdfs():
    """Nothing in the published demo touches the external provenance engine.

    The README says the library runs standalone. This is the claim tested
    rather than asserted: the example imports the pipeline, the evaluation and
    the export, and if any of them pulled CDFS in, the module would be loaded
    by the end of the run.
    """
    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; "
            "sys.argv = ['quickstart']; "
            f"runpy.run_path({str(QUICKSTART)!r}, run_name='__main__'); "
            "print('CDFS_LOADED' if any("
            "m == 'cdfs' or m.startswith('cdfs.') for m in sys.modules) "
            "else 'CDFS_ABSENT', file=sys.stderr)",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=600,
    )
    assert loaded.returncode == 0, loaded.stderr
    assert "CDFS_ABSENT" in loaded.stderr, loaded.stderr


def test_the_scores_are_labelled_synthetic(produced):
    """A reader who sees only the output must not mistake it for a result.

    Printed at the top, beside the scores themselves, and again at the end,
    because terminal output gets screenshotted from the middle.
    """
    output, _trace = produced
    # Wrapped for the terminal, so compared with the wrapping collapsed.
    flowed = " ".join(output.split())
    assert "THE SCORES THIS PRINTS ARE MEANINGLESS" in flowed
    assert "describe the generator, not a result" in flowed
    assert "Synthetic data. The scores above are a property of this script." in flowed
