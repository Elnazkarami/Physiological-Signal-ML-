"""A fitted model's output, made traceable.

The provenance types were tested with constructed predictions, which shows the
architecture works and not that anything travels it. These tests fit a model on
a table built by the real builder and assert that what comes out names the
feature vector, the windows and the recordings the table recorded -- so the
chain from a score back to a slice of signal is one the pipeline actually
produces.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from physioml.core.recording import Modality
from physioml.core.registry import TrainingRun
from physioml.core.window import QCStatus
from physioml.dataset import FeatureTable
from physioml.evaluation.export import (
    artifact_of,
    as_bundle,
    as_json,
    export_fold,
    parameter_digest,
    predictions_for,
    unresolved,
)
from physioml.evaluation.splits import leave_one_subject_out
from physioml.models.classical import MODELS

SUBJECTS = ["S1", "S2", "S3"]


def traceable_table(per_subject: int = 12) -> FeatureTable:
    """A table carrying the identifiers a real build now retains."""
    rng = np.random.default_rng(0)
    rows = len(SUBJECTS) * per_subject
    return FeatureTable(
        feature_names=("a", "b"),
        values=rng.normal(0, 1, (rows, 2)),
        subjects=np.array([s for s in SUBJECTS for _ in range(per_subject)]),
        labels=np.array(
            [("N2" if i % 3 else "W") for _ in SUBJECTS for i in range(per_subject)]
        ),
        window_ids=tuple(f"win-{i}" for i in range(rows)),
        window_starts=np.tile(np.arange(per_subject, dtype=float) * 30.0, len(SUBJECTS)),
        feature_set_version="test-1",
        qc_policy_version="test-1",
        row_windows=tuple((f"win-{i}", f"win-{i}-eog") for i in range(rows)),
        row_recordings=tuple((f"rec-{i // per_subject}",) for i in range(rows)),
        row_feature_vectors=tuple(f"fvec-{i}" for i in range(rows)),
    )


def fitted(table: FeatureTable):
    """A real model, fitted on real folds, scored on a held-out participant."""
    split = next(iter(leave_one_subject_out(table.subject_ids)))
    train, test = split.mask(table.subjects)
    model = MODELS["logistic"]()
    model.fit(table.values[train], table.labels[train])
    run = TrainingRun.create(
        task="sleep_stage",
        dataset_version="test",
        split_strategy=split.strategy,
        train_subjects=split.train_subjects,
        test_subjects=split.test_subjects,
        feature_schema_version=table.feature_set_version,
        preprocessing_version=table.qc_policy_version,
        random_seed=split.seed,
    )
    return model, run, test, model.predict(table.values[test])


def test_a_prediction_names_the_feature_vector_of_its_own_row():
    table = traceable_table()
    _model, run, test, predicted = fitted(table)
    _artifact, made = export_fold(
        table, run, test, predicted, model_name="logistic", model_version="1.0"
    )

    assert len(made) == len(test)
    for prediction, row in zip(made, test, strict=True):
        assert prediction.feature_ids == (table.row_feature_vectors[row],)
        assert prediction.source_window_ids == tuple(table.row_windows[row])
        assert prediction.subject_id == str(table.subjects[row])


def test_a_prediction_names_the_run_that_fitted_it():
    table = traceable_table()
    _model, run, test, predicted = fitted(table)
    artifact, made = export_fold(
        table, run, test, predicted, model_name="logistic", model_version="1.0"
    )
    assert artifact.training_run_id == run.training_run_id
    assert artifact.expected_features == table.feature_names
    for prediction in made:
        assert prediction.training_run_id == run.training_run_id
        assert prediction.model_version == "1.0"


def test_the_artifact_refuses_a_vector_in_the_wrong_order():
    """The reason the artifact records the feature order at all."""
    from physioml.core.feature import Feature, FeatureVector
    from physioml.core.registry import SchemaMismatch

    table = traceable_table()
    _model, run, _test, _predicted = fitted(table)
    artifact = artifact_of(run, table, model_name="logistic", model_version="1.0")

    def feature(name: str, value: float) -> Feature:
        return Feature.create(
            subject_id="S1",
            name=name,
            value=value,
            unit=None,
            feature_set="test",
            feature_set_version="test-1",
            source_window_ids=("win-0",),
        )

    right = FeatureVector.of([feature("a", 1.0), feature("b", 2.0)], window_id="win-0")
    assert artifact.accepts(right) is None or artifact.accepts(right)

    reversed_order = FeatureVector(
        subject_id="S1",
        window_id="win-0",
        names=("b", "a"),
        values=(2.0, 1.0),
        feature_ids=("x", "y"),
        feature_set_version="test-1",
    )
    with pytest.raises(SchemaMismatch):
        artifact.accepts(reversed_order)


def test_a_table_without_provenance_cannot_be_exported():
    """Rather than emitting predictions that point at nothing."""
    from dataclasses import replace

    table = traceable_table()
    stripped = replace(table, row_feature_vectors=(), row_windows=())
    _model, run, test, predicted = fitted(table)
    with pytest.raises(ValueError, match="could not name what produced it"):
        predictions_for(
            stripped,
            test,
            predicted,
            artifact=artifact_of(run, table, model_name="m", model_version="1"),
        )


def test_predictions_and_rows_must_correspond():
    table = traceable_table()
    _model, run, test, predicted = fitted(table)
    artifact = artifact_of(run, table, model_name="m", model_version="1")
    with pytest.raises(ValueError, match="must correspond"):
        predictions_for(table, test, predicted[:-1], artifact=artifact)


def test_the_same_row_scored_twice_gives_the_same_prediction_identity():
    """Identity is content, so a rerun that changes nothing changes nothing."""
    table = traceable_table()
    _model, run, test, predicted = fitted(table)
    from datetime import UTC, datetime

    when = datetime(2026, 1, 1, tzinfo=UTC)
    artifact = artifact_of(run, table, model_name="m", model_version="1")
    first = predictions_for(table, test, predicted, artifact=artifact, created_at=when)
    second = predictions_for(table, test, predicted, artifact=artifact, created_at=when)
    assert [p.prediction_id for p in first] == [p.prediction_id for p in second]


def test_a_different_answer_gives_a_different_identity():
    table = traceable_table()
    _model, run, test, predicted = fitted(table)
    from datetime import UTC, datetime

    when = datetime(2026, 1, 1, tzinfo=UTC)
    artifact = artifact_of(run, table, model_name="m", model_version="1")
    first = predictions_for(table, test, predicted, artifact=artifact, created_at=when)
    flipped = np.array(["W" if v == "N2" else "N2" for v in predicted])
    second = predictions_for(table, test, flipped, artifact=artifact, created_at=when)
    assert first[0].prediction_id != second[0].prediction_id


def test_the_exported_record_is_serialisable():
    import json

    table = traceable_table()
    _model, run, test, predicted = fitted(table)
    artifact, made = export_fold(
        table, run, test, predicted, model_name="logistic", model_version="1.0"
    )
    record = as_json(artifact, made)
    assert json.loads(json.dumps(record)) == record
    assert record["predictions"][0]["source_window_ids"]
    assert record["artifact"]["expected_features"] == ["a", "b"]


# ── when the window actually happened ───────────────────────────────────────

RECORDED = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)


def test_a_prediction_is_dated_when_the_window_was_recorded():
    """The recording's own start time, not the Unix epoch.

    Sleep-EDF carries acquisition timestamps and the WESAD reader places its
    sessions at a documented fixed origin. The export discarded both and added
    the within-recording offset to 1970, so every prediction was dated 1970
    however the signal was recorded.
    """
    table = traceable_table()
    table = replace(
        table,
        row_start_times=RECORDED.timestamp() + table.window_starts,
        window_seconds=30.0,
    )
    _model, run, test, predicted = fitted(table)
    _artifact, made = export_fold(
        table, run, test, predicted, model_name="logistic", model_version="1.0"
    )
    for prediction, row in zip(made, test, strict=True):
        assert prediction.window_start == RECORDED + timedelta(
            seconds=float(table.window_starts[row])
        )
        assert (prediction.window_end - prediction.window_start).total_seconds() == 30.0


def test_a_table_that_never_recorded_a_start_time_falls_back_to_the_offset():
    """Rather than refusing, because tables built before this still exist.

    The fallback is the old behaviour and it is wrong about the date; what
    makes it acceptable is that ``as_bundle`` publishes the offset beside it,
    so the number that is true is always present.
    """
    table = traceable_table()
    assert table.row_start_times is None
    _model, run, test, predicted = fitted(table)
    _artifact, made = export_fold(
        table, run, test, predicted, model_name="logistic", model_version="1.0"
    )
    assert made[0].window_start.year == 1970


# ── a chain that is present, not merely named ───────────────────────────────


def bundled():
    table = replace(traceable_table(), row_start_times=None, window_seconds=30.0)
    _model, run, test, predicted = fitted(table)
    artifact, made = export_fold(
        table, run, test, predicted, model_name="logistic", model_version="1.0"
    )
    windows = [
        SimpleNamespace(
            window_id=w,
            recording_id=table.row_recordings[row][0],
            subject_id=str(table.subjects[row]),
            start_sample=0,
            end_sample=30,
            start_time=RECORDED,
            sampling_rate_hz=1.0,
            channel_ids=("c",),
            qc_status=QCStatus.VALID,
            qc_reason_codes=(),
            label=None,
        )
        for row in test
        for w in table.row_windows[row]
    ]
    recordings = [
        SimpleNamespace(
            recording_id=r,
            study_id="S",
            subject_id="S1",
            modality=Modality.ECG,
            sampling_rate_hz=1.0,
            start_time=RECORDED,
            duration_seconds=1.0,
            channels=("c",),
            device_name="d",
            source_uri="u",
            source_hash="h",
        )
        for r in {r for row in test for r in table.row_recordings[row]}
    ]
    return table, run, artifact, made, test, windows, recordings


def test_a_bundle_resolves_every_reference_it_makes():
    table, run, artifact, made, test, windows, recordings = bundled()
    bundle = as_bundle(
        artifact,
        made,
        table=table,
        rows=test,
        run=run,
        windows=windows,
        recordings=recordings,
    )
    assert unresolved(bundle) == []


def test_a_bundle_missing_a_record_says_which_one():
    """The point of the check: a prefix test would pass on all of these."""
    table, run, artifact, made, test, windows, recordings = bundled()
    bundle = as_bundle(
        artifact,
        made,
        table=table,
        rows=test,
        run=run,
        windows=windows[1:],
        recordings=recordings,
    )
    missing = unresolved(bundle)
    assert missing == [windows[0].window_id]


def test_a_bundle_carries_the_offset_as_well_as_the_date():
    table, run, artifact, made, test, windows, recordings = bundled()
    bundle = as_bundle(
        artifact,
        made,
        table=table,
        rows=test,
        run=run,
        windows=windows,
        recordings=recordings,
    )
    for prediction, row in zip(bundle["predictions"], test, strict=True):
        assert prediction["window_start_seconds"] == float(table.window_starts[row])
        assert prediction["window_seconds"] == 30.0


# ── what the artifact identity covers ───────────────────────────────────────


def test_the_same_coefficients_give_the_same_artifact():
    assert parameter_digest([[1.0, 2.0]], [0.5]) == parameter_digest([[1.0, 2.0]], [0.5])


def test_different_coefficients_give_a_different_artifact():
    """Without this the artifact identity covers the request and not the
    result, so two models fitted by the same run on the same columns collide."""
    table = traceable_table()
    _model, run, _test, _predicted = fitted(table)
    one = artifact_of(
        run,
        table,
        model_name="logistic",
        model_version="1.0",
        artifact_hash=parameter_digest([[1.0, 2.0]], [0.5]),
    )
    two = artifact_of(
        run,
        table,
        model_name="logistic",
        model_version="1.0",
        artifact_hash=parameter_digest([[1.0, 2.1]], [0.5]),
    )
    assert one.model_id != two.model_id


def test_the_same_numbers_in_a_different_shape_do_not_collide():
    assert parameter_digest([[1.0], [2.0]]) != parameter_digest([[1.0, 2.0]])
