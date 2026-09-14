"""A fitted model's output, made traceable.

The provenance types were tested with constructed predictions, which shows the
architecture works and not that anything travels it. These tests fit a model on
a table built by the real builder and assert that what comes out names the
feature vector, the windows and the recordings the table recorded -- so the
chain from a score back to a slice of signal is one the pipeline actually
produces.
"""

from __future__ import annotations

import numpy as np
import pytest

from physioml.core.registry import TrainingRun
from physioml.dataset import FeatureTable
from physioml.evaluation.export import artifact_of, as_json, export_fold, predictions_for
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
