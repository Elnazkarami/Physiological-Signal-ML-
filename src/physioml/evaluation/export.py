"""Turning a fitted model and its scores into things that can be traced.

The evaluation produces numbers. Numbers are not traceable: a balanced accuracy
cannot say which windows it came from, what fitted it, or which observations
lie behind it. This turns the same run into the objects that can --
a :class:`~physioml.core.registry.ModelArtifact` for what was fitted, and a
:class:`~physioml.core.prediction.Prediction` per scored row carrying the
feature vector, the windows and the source facts it rests on.

Until this existed the provenance chain was implemented and tested and the
empirical pipelines did not travel it: they stopped at a feature table. Which
made the architecture real and the claim about the reported results false.

**Nothing here invents an identifier.** A prediction names the feature vector
the table recorded for that row, the windows those features were computed from,
and -- where the observations came through CDFS -- the facts behind them. A
table built before those were retained cannot be exported, and says so, rather
than emitting predictions that point at nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from physioml.core.prediction import Prediction
from physioml.core.registry import ModelArtifact, TrainingRun
from physioml.dataset import FeatureTable


def artifact_of(
    run: TrainingRun,
    table: FeatureTable,
    *,
    model_name: str,
    model_version: str,
) -> ModelArtifact:
    """What was fitted, recorded so a later prediction can name it.

    The expected features come from the table the model was fitted on, in the
    order it was fitted on them, which is what lets
    :meth:`~physioml.core.registry.ModelArtifact.accepts` refuse a vector whose
    columns are right and whose order is not.
    """
    return ModelArtifact.create(
        model_name=model_name,
        model_version=model_version,
        task=run.task,
        training_run_id=run.training_run_id,
        expected_features=tuple(table.feature_names),
        feature_schema_version=table.feature_set_version,
    )


def predictions_for(
    table: FeatureTable,
    rows: Sequence[int],
    predicted: np.ndarray,
    *,
    artifact: ModelArtifact,
    probability: np.ndarray | None = None,
    study_id: str = "PHYSIOML",
    epoch_seconds: float = 30.0,
    created_at: datetime | None = None,
) -> list[Prediction]:
    """One traceable prediction per scored row.

    ``rows`` are positions in ``table`` -- the test rows of a fold -- so the
    prediction for a row names that row's own feature vector and windows rather
    than anything reconstructed by position afterwards.
    """
    if not table.row_feature_vectors or not table.row_windows:
        raise ValueError(
            "this table does not carry per-row feature-vector and window "
            "identifiers, so a prediction made from it could not name what "
            "produced it; rebuild it with a current version of the builder"
        )
    if len(predicted) != len(rows):
        raise ValueError(
            f"{len(predicted)} predictions for {len(rows)} rows; they must correspond"
        )

    when = created_at or datetime.now(UTC)
    made: list[Prediction] = []
    for position, row in enumerate(rows):
        start = datetime.fromtimestamp(0, UTC) + timedelta(
            seconds=float(table.window_starts[row])
        )
        made.append(
            Prediction.create(
                study_id=study_id,
                subject_id=str(table.subjects[row]),
                task=artifact.task,
                window_start=start,
                window_end=start + timedelta(seconds=epoch_seconds),
                predicted_class=str(predicted[position]),
                probability=(
                    float(probability[position]) if probability is not None else None
                ),
                model_name=artifact.model_name,
                model_version=artifact.model_version,
                training_run_id=artifact.training_run_id,
                feature_set_version=table.feature_set_version,
                feature_ids=(table.row_feature_vectors[row],),
                source_window_ids=tuple(table.row_windows[row]),
                source_fact_ids=(
                    tuple(table.row_source_facts[row]) if table.row_source_facts else ()
                ),
                created_at=when,
            )
        )
    return made


def export_fold(
    table: FeatureTable,
    run: TrainingRun,
    rows: Sequence[int],
    predicted: np.ndarray,
    *,
    model_name: str,
    model_version: str,
    probability: np.ndarray | None = None,
    study_id: str = "PHYSIOML",
) -> tuple[ModelArtifact, list[Prediction]]:
    """The artifact and the predictions for one fold, together."""
    artifact = artifact_of(run, table, model_name=model_name, model_version=model_version)
    return artifact, predictions_for(
        table,
        rows,
        predicted,
        artifact=artifact,
        probability=probability,
        study_id=study_id,
    )


def as_json(artifact: ModelArtifact, predictions: Sequence[Prediction]) -> dict[str, Any]:
    """A serialisable record of one fold's output, for writing beside a score."""
    return {
        "artifact": {
            "artifact_hash": artifact.artifact_hash,
            "model_name": artifact.model_name,
            "model_version": artifact.model_version,
            "task": artifact.task,
            "training_run_id": artifact.training_run_id,
            "feature_schema_version": artifact.feature_schema_version,
            "expected_features": list(artifact.expected_features),
        },
        "predictions": [
            {
                "prediction_id": p.prediction_id,
                "subject_id": p.subject_id,
                "window_start": p.window_start.isoformat(),
                "window_end": p.window_end.isoformat(),
                "predicted_class": p.predicted_class,
                "probability": p.probability,
                "feature_ids": list(p.feature_ids),
                "source_window_ids": list(p.source_window_ids),
                "source_fact_ids": list(p.source_fact_ids),
            }
            for p in predictions
        ],
    }
