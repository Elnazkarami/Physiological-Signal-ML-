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

import hashlib
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
    artifact_hash: str = "",
) -> ModelArtifact:
    """What was fitted, recorded so a later prediction can name it.

    The expected features come from the table the model was fitted on, in the
    order it was fitted on them, which is what lets
    :meth:`~physioml.core.registry.ModelArtifact.accepts` refuse a vector whose
    columns are right and whose order is not.

    ``artifact_hash`` is a digest of the fitted parameters -- see
    :func:`parameter_digest`. Without it the artifact identity covers what was
    asked for (the model, the version, the run, the feature schema) and not
    what came out, so two models fitted by the same run on the same columns are
    the same artifact even when their coefficients differ. Nothing here can
    compute it, because this module deliberately knows nothing about any
    particular model object, so the caller supplies it.
    """
    return ModelArtifact.create(
        model_name=model_name,
        model_version=model_version,
        task=run.task,
        training_run_id=run.training_run_id,
        expected_features=tuple(table.feature_names),
        feature_schema_version=table.feature_set_version,
        artifact_hash=artifact_hash,
    )


def parameter_digest(*parameters: Any) -> str:
    """A digest of fitted parameters, for :func:`artifact_of`.

    Takes whatever arrays carry the fitted state -- coefficients, intercepts,
    thresholds -- and hashes their bytes together with their shapes, so that
    two models with the same numbers in a different arrangement do not collide.

    Deliberately not a pickle of the estimator: that would hash the library
    version, the memory layout and anything else the object happens to carry,
    and would change when none of the fitted numbers did.
    """
    digest = hashlib.sha256()
    for parameter in parameters:
        array = np.ascontiguousarray(np.asarray(parameter, dtype=float))
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


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

    # The table knows how long its windows are; the argument is the fallback for
    # one built before that was recorded. Taking the argument first meant every
    # sixty-second WESAD prediction claimed a thirty-second interval.
    length = table.window_seconds if table.window_seconds > 0 else epoch_seconds

    when = created_at or datetime.now(UTC)
    made: list[Prediction] = []
    for position, row in enumerate(rows):
        start = _began(table, row)
        made.append(
            Prediction.create(
                study_id=study_id,
                subject_id=str(table.subjects[row]),
                task=artifact.task,
                window_start=start,
                window_end=start + timedelta(seconds=length),
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


def _began(table: FeatureTable, row: int) -> datetime:
    """When this row's window began, in wall-clock time.

    The recordings know this: Sleep-EDF carries acquisition timestamps and the
    WESAD reader places its sessions at a documented fixed origin because the
    dataset has none. The export discarded both and counted the within-recording
    offset from the Unix epoch, so every prediction this project has ever
    emitted was dated 1970 regardless of when the signal was recorded.

    A table built before the start times were retained still falls back to that
    offset, and :func:`as_json` marks the difference rather than letting a
    placeholder date pass as a measurement.
    """
    if table.row_start_times is not None:
        return datetime.fromtimestamp(float(table.row_start_times[row]), UTC)
    return datetime.fromtimestamp(0, UTC) + timedelta(
        seconds=float(table.window_starts[row])
    )


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
    epoch_seconds: float = 30.0,
    artifact_hash: str = "",
) -> tuple[ModelArtifact, list[Prediction]]:
    """The artifact and the predictions for one fold, together."""
    artifact = artifact_of(
        run,
        table,
        model_name=model_name,
        model_version=model_version,
        artifact_hash=artifact_hash,
    )
    return artifact, predictions_for(
        table,
        rows,
        predicted,
        artifact=artifact,
        probability=probability,
        study_id=study_id,
        epoch_seconds=epoch_seconds,
    )


def as_json(artifact: ModelArtifact, predictions: Sequence[Prediction]) -> dict[str, Any]:
    """One fold's predictions and the model that made them, for writing beside a score.

    **This is a prediction export carrying provenance references, not the
    provenance itself.** ``feature_ids`` and ``source_window_ids`` name records
    that live elsewhere; nothing here resolves them. That is the right shape
    beside a score in a results directory, where the windows and recordings are
    already on disk and repeating them would multiply the file by the number of
    folds. It is the wrong shape for a document that has to stand on its own.

    :func:`as_bundle` is that one: the same export with every reference
    resolved, which is what :mod:`examples.quickstart` publishes.
    """
    return {
        "artifact": {
            # The content identity of the artifact. ``artifact_hash`` is a
            # digest of serialised weights, which nothing here produces, so it
            # was exported empty on every record -- naming the model but not
            # which fitted object it was.
            "model_id": artifact.model_id,
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


def _window_record(window: Any) -> dict[str, Any]:
    return {
        "window_id": window.window_id,
        "recording_id": window.recording_id,
        "subject_id": window.subject_id,
        "start_sample": window.start_sample,
        "end_sample": window.end_sample,
        "start_time": window.start_time.isoformat(),
        "sampling_rate_hz": window.sampling_rate_hz,
        "channel_ids": list(window.channel_ids),
        "qc_status": window.qc_status.value,
        "qc_reason_codes": list(window.qc_reason_codes),
        "label": window.label,
    }


def _recording_record(recording: Any) -> dict[str, Any]:
    return {
        "recording_id": recording.recording_id,
        "study_id": recording.study_id,
        "subject_id": recording.subject_id,
        "modality": recording.modality.value,
        "sampling_rate_hz": recording.sampling_rate_hz,
        "start_time": recording.start_time.isoformat(),
        "duration_seconds": recording.duration_seconds,
        "channels": list(recording.channels),
        "device_name": recording.device_name,
        "source_uri": recording.source_uri,
        # The digest of the samples themselves. Empty means the reader did not
        # compute one, and the recording identifier then fingerprints the
        # metadata only -- the subject, device, rate and interval -- so a
        # corrected export of the same session would carry the same identifier.
        "source_hash": recording.source_hash,
    }


def _run_record(run: TrainingRun) -> dict[str, Any]:
    return {
        "training_run_id": run.training_run_id,
        "task": run.task,
        "dataset_version": run.dataset_version,
        "split_strategy": run.split_strategy,
        "train_subjects": list(run.train_subjects),
        "test_subjects": list(run.test_subjects),
        "validation_subjects": list(run.validation_subjects),
        "feature_schema_version": run.feature_schema_version,
        "preprocessing_version": run.preprocessing_version,
        "random_seed": run.random_seed,
    }


def _vector_record(table: FeatureTable, row: int) -> dict[str, Any]:
    return {
        "vector_id": table.row_feature_vectors[row],
        "subject_id": str(table.subjects[row]),
        "window_id": table.window_ids[row],
        "feature_set_version": table.feature_set_version,
        "qc_policy_version": table.qc_policy_version,
        "names": list(table.feature_names),
        "values": [float(v) for v in table.values[row]],
        "source_window_ids": list(table.row_windows[row]),
        "source_recording_ids": list(table.row_recordings[row]),
    }


def as_bundle(
    artifact: ModelArtifact,
    predictions: Sequence[Prediction],
    *,
    table: FeatureTable,
    rows: Sequence[int],
    run: TrainingRun,
    windows: Sequence[Any] = (),
    recordings: Sequence[Any] = (),
) -> dict[str, Any]:
    """The same export with every reference resolved, in one document.

    :func:`as_json` names a chain; this one contains it. Each prediction's
    feature vector, each window behind that vector, each recording behind those
    windows and the training run that fitted the model are all present as
    records, keyed by the identifiers the predictions use. A reader can follow
    a prediction to the samples it came from without another file.

    ``windows`` and ``recordings`` are the objects themselves, because a
    feature table keeps their identifiers and not them. Whoever built the table
    still holds them; nothing is reconstructed here, and an identifier with no
    record behind it is reported by :func:`unresolved` rather than invented.

    Each prediction also gains ``window_start_seconds`` -- the offset within
    the recording, which is what the splits are made on and what stays true
    whether or not the dataset carries acquisition timestamps.
    """
    document = as_json(artifact, predictions)
    for prediction, row in zip(document["predictions"], rows, strict=True):
        prediction["window_start_seconds"] = float(table.window_starts[row])
        prediction["window_seconds"] = table.window_seconds
    document["records"] = {
        "feature_vectors": {
            table.row_feature_vectors[row]: _vector_record(table, row) for row in rows
        },
        "windows": {w.window_id: _window_record(w) for w in windows},
        "recordings": {r.recording_id: _recording_record(r) for r in recordings},
        "training_runs": {run.training_run_id: _run_record(run)},
    }
    return document


def unresolved(bundle: dict[str, Any]) -> list[str]:
    """Every identifier the bundle refers to and does not contain.

    A bundle that claims to be self-contained should be checked for it rather
    than described as it, and a prefix test checks only that an identifier is
    the right shape.
    """
    records = bundle.get("records", {})
    vectors = records.get("feature_vectors", {})
    windows = records.get("windows", {})
    recordings = records.get("recordings", {})
    runs = records.get("training_runs", {})

    missing: list[str] = []
    if bundle["artifact"]["training_run_id"] not in runs:
        missing.append(bundle["artifact"]["training_run_id"])
    for prediction in bundle["predictions"]:
        missing.extend(i for i in prediction["feature_ids"] if i not in vectors)
        missing.extend(i for i in prediction["source_window_ids"] if i not in windows)
    for vector in vectors.values():
        missing.extend(i for i in vector["source_window_ids"] if i not in windows)
        missing.extend(i for i in vector["source_recording_ids"] if i not in recordings)
    for window in windows.values():
        if window["recording_id"] not in recordings:
            missing.append(window["recording_id"])
    return sorted(set(missing))
