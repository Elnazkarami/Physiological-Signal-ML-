from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from physioml.core import (
    Feature,
    FeatureVector,
    Modality,
    ModelArtifact,
    Prediction,
    QCStatus,
    Recording,
    SchemaMismatch,
    SignalWindow,
    TrainingRun,
)

T0 = datetime(2026, 3, 1, 22, 0, tzinfo=UTC)


def recording(**over) -> Recording:
    return Recording.create(
        **{
            "study_id": "SLEEP-01",
            "subject_id": "S017",
            "modality": Modality.EEG,
            "sampling_rate_hz": 256.0,
            "start_time": T0,
            "duration_seconds": 3600.0,
            "channels": ("Fp1", "Fp2", "C3", "C4"),
            **over,
        }
    )


def window(rec: Recording, start: int = 0, **over) -> SignalWindow:
    return SignalWindow.create(
        **{
            "recording_id": rec.recording_id,
            "subject_id": rec.subject_id,
            "start_sample": start,
            "end_sample": start + 7680,
            "start_time": T0 + timedelta(seconds=start / 256),
            "sampling_rate_hz": 256.0,
            **over,
        }
    )


def feature(win: SignalWindow, name: str = "alpha_power", value: float = 12.5) -> Feature:
    return Feature.create(
        subject_id=win.subject_id,
        name=name,
        value=value,
        feature_set="eeg-spectral",
        feature_set_version="2.0",
        source_window_ids=(win.window_id,),
        channel="Fp1",
    )


# ── identity ────────────────────────────────────────────────────────────────


def test_the_same_inputs_produce_the_same_identifier():
    """A recomputation that changes nothing must be visibly a no-op."""
    assert recording().recording_id == recording().recording_id


def test_changing_what_defines_an_artifact_changes_its_identifier():
    assert recording().recording_id != recording(sampling_rate_hz=512.0).recording_id
    rec = recording()
    assert window(rec).window_id != window(rec, start=7680).window_id


def test_identifiers_say_what_they_point_at():
    rec = recording()
    win = window(rec)
    assert rec.recording_id.startswith("rec-")
    assert win.window_id.startswith("win-")
    assert feature(win).feature_id.startswith("feat-")


def test_naive_timestamps_are_refused():
    """A recording and a prediction disagreeing about what 14:30 means is the
    kind of defect that only appears once the data crosses a timezone."""
    with pytest.raises(ValueError, match="timezone-aware"):
        recording(start_time=datetime(2026, 3, 1, 22, 0))


# ── recordings ──────────────────────────────────────────────────────────────


def test_a_unit_per_channel_or_none_at_all():
    with pytest.raises(ValueError, match="unit per channel"):
        recording(units=("uV",))
    assert recording(units=("uV",) * 4).units == ("uV",) * 4


def test_a_recording_points_at_signal_rather_than_carrying_it():
    rec = recording(source_uri="s3://bucket/S017.edf", source_hash="abc123")
    assert rec.n_samples == 921_600
    assert "samples" not in rec.__slots__


# ── windows and QC ──────────────────────────────────────────────────────────


def test_a_rejected_window_must_say_why():
    """QC that rejects without a reason cannot be audited or counted."""
    rec = recording()
    with pytest.raises(ValueError, match="record why"):
        SignalWindow.create(
            recording_id=rec.recording_id,
            subject_id="S017",
            start_sample=0,
            end_sample=7680,
            start_time=T0,
            sampling_rate_hz=256.0,
            qc_status=QCStatus.REJECTED,
        )


def test_rejecting_a_window_does_not_mutate_the_original():
    rec = recording()
    good = window(rec)
    bad = good.rejected("flat_channel", "amplitude_out_of_range")

    assert good.qc_status is QCStatus.VALID, "the original is untouched"
    assert bad.qc_status is QCStatus.REJECTED
    assert bad.qc_reason_codes == ("flat_channel", "amplitude_out_of_range")


def test_a_qc_verdict_does_not_change_a_window_identity():
    """Identity is the physical slice; QC is a judgement about it.

    This is what makes cascade invalidation expressible: "window W was
    rejected, so every feature naming W is stale". Were the identifier to
    change on rejection, features computed while the window was still
    considered good would name an identifier that no longer exists.
    """
    rec = recording()
    good = window(rec)
    assert good.rejected("motion").window_id == good.window_id


def test_a_rejected_window_is_kept_not_discarded():
    """'How much of this subject survived QC' has to be answerable."""
    rec = recording()
    windows = [window(rec, start=i * 7680) for i in range(4)]
    windows[2] = windows[2].rejected("motion")
    assert len(windows) == 4
    assert sum(1 for w in windows if w.qc_status.usable) == 3


def test_an_empty_window_is_refused():
    rec = recording()
    with pytest.raises(ValueError, match="must contain samples"):
        window(rec, start=0, end_sample=0)


# ── features ────────────────────────────────────────────────────────────────


def test_a_feature_must_name_what_it_came_from():
    with pytest.raises(ValueError, match="no source window"):
        Feature.create(
            subject_id="S017",
            name="alpha_power",
            value=1.0,
            feature_set="eeg-spectral",
            feature_set_version="2.0",
            source_window_ids=(),
        )


def test_a_feature_must_carry_its_algorithm_version():
    """Without it, a later correction to the algorithm cannot find its outputs."""
    rec = recording()
    with pytest.raises(ValueError, match="feature_set_version"):
        Feature.create(
            subject_id="S017",
            name="alpha_power",
            value=1.0,
            feature_set="eeg-spectral",
            feature_set_version="",
            source_window_ids=(window(rec).window_id,),
        )


def test_a_vector_orders_features_reproducibly():
    rec = recording()
    win = window(rec)
    made = [feature(win, "theta_power"), feature(win, "alpha_power")]
    assert FeatureVector.of(made, window_id=win.window_id).names == (
        "alpha_power@Fp1",
        "theta_power@Fp1",
    )


def test_a_vector_refuses_to_mix_feature_set_versions():
    rec = recording()
    win = window(rec)
    other = Feature.create(
        subject_id="S017",
        name="beta_power",
        value=3.0,
        feature_set="eeg-spectral",
        feature_set_version="1.0",
        source_window_ids=(win.window_id,),
        channel="Fp1",
    )
    with pytest.raises(ValueError, match="feature-set versions"):
        FeatureVector.of([feature(win), other], window_id=win.window_id)


# ── training runs: the leakage guarantee ────────────────────────────────────


def test_a_subject_may_not_appear_in_two_splits():
    """The single most common way a physiological classifier is wrong."""
    with pytest.raises(ValueError, match="leakage"):
        TrainingRun.create(
            task="sleep_stage",
            dataset_version="v1",
            split_strategy="group_k_fold",
            train_subjects=("S001", "S017"),
            test_subjects=("S017",),
        )


def test_leakage_is_caught_through_validation_too():
    with pytest.raises(ValueError, match="leakage"):
        TrainingRun.create(
            task="sleep_stage",
            dataset_version="v1",
            split_strategy="group_k_fold",
            train_subjects=("S001",),
            validation_subjects=("S002",),
            test_subjects=("S002",),
        )


def test_a_run_needs_both_train_and_test_subjects():
    with pytest.raises(ValueError, match="train and test"):
        TrainingRun.create(
            task="sleep_stage",
            dataset_version="v1",
            split_strategy="loso",
            train_subjects=("S001",),
            test_subjects=(),
        )


# ── models: refusing the mismatch ───────────────────────────────────────────


def run() -> TrainingRun:
    return TrainingRun.create(
        task="sleep_stage",
        dataset_version="sleep-edf-1",
        split_strategy="leave_one_subject_out",
        train_subjects=("S001", "S002"),
        test_subjects=("S017",),
        random_seed=7,
    )


def model(features: tuple[str, ...], version: str = "2.0") -> ModelArtifact:
    return ModelArtifact.create(
        model_name="sleep_rf",
        model_version="1.2.0",
        task="sleep_stage",
        training_run_id=run().training_run_id,
        expected_features=features,
        feature_schema_version=version,
    )


def test_a_matching_vector_is_accepted():
    rec = recording()
    win = window(rec)
    vector = FeatureVector.of([feature(win)], window_id=win.window_id)
    model(vector.names).accepts(vector)


def test_a_vector_from_a_different_feature_version_is_refused():
    rec = recording()
    win = window(rec)
    vector = FeatureVector.of([feature(win)], window_id=win.window_id)
    with pytest.raises(SchemaMismatch, match="feature schema"):
        model(vector.names, version="1.0").accepts(vector)


def test_the_right_features_in_the_wrong_order_are_refused():
    """Scoring would silently misalign every column and look fine."""
    rec = recording()
    win = window(rec)
    vector = FeatureVector.of(
        [feature(win, "alpha_power"), feature(win, "theta_power")], window_id=win.window_id
    )
    with pytest.raises(SchemaMismatch, match="different order"):
        model(tuple(reversed(vector.names))).accepts(vector)


def test_missing_and_unexpected_features_are_named():
    rec = recording()
    win = window(rec)
    vector = FeatureVector.of([feature(win, "alpha_power")], window_id=win.window_id)
    with pytest.raises(SchemaMismatch, match="missing"):
        model(("delta_power@Fp1",)).accepts(vector)


def test_a_model_must_declare_what_it_expects():
    with pytest.raises(ValueError, match="declare the features"):
        ModelArtifact.create(
            model_name="m",
            model_version="1",
            task="t",
            training_run_id="trun-x",
            expected_features=(),
            feature_schema_version="1.0",
        )


# ── predictions: accountability is required, not encouraged ─────────────────


def prediction(**over) -> Prediction:
    rec = recording()
    win = window(rec)
    feat = feature(win)
    return Prediction.create(
        **{
            "study_id": "SLEEP-01",
            "subject_id": "S017",
            "task": "sleep_stage",
            "window_start": win.start_time,
            "window_end": win.end_time,
            "predicted_class": "N2",
            "probability": 0.87,
            "model_name": "sleep_rf",
            "model_version": "1.2.0",
            "training_run_id": run().training_run_id,
            "feature_set_version": "2.0",
            "feature_ids": (feat.feature_id,),
            "source_window_ids": (win.window_id,),
            **over,
        }
    )


def test_a_complete_prediction_is_built():
    made = prediction()
    assert made.prediction_id.startswith("pred-")
    assert made.outcome == "N2"


@pytest.mark.parametrize(
    "missing", ["model_name", "model_version", "training_run_id", "feature_set_version"]
)
def test_a_prediction_that_cannot_name_what_produced_it_is_refused(missing):
    with pytest.raises(ValueError, match="not accountable"):
        prediction(**{missing: ""})


def test_a_prediction_must_trace_to_signal():
    with pytest.raises(ValueError, match="no source windows"):
        prediction(source_window_ids=())


def test_a_prediction_must_predict_something():
    with pytest.raises(ValueError, match="must predict something"):
        prediction(predicted_class=None, predicted_value=None)


def test_an_impossible_probability_is_refused():
    with pytest.raises(ValueError, match="not in"):
        prediction(probability=1.4)


def test_a_backwards_window_is_refused():
    with pytest.raises(ValueError, match="must follow"):
        prediction(window_end=T0 - timedelta(seconds=1))


def test_the_full_chain_is_traceable_end_to_end():
    """The plan's primary question, asserted: prediction back to source."""
    rec = recording(source_fact_ids=("fact-a", "fact-b"))
    win = window(rec)
    feat = feature(win)
    vector = FeatureVector.of([feat], window_id=win.window_id)
    trained = run()
    artifact = model(vector.names)
    artifact.accepts(vector)

    made = Prediction.create(
        study_id="SLEEP-01",
        subject_id="S017",
        task="sleep_stage",
        window_start=win.start_time,
        window_end=win.end_time,
        predicted_class="N2",
        probability=0.87,
        model_name=artifact.model_name,
        model_version=artifact.model_version,
        training_run_id=trained.training_run_id,
        feature_set_version=vector.feature_set_version,
        feature_ids=vector.feature_ids,
        source_window_ids=(win.window_id,),
    )

    # prediction → features → window → recording → CDFS facts
    assert feat.feature_id in made.feature_ids
    assert win.window_id in made.source_window_ids
    assert feat.source_window_ids == (win.window_id,)
    assert win.recording_id == rec.recording_id
    assert rec.source_fact_ids == ("fact-a", "fact-b")
    # prediction → model → training run → the subjects it never saw
    assert made.training_run_id == trained.training_run_id
    assert "S017" not in trained.train_subjects
