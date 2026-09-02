"""A quality-control revision, walked outwards through what it reaches.

This is the half of the cascade that starts inside PhysioML. The CDFS half --
a correction upstream making a prediction stale -- is in
``test_cdfs_integration``; this one is a policy deciding that windows it used
to accept are artifacts, and finding everything computed from them.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from physioml.core import Feature, Prediction, invalidated_by

T0 = datetime(2017, 1, 1, tzinfo=UTC)


def feature(name: str, windows: tuple[str, ...], value: float = 1.0) -> Feature:
    return Feature.create(
        subject_id="S2",
        name=name,
        value=value,
        unit=None,
        feature_set="test",
        feature_set_version="1",
        source_window_ids=windows,
    )


def prediction(features: tuple[Feature, ...], windows: tuple[str, ...], facts=("fact-1",)):
    return Prediction.create(
        study_id="T",
        subject_id="S2",
        task="stress",
        window_start=T0,
        window_end=T0 + timedelta(seconds=60),
        predicted_class="stressed",
        probability=0.8,
        model_name="m",
        model_version="1",
        training_run_id="run-1",
        feature_set_version="1",
        feature_ids=tuple(f.feature_id for f in features),
        source_window_ids=windows,
        source_fact_ids=facts,
    )


def test_a_revision_reaches_the_features_computed_from_a_rejected_window():
    kept = feature("eda_mean", ("w-good",))
    lost = feature("pulse_rate_mean", ("w-bad",))
    found = invalidated_by({"w-bad": ("motion",)}, [kept, lost])

    assert found.windows == ("w-bad",)
    assert found.features == (lost.feature_id,)
    assert kept.feature_id not in found.features


def test_it_reaches_the_predictions_that_used_those_features():
    lost = feature("pulse_rate_mean", ("w-bad",))
    other = feature("eda_mean", ("w-good",))
    affected = prediction((lost, other), ("w-bad", "w-good"))
    unaffected = prediction((other,), ("w-good",))

    found = invalidated_by({"w-bad": ("no_pulse",)}, [lost, other], [affected, unaffected])
    assert affected.prediction_id in found.predictions
    assert unaffected.prediction_id not in found.predictions


def test_a_feature_spanning_two_windows_is_invalid_if_either_is():
    """It is not partly right. A feature computed across an artifact and a
    clean window is a feature computed across an artifact."""
    spanning = feature("acc_magnitude_mean", ("w-good", "w-bad"))
    found = invalidated_by({"w-bad": ("saturated",)}, [spanning])
    assert found.features == (spanning.feature_id,)


def test_the_reasons_travel_with_the_windows():
    """A person acting on this needs to know what the new policy objected to."""
    found = invalidated_by({"w-bad": ("motion", "no_pulse")}, [])
    assert found.reasons["w-bad"] == ("motion", "no_pulse")
    assert "motion" in found.summary() and "no_pulse" in found.summary()


def test_it_carries_the_cdfs_facts_the_recomputation_must_replace():
    """A caller holding only PhysioML identifiers cannot say what to supersede
    on the other side of the boundary."""
    lost = feature("pulse_rate_mean", ("w-bad",))
    affected = prediction((lost,), ("w-bad",), facts=("cdfs-a", "cdfs-b"))
    found = invalidated_by({"w-bad": ("motion",)}, [lost], [affected])
    assert found.source_fact_ids == ("cdfs-a", "cdfs-b")


def test_a_revision_that_rejects_nothing_reaches_nothing():
    lost = feature("pulse_rate_mean", ("w-bad",))
    found = invalidated_by({}, [lost], [prediction((lost,), ("w-bad",))])
    assert not found
    assert found.summary() == "no window is affected by this revision"


def test_a_prediction_naming_a_rejected_window_directly_is_stale():
    """Even when none of its features came from that window -- a model given
    the window itself has still seen the artifact."""
    clean = feature("eda_mean", ("w-good",))
    affected = prediction((clean,), ("w-good", "w-bad"))
    found = invalidated_by({"w-bad": ("flatline",)}, [clean], [affected])
    assert affected.prediction_id in found.predictions


def test_re_judging_a_window_does_not_change_its_identity():
    """The property the whole walk depends on. Were the quality verdict part of
    a window's identity, a revision would produce new identifiers with nothing
    pointing at them, and the features derived from the old ones would name
    windows that no longer existed."""
    from physioml.core import SignalWindow

    window = SignalWindow.create(
        recording_id="r",
        subject_id="S2",
        start_sample=0,
        end_sample=3840,
        start_time=T0,
        sampling_rate_hz=64.0,
    )
    rejected = window.rejected("motion")
    assert rejected.window_id == window.window_id
    assert rejected.qc_reason_codes == ("motion",)

    # And so the walk still finds it.
    derived = feature("pulse_rate_mean", (window.window_id,))
    found = invalidated_by({rejected.window_id: rejected.qc_reason_codes}, [derived])
    assert found.features == (derived.feature_id,)
