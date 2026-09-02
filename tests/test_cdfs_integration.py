"""The loop, end to end, against a real CDFS.

CDFS facts → PhysioML windows and features → a prediction → a CDFS derived fact
→ a lineage query that reaches the original observations. That chain is the
project's primary claim, so it is asserted against a running deployment rather
than a mock: a mock would agree with whatever this file believed CDFS does.

Skipped when CDFS is not installed, since PhysioML is usable without it.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from datetime import UTC, datetime
from wsgiref.simple_server import WSGIServer, make_server

import pytest

from physioml.cdfs.client import CDFSClient, CDFSError
from physioml.cdfs.invalidation import stale_predictions
from physioml.core import (
    Feature,
    FeatureVector,
    ModelArtifact,
    Prediction,
    Recording,
    SignalWindow,
    TrainingRun,
    invalidated_by,
)
from tests.paths import CDFS_MISSING, CDFS_REPO

cdfs = pytest.importorskip("cdfs", reason="CDFS is not installed")

pytestmark = pytest.mark.skipif(
    not (CDFS_REPO / "studies" / "cardio_fx_01" / "study.toml").is_file(),
    reason=CDFS_MISSING,
)

STUDY_ID = "CARDIO-FX-01"
SUBJECT = "CARDIO-01-001"
T0 = datetime(2026, 3, 1, 22, 0, tzinfo=UTC)


class QuietServer(WSGIServer):
    """A single-threaded WSGI server; the tests drive it one request at a time."""


@pytest.fixture(scope="module")
def deployment():
    """A real CDFS, populated, served over HTTP, with a model-service token."""
    from cdfs.api import CdfsService, ServiceConfig, WsgiApp
    from cdfs.auth import Grant, TokenAuthenticator, TokenStore, system
    from cdfs.model import FactStore
    from cdfs.study import StudyRegistry, load_study

    loader = system("test-loader")
    study = load_study(CDFS_REPO / "studies" / "cardio_fx_01")
    service = CdfsService(
        StudyRegistry([study]),
        FactStore(":memory:"),
        ServiceConfig(data_root=CDFS_REPO / "raw_data"),
    )
    service.ingest_source(loader, STUDY_ID, "REDCAP", "redcap_export_all_sites.csv")
    service.derive(loader, STUDY_ID, apply=True)

    tokens = TokenStore(":memory:")
    issued = tokens.issue("physioml", [Grant("model_service", STUDY_ID)])
    app = WsgiApp(service, TokenAuthenticator(tokens))

    server = make_server("127.0.0.1", 0, app, server_class=QuietServer)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield CDFSClient(f"http://127.0.0.1:{port}", issued.value), service, loader
    finally:
        # shutdown() stops the serve loop; server_close() releases the listening
        # socket. Only doing the first leaks it, and the warning surfaces during
        # some later test's teardown rather than here.
        server.shutdown()
        server.server_close()
        service.close()
        tokens.close()


@pytest.fixture(scope="module")
def client(deployment):
    return deployment[0]


def build_prediction(observations: list[dict], *, subject=SUBJECT):
    """Everything PhysioML would produce for one window, from real facts."""
    used = [f["fact_id"] for f in observations if f["coordinate"]["field"] == "bmi"][:2]
    assert used, "the fixture study should have derived a BMI"

    recording = Recording.create(
        study_id=STUDY_ID,
        subject_id=subject,
        modality="bvp",
        sampling_rate_hz=64.0,
        start_time=T0,
        duration_seconds=300.0,
        channels=("bvp",),
        source_fact_ids=tuple(used),
    )
    window = SignalWindow.create(
        recording_id=recording.recording_id,
        subject_id=subject,
        start_sample=0,
        end_sample=1920,
        start_time=T0,
        sampling_rate_hz=64.0,
        source_fact_ids=tuple(used),
    )
    feature = Feature.create(
        subject_id=subject,
        name="hr_mean",
        value=72.4,
        unit="bpm",
        feature_set="peripheral-basic",
        feature_set_version="1.0",
        source_window_ids=(window.window_id,),
    )
    vector = FeatureVector.of([feature], window_id=window.window_id)
    # Held out from whoever is being predicted for. Naming the subjects
    # literally put the test subject in the training set as soon as this helper
    # was reused for a second person, and TrainingRun refused the run -- which
    # is what it is for.
    cohort = ("CARDIO-01-001", "CARDIO-01-002", "CARDIO-01-003")
    run = TrainingRun.create(
        task="event_risk",
        dataset_version="cardio-fx-01",
        split_strategy="leave_one_subject_out",
        train_subjects=tuple(s for s in cohort if s != subject),
        test_subjects=(subject,),
        random_seed=7,
    )
    model = ModelArtifact.create(
        model_name="cv_rf",
        model_version="1.2.0",
        task="event_risk",
        training_run_id=run.training_run_id,
        expected_features=vector.names,
        feature_schema_version=vector.feature_set_version,
    )
    model.accepts(vector)

    prediction = Prediction.create(
        study_id=STUDY_ID,
        subject_id=subject,
        task="event_risk",
        window_start=window.start_time,
        window_end=window.end_time,
        predicted_class="elevated",
        probability=0.81,
        model_name=model.model_name,
        model_version=model.model_version,
        training_run_id=run.training_run_id,
        feature_set_version=vector.feature_set_version,
        feature_ids=vector.feature_ids,
        source_window_ids=(window.window_id,),
        source_fact_ids=tuple(used),
    )
    return prediction, used


# ── reading ─────────────────────────────────────────────────────────────────


def test_the_client_reads_a_subject_over_the_api(client):
    values = client.subject_values(STUDY_ID, SUBJECT)
    assert values, "the subject should have observations"
    assert all("fact_id" in v for v in values)


def test_identifiers_are_not_returned_to_a_model_service(client):
    """This client never asks for PHI, and the service would refuse if it did."""
    fields = {v["coordinate"]["field"] for v in client.subject_values(STUDY_ID, SUBJECT)}
    assert not fields & {"mrn", "last_name", "first_name", "birth_date"}


def test_the_study_declares_where_model_output_may_go(client):
    """Checkable before a training run, rather than discovered after one."""
    assert set(client.model_fields(STUDY_ID)) >= {
        "predicted_event_risk",
        "predicted_event_confidence",
    }


# ── writing, and the chain it creates ───────────────────────────────────────


def test_a_prediction_written_back_traces_to_its_source_observations(client):
    """§15.5, and the project's primary question, asserted end to end."""
    observations = client.subject_values(STUDY_ID, SUBJECT)
    prediction, _used = build_prediction(observations)

    result = client.write_predictions(
        STUDY_ID,
        [prediction],
        field="predicted_event_risk",
        confidence_field="predicted_event_confidence",
    )
    assert result["written"] == 2, "the class and its confidence"

    written = client.lineage(result["fact_ids"][0])
    fact = written["fact"]
    assert fact["value"] == "elevated"
    assert fact["transform_id"] == "cv_rf@1.2.0"
    assert fact["entered_by"] == "physioml", "attribution is the token, not the payload"

    # the window is recoverable even though CDFS addresses visits, not intervals
    reference = fact["source_record_ref"]
    assert f"prediction={prediction.prediction_id}" in reference
    assert f"training_run={prediction.training_run_id}" in reference
    assert prediction.window_start.isoformat() in reference

    # and the chain reaches the observations the model was given
    ancestors = {f["coordinate"]["field"] for f in written["ancestors"]}
    assert {"weight_kg", "height_cm"} <= ancestors, (
        "the prediction should reach the collected values behind its inputs"
    )


def test_correcting_an_input_reports_the_prediction_stale(client, deployment):
    """The §13 loop, from the ML side: CDFS names what PhysioML must redo."""
    _, service, loader = deployment
    observations = client.subject_values(STUDY_ID, SUBJECT)
    prediction, _ = build_prediction(observations)
    client.write_predictions(STUDY_ID, [prediction], field="predicted_event_risk")

    plan = service.plan_correction(
        loader,
        STUDY_ID,
        subject_id=SUBJECT,
        field="weight_kg",
        value=99.0,
        reason="transcription error found on source",
        visit_num=1,
    )
    stale = [b for b in plan["blocked"] if b["field"] == "predicted_event_risk"]
    assert stale, f"the prediction was not reported stale; blocked={plan['blocked']}"
    assert "write it back" in stale[0]["reason"]


# ── what the boundary refuses ───────────────────────────────────────────────


def test_a_model_service_cannot_write_a_collected_field(client):
    observations = client.subject_values(STUDY_ID, SUBJECT)
    prediction, _ = build_prediction(observations)
    with pytest.raises(CDFSError) as raised:
        client.write_predictions(STUDY_ID, [prediction], field="sbp_mmhg")
    assert raised.value.status == 403


def test_a_prediction_claiming_invented_inputs_is_refused(client):
    """Provenance is checked at the boundary, not taken on trust."""
    observations = client.subject_values(STUDY_ID, SUBJECT)
    prediction, _ = build_prediction(observations)
    forged = Prediction.create(
        study_id=STUDY_ID,
        subject_id=SUBJECT,
        task=prediction.task,
        window_start=prediction.window_start,
        window_end=prediction.window_end,
        predicted_class="elevated",
        model_name=prediction.model_name,
        model_version=prediction.model_version,
        training_run_id=prediction.training_run_id,
        feature_set_version=prediction.feature_set_version,
        feature_ids=prediction.feature_ids,
        source_window_ids=prediction.source_window_ids,
        source_fact_ids=("fact-invented",),
    )
    with pytest.raises(CDFSError, match="not in the store"):
        client.write_predictions(STUDY_ID, [forged], field="predicted_event_risk")


def test_an_unauthenticated_client_gets_nowhere(client):
    anonymous = CDFSClient(client.base_url, "not-a-token")
    with pytest.raises(CDFSError) as raised:
        anonymous.subject_values(STUDY_ID, SUBJECT)
    assert raised.value.status == 401


# ── cascade invalidation, §13 ───────────────────────────────────────────────


def live_prediction(client, field="predicted_event_risk", subject=SUBJECT):
    facts = [
        f
        for f in client.subject_values(STUDY_ID, subject)
        if f["coordinate"]["field"] == field
    ]
    assert len(facts) <= 1, f"{len(facts)} predictions in force at one coordinate"
    return facts[0] if facts else None


def test_a_correction_upstream_makes_a_prediction_stale_and_it_is_recomputed(deployment):
    """§13, end to end, across both systems.

    A weight is corrected. CDFS recomputes the BMI it derived and supersedes
    it, then reports the prediction downstream as blocked, because it has no
    model to recompute it with. PhysioML discovers that its inputs moved,
    recomputes, and writes the new value back as a replacement -- leaving one
    prediction in force and a chain a reviewer can follow from the retracted
    weight to the value that stands today.
    """
    client, service, loader = deployment
    subject = "CARDIO-01-002"

    observations = client.subject_values(STUDY_ID, subject)
    prediction, used = build_prediction(observations, subject=subject)
    written = client.write_predictions(
        STUDY_ID,
        [prediction],
        field="predicted_event_risk",
        confidence_field="predicted_event_confidence",
    )
    original = written["fact_ids"][0]
    assert not stale_predictions(client, STUDY_ID, subject), "nothing has changed yet"

    # ── the correction ──────────────────────────────────────────────────────
    weight = next(f for f in observations if f["coordinate"]["field"] == "weight_kg")
    plan = service.apply_correction(
        loader,
        STUDY_ID,
        subject_id=subject,
        field="weight_kg",
        value=float(weight["value"]) + 12.0,
        visit_num=weight["coordinate"].get("visit_num"),
        reason="site reported the wrong scale reading",
    )

    # CDFS recomputes what it derived, and says what it cannot.
    blocked = " ".join(str(b) for b in plan.get("blocked", []))
    assert "predicted_event_risk" in blocked
    assert "recompute it in the system that produced it" in blocked

    # ── PhysioML notices ────────────────────────────────────────────────────
    stale = stale_predictions(client, STUDY_ID, subject)
    # The class and its confidence are two facts at two coordinates, and both
    # rest on the BMI that moved. Both are stale; retiring only one would leave
    # a confidence in force for a prediction that no longer exists.
    by_field = {item.field: item for item in stale}
    assert set(by_field) == {"predicted_event_risk", "predicted_event_confidence"}
    found = by_field["predicted_event_risk"]
    assert found.fact_id == original
    assert found.value == "elevated"
    assert [item.field for item in found.superseded] == ["bmi"]
    moved = found.superseded[0]
    # Both sides of the change, and both real: an absent old value would still
    # satisfy "was != now" while telling a reviewer nothing about what moved.
    assert isinstance(moved.was, int | float) and isinstance(moved.now, int | float)
    assert moved.now > moved.was, "twelve kilograms heavier is a higher BMI"
    assert f"bmi {moved.was} -> {moved.now}" in found.reason
    assert "upstream correction" in found.reason

    # The recomputation is told which facts to cite, not left to guess.
    assert set(found.current_inputs) != set(used)
    assert all(
        client.lineage(f)["history"][-1]["fact_id"] == f for f in found.current_inputs
    ), "every input a recomputation cites should itself be in force"

    # ── the recomputation ───────────────────────────────────────────────────
    recomputed, _ = build_prediction(
        client.subject_values(STUDY_ID, subject), subject=subject
    )
    recomputed = replace(
        recomputed,
        predicted_class="high",
        probability=0.93,
        source_fact_ids=found.current_inputs,
    )
    result = client.replace_prediction(
        STUDY_ID,
        recomputed,
        field="predicted_event_risk",
        supersedes=found.fact_id,
        confidence_field="predicted_event_confidence",
        confidence_supersedes=by_field["predicted_event_confidence"].fact_id,
        reason=found.reason,
    )
    assert result["written"] == 2, "the class and its confidence, both replaced"

    # ── what a reader sees afterwards ───────────────────────────────────────
    current = live_prediction(client, subject=subject)
    assert current is not None
    assert current["fact_id"] == result["fact_ids"][0]
    assert current["value"] == "high", "the retracted value is no longer in force"

    confidence = live_prediction(client, "predicted_event_confidence", subject)
    assert confidence is not None
    assert float(confidence["value"]) == pytest.approx(0.93)

    history = client.lineage(current["fact_id"])["history"]
    assert [h["fact_id"] for h in history] == [original, current["fact_id"]]
    assert history[-1]["reason"] == found.reason

    assert not stale_predictions(client, STUDY_ID, subject), "the cascade is settled"


def test_a_prediction_already_replaced_is_not_reported_stale_again(deployment):
    """Only values in force are checked; history is not work to redo."""
    client, _service, _loader = deployment
    subject = "CARDIO-01-002"
    current = live_prediction(client, subject=subject)
    assert current is not None
    for found in stale_predictions(client, STUDY_ID, subject):
        assert found.fact_id == current["fact_id"]


def test_a_replacement_must_say_why(deployment):
    client, _service, _loader = deployment
    observations = client.subject_values(STUDY_ID, SUBJECT)
    prediction, _ = build_prediction(observations)
    with pytest.raises(ValueError, match="must carry a reason"):
        client.replace_prediction(
            STUDY_ID,
            prediction,
            field="predicted_event_risk",
            supersedes="f" * 64,
            reason="",
        )


def test_a_prediction_cannot_replace_an_observation(deployment):
    """The model service holds derived:write, and that is all it holds."""
    client, _service, _loader = deployment
    observations = client.subject_values(STUDY_ID, SUBJECT)
    prediction, used = build_prediction(observations)
    with pytest.raises(CDFSError) as caught:
        client.replace_prediction(
            STUDY_ID,
            prediction,
            field="predicted_event_risk",
            supersedes=used[0],
            reason="attempting to overwrite an observation",
        )
    assert caught.value.status == 400


# ── the other direction: a quality-control revision ─────────────────────────


def test_a_quality_revision_invalidates_a_prediction_and_the_replacement_stands(
    deployment,
):
    """The physiological half of the cascade, end to end against a real CDFS.

    The other integration test starts with a clinical correction: a weight is
    wrong, the engine recomputes the body-mass index it derived, and the
    prediction downstream goes stale. This one starts where the signal is. A
    quality-control policy is revised, a window that used to pass is now an
    artifact, and everything computed from it has to be found and replaced --
    inside PhysioML first, then across the boundary.
    """
    client, _service, _loader = deployment
    subject = "CARDIO-01-003"

    observations = client.subject_values(STUDY_ID, subject)
    original, used = build_prediction(observations, subject=subject)
    written = client.write_predictions(
        STUDY_ID,
        [original],
        field="predicted_event_risk",
        confidence_field="predicted_event_confidence",
    )
    assert written["written"] == 2

    # ── the revision ────────────────────────────────────────────────────────
    # A better artifact detector now rejects the window this rested on. The
    # window's identity does not change, which is what makes the next line
    # possible at all.
    window_id = original.source_window_ids[0]
    feature = Feature.create(
        subject_id=subject,
        name="hr_mean",
        value=72.4,
        unit="bpm",
        feature_set="peripheral-basic",
        feature_set_version="1.0",
        source_window_ids=(window_id,),
    )
    reached = invalidated_by({window_id: ("motion", "no_pulse")}, [feature], [original])

    assert reached, "the revision should reach something"
    assert reached.windows == (window_id,)
    assert original.prediction_id in reached.predictions
    assert set(reached.source_fact_ids) == set(used), (
        "and it should name the CDFS facts the replacement has to be written against"
    )

    # ── the replacement ─────────────────────────────────────────────────────
    recomputed = replace(
        original,
        predicted_class="low",
        probability=0.22,
        source_window_ids=(f"{window_id}-rescored",),
    )
    stale = live_prediction(client, subject=subject)
    assert stale is not None

    result = client.replace_prediction(
        STUDY_ID,
        recomputed,
        field="predicted_event_risk",
        supersedes=stale["fact_id"],
        reason=(
            "recomputed after a quality-control revision rejected the source "
            f"window ({', '.join(reached.reasons[window_id])})"
        ),
    )
    assert result["written"] == 1

    # ── what a reviewer sees ────────────────────────────────────────────────
    current = live_prediction(client, subject=subject)
    assert current is not None
    assert current["value"] == "low", "the value computed from the artifact is retired"

    history = client.lineage(current["fact_id"])["history"]
    assert [h["fact_id"] for h in history] == [stale["fact_id"], current["fact_id"]]
    assert "quality-control revision" in history[-1]["reason"]
    assert "motion" in history[-1]["reason"], "and it names what the policy objected to"

    # The chain still reaches the observations, through the new value.
    lineage = client.lineage(current["fact_id"])
    assert {a["fact_id"] for a in lineage["ancestors"]} & set(used)


def test_a_revision_that_rejects_nothing_leaves_the_prediction_in_force(deployment):
    """The control. A policy change that clears no window changes nothing."""
    client, _service, _loader = deployment
    subject = "CARDIO-01-003"
    before = live_prediction(client, subject=subject)
    assert before is not None

    reached = invalidated_by({}, [], [])
    assert not reached

    after = live_prediction(client, subject=subject)
    assert after is not None
    assert after["fact_id"] == before["fact_id"]
