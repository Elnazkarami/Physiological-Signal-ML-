"""The loop, end to end, against a real CDFS.

CDFS facts → PhysioML windows and features → a prediction → a CDFS derived fact
→ a lineage query that reaches the original observations. That chain is the
project's primary claim, so it is asserted against a running deployment rather
than a mock: a mock would agree with whatever this file believed CDFS does.

Skipped when CDFS is not installed, since PhysioML is usable without it.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path
from wsgiref.simple_server import WSGIServer, make_server

import pytest

from physioml.cdfs.client import CDFSClient, CDFSError
from physioml.core import (
    Feature,
    FeatureVector,
    ModelArtifact,
    Prediction,
    Recording,
    SignalWindow,
    TrainingRun,
)

cdfs = pytest.importorskip("cdfs", reason="CDFS is not installed")

CDFS_REPO = Path.home() / "Downloads" / "clinical-data-fabric-system"
pytestmark = pytest.mark.skipif(
    not (CDFS_REPO / "studies" / "cardio_fx_01" / "study.toml").is_file(),
    reason="the CDFS study bundle is not available",
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
    run = TrainingRun.create(
        task="event_risk",
        dataset_version="cardio-fx-01",
        split_strategy="leave_one_subject_out",
        train_subjects=("CARDIO-01-002", "CARDIO-01-003"),
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
