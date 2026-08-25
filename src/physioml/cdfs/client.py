"""Reading from CDFS and writing back, over its API rather than its database.

Going through the API is not politeness about layering. It is what keeps the
permission model, the site scoping, the PHI boundary and the access trail
applying to this system as they apply to every other caller. A client that
opened the SQLite file directly would inherit none of them, and the first thing
anyone would ask about a prediction is which of those it had bypassed.

The client is deliberately thin: it does not model CDFS's schema, cache, or
interpret. It fetches observations, and it posts derived facts back through the
one endpoint that accepts them.
"""

from __future__ import annotations

import contextlib
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from physioml.core.prediction import Prediction


class CDFSError(RuntimeError):
    """A request to CDFS failed. Carries the status and the code it returned."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(f"{status} {code}: {message}")
        self.status = status
        self.code = code


@dataclass(frozen=True, slots=True)
class CDFSClient:
    """A CDFS deployment, addressed as a client with a bearer token."""

    base_url: str
    token: str
    timeout: float = 30.0

    # ── reading ──────────────────────────────────────────────────────────────

    def subject_values(self, study_id: str, subject_id: str) -> list[dict[str, Any]]:
        """Everything currently in force for one subject.

        Identifiers are withheld: this client never asks for them. A model that
        needs a patient's name to make a prediction is not a model anyone should
        deploy, and asking would put the request in the access trail as an
        attempt.
        """
        body = self._get(f"/studies/{study_id}/subjects/{subject_id}/values")
        return list(body.get("values", []))

    def schema(self, study_id: str) -> list[dict[str, Any]]:
        return list(self._get(f"/studies/{study_id}/schema").get("fields", []))

    def model_fields(self, study_id: str) -> list[str]:
        """Fields this study will accept model output for.

        Worth checking before a training run rather than after: a pipeline that
        produces something the study has nowhere to put has failed at the point
        it was configured, not at the point it tries to write.
        """
        return [f["name"] for f in self.schema(study_id) if f.get("kind") == "model"]

    def lineage(self, fact_id: str) -> dict[str, Any]:
        return self._get(f"/facts/{fact_id}/lineage")

    # ── writing ──────────────────────────────────────────────────────────────

    def write_predictions(
        self,
        study_id: str,
        predictions: list[Prediction],
        *,
        field: str,
        confidence_field: str | None = None,
        source_system: str = "PHYSIOML",
    ) -> dict[str, Any]:
        """Post predictions back as model-derived facts.

        ``derived_from`` carries the **CDFS** facts the prediction rests on, not
        PhysioML's feature identifiers -- those mean nothing on the other side of
        the boundary, and CDFS checks that every id it is given exists. The
        PhysioML chain travels in ``source_record_ref`` instead, which is what a
        source reference is for.

        A prediction becomes one fact for the predicted class and, when a
        confidence field is configured, a second for the probability. Two facts
        rather than one structured value because CDFS addresses a fact by
        coordinate and gives it a single value -- so a probability that belongs
        to a class is its own field, and inherits validation and lineage like
        any other.
        """
        missing = [p.prediction_id for p in predictions if not p.source_fact_ids]
        if missing:
            raise ValueError(
                f"{len(missing)} prediction(s) carry no source_fact_ids and cannot be "
                "written back; without them CDFS has nothing to attach the value to"
            )

        facts: list[dict[str, Any]] = []
        for prediction in predictions:
            facts.append(self._fact(prediction, field, prediction.outcome))
            if confidence_field is not None and prediction.probability is not None:
                facts.append(
                    self._fact(prediction, confidence_field, prediction.probability)
                )
        if not facts:
            raise ValueError("no predictions to write")
        return self._post(
            f"/studies/{study_id}/derived",
            {"facts": facts, "source_system": source_system},
        )

    @staticmethod
    def _fact(prediction: Prediction, field: str, value: Any) -> dict[str, Any]:
        """One CDFS fact from one prediction.

        The window is carried in ``source_record_ref`` rather than the
        coordinate, because a CDFS coordinate addresses a visit and this
        addresses an interval. Keeping the bounds in the reference means the
        precise window is recoverable without CDFS having to learn what a window
        is -- which is the same shape as a source reference naming a file and a
        row.
        """
        return {
            "subject_id": prediction.subject_id,
            "field": field,
            "value": value,
            "transform_id": f"{prediction.model_name}@{prediction.model_version}",
            "derived_from": list(prediction.source_fact_ids),
            "source_record_ref": ";".join(
                (
                    f"prediction={prediction.prediction_id}",
                    f"features={','.join(prediction.feature_ids)}",
                    f"windows={','.join(prediction.source_window_ids)}",
                    f"training_run={prediction.training_run_id}",
                    f"feature_set={prediction.feature_set_version}",
                    f"start={prediction.window_start.isoformat()}",
                    f"end={prediction.window_end.isoformat()}",
                )
            ),
            "recorded_at": prediction.created_at.isoformat()
            if prediction.created_at
            else None,
        }

    # ── transport ────────────────────────────────────────────────────────────

    def _get(self, path: str) -> dict[str, Any]:
        return self._send("GET", path, None)

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._send("POST", path, body)

    def _send(self, method: str, path: str, body: dict[str, Any] | None):
        url = urllib.parse.urljoin(self.base_url.rstrip("/") + "/", path.lstrip("/"))
        request = urllib.request.Request(
            url,
            method=method,
            data=json.dumps(body).encode("utf-8") if body is not None else None,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            # HTTPError is itself a response and holds the socket, so it is read
            # and closed rather than only read. Left open it surfaces later as a
            # ResourceWarning attributed to whatever happened to be running.
            payload: dict[str, Any] = {}
            with exc, contextlib.suppress(ValueError, OSError):
                payload = json.loads(exc.read() or b"{}")
            error = payload.get("error", {})
            raise CDFSError(
                exc.code,
                error.get("code", "http_error"),
                error.get("message", exc.reason or "request failed"),
            ) from exc
