"""Finding out that a prediction is no longer answerable.

CDFS corrects a value by superseding it, never by editing it, and its cascade
walks the derivation graph to find everything computed from what changed. For
its own derived fields it recomputes them. For a model field it cannot: the
engine has no model, and its impact report says so, naming the coordinate and
telling whoever produced the value to recompute it and write it back.

This is the other end of that message. A prediction names the CDFS facts it was
computed from, so asking whether it is still current means asking whether those
facts are still the ones in force. If any has been superseded, the prediction
was computed from a value the study has since retracted, and it should not be
read as current -- however plausible it still looks.

Nothing here decides what to do about it. It reports which predictions are
stale and exactly which input changed under each, so a recomputation can be run
and the result written back as a replacement rather than as a second opinion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from physioml.cdfs.client import CDFSClient


@dataclass(frozen=True)
class SupersededInput:
    """One fact a prediction rested on, and the fact that replaced it."""

    fact_id: str
    replaced_by: str
    field: str
    was: Any
    now: Any

    def describe(self) -> str:
        return f"{self.field} {self.was!r} -> {self.now!r}"


@dataclass(frozen=True)
class StalePrediction:
    """A model value whose inputs have moved underneath it."""

    fact_id: str
    subject_id: str
    field: str
    value: Any
    transform_id: str
    superseded: tuple[SupersededInput, ...]
    current_inputs: tuple[str, ...]
    """``derived_from`` with each superseded fact replaced by the live one.

    What a recomputation should cite, so the new prediction rests on the values
    in force rather than repeating the retracted ones.
    """

    @property
    def reason(self) -> str:
        """The audit sentence for the replacement, naming what changed.

        CDFS requires a reason on any fact that supersedes another, and "model
        rerun" would satisfy the rule while telling a reviewer nothing. The
        input that moved is the answer to the question the reviewer is asking.
        """
        changes = "; ".join(item.describe() for item in self.superseded)
        return f"recomputed after upstream correction: {changes}"


def _replacement(
    client: CDFSClient, fact_id: str
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """The old fact and the one now in force, or None if it has not moved.

    The supersession chain runs oldest to newest and contains the fact itself,
    so one request answers all three questions -- whether it moved, what it
    moved to, and what it used to say. Fetching the old value separately would
    be a second request for something already in hand.
    """
    history = client.lineage(fact_id).get("history") or []
    current = history[-1] if history else None
    if current is None or current.get("fact_id") == fact_id:
        return None
    empty: dict[str, Any] = {}
    was = next((f for f in history if f.get("fact_id") == fact_id), empty)
    return was, current


def check_prediction(client: CDFSClient, fact: dict[str, Any]) -> StalePrediction | None:
    """Whether one model fact still rests on the values in force."""
    inputs = list(fact.get("derived_from") or ())
    superseded: list[SupersededInput] = []
    current: list[str] = []

    for input_id in inputs:
        moved = _replacement(client, input_id)
        if moved is None:
            current.append(input_id)
            continue
        was, now = moved
        superseded.append(
            SupersededInput(
                fact_id=input_id,
                replaced_by=now["fact_id"],
                field=now.get("coordinate", {}).get("field", "?"),
                was=was.get("value"),
                now=now.get("value"),
            )
        )
        current.append(now["fact_id"])

    if not superseded:
        return None
    return StalePrediction(
        fact_id=fact["fact_id"],
        subject_id=fact.get("coordinate", {}).get("subject_id", "?"),
        field=fact.get("coordinate", {}).get("field", "?"),
        value=fact.get("value"),
        transform_id=fact.get("transform_id") or "",
        superseded=tuple(superseded),
        current_inputs=tuple(current),
    )


def stale_predictions(
    client: CDFSClient, study_id: str, subject_id: str
) -> list[StalePrediction]:
    """Every model value in force for a subject that its inputs have outrun.

    Only values in force are examined. A prediction that has already been
    replaced is not stale, it is history, and reporting it would ask for the
    same recomputation twice.
    """
    fields = set(client.model_fields(study_id))
    if not fields:
        return []

    stale: list[StalePrediction] = []
    for fact in client.subject_values(study_id, subject_id):
        if fact.get("coordinate", {}).get("field") not in fields:
            continue
        found = check_prediction(client, fact)
        if found is not None:
            stale.append(found)
    return stale
