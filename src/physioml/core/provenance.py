"""Identity and lineage, shared by everything the pipeline produces.

Every artifact here — a recording, a window, a feature, a prediction — is
identified by a hash over the fields that define it rather than by a counter.
That choice is inherited from CDFS and it buys the same three things:

* the same inputs and the same transformation produce the same identifier, so
  a recomputation that changes nothing is visibly a no-op;
* an artifact cannot be edited after the fact without its identifier ceasing to
  match its contents;
* two machines processing the same recording agree on identifiers without
  coordinating, which is what makes a training run reproducible somewhere else.

Lineage is expressed as identifiers pointing backwards — a prediction names the
features it used, a feature names its window, a window names its recording. The
graph is therefore reconstructible from the artifacts alone, with no separate
index that could disagree with them.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any


def content_id(prefix: str, payload: dict[str, Any]) -> str:
    """A stable identifier for an artifact, from the fields that define it.

    The prefix is carried in the identifier rather than inferred, so a stray
    identifier in a log or a payload says what kind of thing it points at.
    """
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_plain)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:32]}"


def _plain(value: Any) -> Any:
    """Render values json cannot, in a form that hashes consistently."""
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def utc(moment: datetime | None = None) -> datetime:
    """A timezone-aware UTC timestamp.

    Naive timestamps are refused rather than assumed to be local: a recording
    and a prediction that disagree about what "14:30" means are the kind of
    defect that only appears when the data crosses a timezone.
    """
    if moment is None:
        return datetime.now(UTC)
    if moment.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return moment.astimezone(UTC)
