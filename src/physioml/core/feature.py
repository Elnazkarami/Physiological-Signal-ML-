"""One computed quantity, and the exact thing it was computed from.

A feature carries the version of the feature set that produced it. That is not
bookkeeping: it is what makes §13.1 of the plan possible. When a feature
algorithm is corrected — a band definition fixed, a filter changed — every
feature carrying the old version is identifiable, and so is every prediction
that used one. Without the version on the artifact, a methodological correction
is untraceable and the only honest response is to recompute everything.

Features are values, not arrays. A feature set that wants to emit a spectrum
emits one feature per band rather than one feature holding a vector, because
a downstream model selecting "alpha power" should be selecting a named thing
rather than an index into something.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from physioml.core.provenance import content_id


@dataclass(frozen=True, slots=True)
class Feature:
    """One named quantity computed from one or more windows."""

    subject_id: str
    name: str
    value: float
    feature_set: str
    feature_set_version: str
    source_window_ids: tuple[str, ...]
    """Usually one window; more when a feature spans them, e.g. a nightly mean."""

    unit: str | None = None
    channel: str | None = None
    """Which channel this came from, for channel-aware features. ``None`` for
    features that are channel-independent or aggregate across channels."""

    transform_id: str = ""
    """The specific computation, at the version that ran."""

    feature_id: str = ""

    @classmethod
    def create(cls, **kwargs: Any) -> Feature:
        kwargs.setdefault("source_window_ids", ())
        kwargs["source_window_ids"] = tuple(kwargs["source_window_ids"])
        feature = cls(**kwargs)
        if not feature.source_window_ids:
            raise ValueError(
                f"feature {feature.name!r} has no source window; a feature that "
                "cannot say what it was computed from is not traceable"
            )
        if not feature.feature_set_version:
            raise ValueError(
                f"feature {feature.name!r} has no feature_set_version; without it "
                "a later correction to this algorithm cannot find its outputs"
            )
        return replace(feature, feature_id=feature._identity())

    def _identity(self) -> str:
        return content_id(
            "feat",
            {
                "name": self.name,
                "value": float(self.value),
                "feature_set": self.feature_set,
                "feature_set_version": self.feature_set_version,
                "channel": self.channel,
                "source_window_ids": sorted(self.source_window_ids),
                "transform_id": self.transform_id,
            },
        )

    @property
    def qualified_name(self) -> str:
        """The name a model's feature schema refers to."""
        return f"{self.name}@{self.channel}" if self.channel else self.name

    def __str__(self) -> str:
        unit = f" {self.unit}" if self.unit else ""
        return f"{self.qualified_name}={self.value:g}{unit}"


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """The features presented to a model for one window, in a fixed order.

    Ordering is explicit rather than dictionary order, because a model trained
    on one column order and scored on another produces confident nonsense and
    nothing in the numbers looks wrong.
    """

    subject_id: str
    window_id: str
    names: tuple[str, ...]
    values: tuple[float, ...]
    feature_ids: tuple[str, ...]
    feature_set_version: str
    label: str | None = None

    @classmethod
    def of(cls, features: list[Feature], *, window_id: str, label: str | None = None):
        """Build a vector from features, ordered by name so it is reproducible."""
        if not features:
            raise ValueError("a feature vector needs at least one feature")
        versions = {f.feature_set_version for f in features}
        if len(versions) > 1:
            raise ValueError(
                f"features span feature-set versions {sorted(versions)}; mixing them "
                "makes the vector untraceable to a single algorithm"
            )
        subjects = {f.subject_id for f in features}
        if len(subjects) > 1:
            raise ValueError(f"features span subjects {sorted(subjects)}")
        ordered = sorted(features, key=lambda f: f.qualified_name)
        return cls(
            subject_id=ordered[0].subject_id,
            window_id=window_id,
            names=tuple(f.qualified_name for f in ordered),
            values=tuple(float(f.value) for f in ordered),
            feature_ids=tuple(f.feature_id for f in ordered),
            feature_set_version=versions.pop(),
            label=label,
        )

    def __len__(self) -> int:
        return len(self.names)
