"""What stops being true when quality control changes its mind.

The CDFS direction of the cascade is a correction upstream making a prediction
stale. This is the other one, and it starts inside PhysioML: a quality-control
policy is revised, windows that used to pass now fail, and everything computed
from them has to be found.

It is expressible for one reason, decided early and stated in
:mod:`physioml.core.window`: **a window's identity is the physical slice, not
the quality verdict on it.** Re-judging a window under a stricter policy does
not produce a different window, so the features that named it still name it,
and the graph can be walked. Had the verdict entered the identity, a revision
would have produced new window identifiers with nothing pointing at them, and
the features derived from the old ones would have referred to windows that no
longer existed — the same information, arranged so it could not be used.

Nothing here deletes anything. It reports what a revision reaches, in the order
a person would have to act on it: windows, then the features that came from
them, then the predictions that used those features.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from physioml.core.feature import Feature
from physioml.core.prediction import Prediction


@dataclass(frozen=True)
class Invalidated:
    """Everything a quality-control revision reaches, one layer at a time."""

    windows: tuple[str, ...]
    features: tuple[str, ...]
    predictions: tuple[str, ...]
    reasons: dict[str, tuple[str, ...]]
    """Why each window is now rejected -- the codes the new policy returned."""

    source_fact_ids: tuple[str, ...] = ()
    """The CDFS facts the affected predictions rest on.

    Carried because the recomputation has to be written back across the
    boundary, and a caller holding only PhysioML identifiers cannot say what to
    replace over there."""

    def __bool__(self) -> bool:
        return bool(self.windows)

    def summary(self) -> str:
        if not self.windows:
            return "no window is affected by this revision"
        codes = sorted({c for found in self.reasons.values() for c in found})
        return (
            f"{len(self.windows)} window(s) now rejected ({', '.join(codes)}); "
            f"{len(self.features)} feature(s) invalid; "
            f"{len(self.predictions)} prediction(s) stale"
        )


def invalidated_by(
    rejected: Mapping[str, Sequence[str]],
    features: Iterable[Feature],
    predictions: Iterable[Prediction] = (),
) -> Invalidated:
    """Walk a revision outwards from the windows it rejected.

    ``rejected`` maps a window identifier to the reason codes the new policy
    gave it. A feature is invalid if *any* of its source windows is rejected --
    not all of them: a feature computed across two windows, one of which turns
    out to be an artifact, is not partly right.
    """
    windows = set(rejected)
    if not windows:
        return Invalidated((), (), (), {})

    touched = [f for f in features if windows & set(f.source_window_ids)]
    invalid = {f.feature_id for f in touched}

    stale = [
        p
        for p in predictions
        if invalid & set(p.feature_ids) or windows & set(p.source_window_ids)
    ]

    return Invalidated(
        windows=tuple(sorted(windows)),
        features=tuple(sorted(invalid)),
        predictions=tuple(sorted(p.prediction_id for p in stale)),
        reasons={w: tuple(rejected[w]) for w in sorted(windows)},
        source_fact_ids=tuple(sorted({f for p in stale for f in p.source_fact_ids})),
    )
