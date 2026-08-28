"""Turning subjects into the table a model is trained on, without losing the thread.

A feature matrix is where provenance usually dies. Rows become anonymous, and by
the time a model is scored nobody can say which window a row came from, which
subject, or which quality verdict it carried. So every row here keeps its
subject, its window identifier, and the identifiers of the features that formed
it — the same chain :mod:`physioml.core` maintains, arranged as a table.

Keeping the subject on the row is not only for provenance. It is what makes
subject-wise validation *possible*: a split cannot hold participants apart if
the matrix has forgotten who they are.

Rows with a missing feature are dropped rather than imputed, and the count is
reported. A window whose pulse was rejected by quality control has no pulse
features, and filling that gap with a cohort mean would state that the
participant's heart rate was average at a moment when it was not measured.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from physioml.core.feature import Feature
from physioml.io.wesad import WESAD
from physioml.peripheral.chest import (
    CHEST_EXTRACTORS,
    CHEST_FEATURE_SET,
    CHEST_FEATURE_SET_VERSION,
    CHEST_POLICY,
    assess_chest,
)
from physioml.peripheral.features import FEATURE_SET_VERSION, extract
from physioml.peripheral.qc import DEFAULT_POLICY, QCPolicy, assess
from physioml.peripheral.windowing import epochs


@dataclass(frozen=True, slots=True)
class FeatureTable:
    """A feature matrix that still knows where each row came from."""

    feature_names: tuple[str, ...]
    values: np.ndarray
    """``(n_rows, n_features)``, columns in ``feature_names`` order."""

    subjects: np.ndarray
    """Subject identifier per row — the grouping every split must respect."""

    labels: np.ndarray
    window_ids: tuple[str, ...]
    window_starts: np.ndarray
    """Seconds from the start of the recording, per row.

    Kept because a table that has forgotten *when* each row happened cannot be
    split in time, and these windows overlap by 55 of their 60 seconds. Any
    within-subject split made at random puts almost the same minute on both
    sides of it.
    """

    feature_set_version: str
    qc_policy_version: str
    dropped_incomplete: int = 0
    """Rows discarded for a missing feature, rather than imputed."""

    qc_codes: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.values.shape[0])

    @property
    def subject_ids(self) -> list[str]:
        return sorted(set(self.subjects.tolist()), key=lambda s: int(s.lstrip("S")))

    def binary(self, positive: str = "stress") -> np.ndarray:
        """Labels as one-vs-rest, for the task WESAD is usually posed as."""
        return (self.labels == positive).astype(int)

    def select(self, names: Sequence[str]) -> FeatureTable:
        """The same rows, restricted to these feature columns.

        Rows, subjects, labels and window identifiers are carried through
        unchanged, so a model fitted on the subset is answerable for exactly
        the windows the full table was. Column order follows ``names``.
        """
        missing = [n for n in names if n not in self.feature_names]
        if missing:
            raise KeyError(f"not in this table: {', '.join(sorted(missing))}")
        if not names:
            raise ValueError("a table with no features cannot be fitted")

        index = {name: i for i, name in enumerate(self.feature_names)}
        columns = [index[n] for n in names]
        return replace(
            self,
            feature_names=tuple(names),
            values=self.values[:, columns],
        )

    def counts(self) -> dict[str, int]:
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist(), strict=True))

    def summary(self) -> str:
        by_label = ", ".join(f"{k} {v}" for k, v in sorted(self.counts().items()))
        return (
            f"{len(self)} rows x {len(self.feature_names)} features, "
            f"{len(self.subject_ids)} subjects ({by_label})"
        )

    def save(self, path: Path | str) -> None:
        """Write the table and everything needed to interpret it."""
        path = Path(path)
        np.savez_compressed(
            path,
            values=self.values,
            subjects=self.subjects,
            labels=self.labels,
            feature_names=np.array(self.feature_names),
            window_ids=np.array(self.window_ids),
            window_starts=self.window_starts,
            meta=np.array(
                json.dumps(
                    {
                        "feature_set_version": self.feature_set_version,
                        "qc_policy_version": self.qc_policy_version,
                        "dropped_incomplete": self.dropped_incomplete,
                        "qc_codes": self.qc_codes,
                    }
                )
            ),
        )

    @classmethod
    def load(cls, path: Path | str) -> FeatureTable:
        loaded = np.load(Path(path), allow_pickle=False)
        meta = json.loads(str(loaded["meta"]))
        return cls(
            feature_names=tuple(loaded["feature_names"].tolist()),
            values=loaded["values"],
            subjects=loaded["subjects"],
            labels=loaded["labels"],
            window_ids=tuple(loaded["window_ids"].tolist()),
            window_starts=(
                loaded["window_starts"]
                if "window_starts" in loaded
                # Written before row times were recorded. Left empty rather
                # than filled with an index, which would silently become a
                # plausible-looking timeline.
                else np.full(len(loaded["subjects"]), np.nan)
            ),
            feature_set_version=meta["feature_set_version"],
            qc_policy_version=meta["qc_policy_version"],
            dropped_incomplete=meta["dropped_incomplete"],
            qc_codes=meta["qc_codes"],
        )


def _version_of(device: str) -> str:
    """What produced these columns, in one string a model can be pinned to.

    A fused table is not version 1.3 of the wrist feature set and not 1.0 of
    the chest one; it is both, and a model artifact recording only one of them
    would accept a table it has never seen.
    """
    if device == "wrist":
        return FEATURE_SET_VERSION
    if device == "chest":
        return f"chest-{CHEST_FEATURE_SET_VERSION}"
    return f"{FEATURE_SET_VERSION}+chest-{CHEST_FEATURE_SET_VERSION}"


def _policy_version(device: str, policy: QCPolicy, chest_policy: QCPolicy) -> str:
    if device == "wrist":
        return policy.version
    if device == "chest":
        return chest_policy.version
    return f"{policy.version}+{chest_policy.version}"


def _wrist(epoch, policy: QCPolicy) -> tuple[list[Feature], dict[str, tuple[str, ...]]]:
    verdict = assess(epoch, policy)
    return extract(epoch, verdict, policy), verdict.codes


def _chest(epoch, policy: QCPolicy) -> tuple[list[Feature], dict[str, tuple[str, ...]]]:
    verdict = assess_chest(epoch, policy)
    found = extract(
        epoch,
        verdict,
        policy,
        extractors=CHEST_EXTRACTORS,
        feature_set=CHEST_FEATURE_SET,
        feature_set_version=CHEST_FEATURE_SET_VERSION,
    )
    return found, {f"chest:{k}": v for k, v in verdict.codes.items()}


def build(
    archive: Path | str,
    *,
    subjects: list[str] | None = None,
    device: str = "wrist",
    length_seconds: float = 60.0,
    stride_seconds: float = 5.0,
    policy: QCPolicy = DEFAULT_POLICY,
    chest_policy: QCPolicy = CHEST_POLICY,
    min_coverage: float = 0.9,
    progress: bool = False,
) -> FeatureTable:
    """Read every subject, window it, run quality control, extract features.

    Subjects are processed one at a time and released, because the signals do
    not fit in memory together and the features do so comfortably.

    ``device`` is ``wrist``, ``chest``, or ``both``. **Both** reads each
    subject twice and joins the two devices on the window interval, not on
    position in a list: the recordings are the same length and the same stride
    produces the same intervals, but pairing by index would be an assumption
    where pairing by time is a fact. A window either device could not produce
    features for is left with the other device's features and dropped later if
    the coverage rule says so.

    ``min_coverage`` is the share of rows a feature must appear in to become a
    column. Below it the feature is dropped; above it, the rows missing it are.
    """
    if device not in ("wrist", "chest", "both"):
        raise ValueError(f"device must be wrist, chest or both, not {device!r}")

    source = WESAD(archive)
    chosen = subjects or source.subjects()

    rows: list[dict[str, Feature]] = []
    row_subjects: list[str] = []
    row_labels: list[str] = []
    row_windows: list[str] = []
    row_starts: list[float] = []
    codes: dict[str, int] = {}

    def count(found: dict[str, tuple[str, ...]]) -> None:
        for signal, reasons in found.items():
            for code in reasons:
                key = f"{signal}:{code}"
                codes[key] = codes.get(key, 0) + 1

    for subject_id in chosen:
        by_interval: dict[tuple[float, float], dict[str, Feature]] = {}
        keeping: dict[tuple[float, float], tuple[str, str, float]] = {}
        order: list[tuple[float, float]] = []

        for which, extract_one, this_policy in (
            ("wrist", _wrist, policy),
            ("chest", _chest, chest_policy),
        ):
            if device not in (which, "both"):
                continue
            data = source.read(subject_id, device=which)
            for epoch in epochs(
                data, length_seconds=length_seconds, stride_seconds=stride_seconds
            ):
                if not epoch.labelled:
                    continue
                features, found = extract_one(epoch, this_policy)
                count(found)
                if not features:
                    continue
                interval = (epoch.start_seconds, epoch.duration_seconds)
                if interval not in by_interval:
                    by_interval[interval] = {}
                    order.append(interval)
                    keeping[interval] = (
                        epoch.label or "",
                        next(iter(epoch.windows.values())).window_id,
                        epoch.start_seconds,
                    )
                by_interval[interval].update({f.qualified_name: f for f in features})
            del data

        for interval in order:
            label, window_id, started = keeping[interval]
            rows.append(by_interval[interval])
            row_subjects.append(subject_id)
            row_labels.append(label)
            row_windows.append(window_id)
            row_starts.append(started)
        if progress:
            print(f"  {subject_id}: {len(rows)} rows so far", flush=True)

    if not rows:
        raise ValueError("no labelled windows produced any features")

    # Columns first, by coverage; rows second. Taking the intersection of every
    # row's features instead looks equivalent and is not: a handful of windows
    # whose pulse quality control rejected would delete every pulse feature from
    # the entire cohort. Measured on WESAD that was 34 windows in 8,091 costing
    # 6 of 28 features. A feature carried by almost every row is kept, and the
    # few rows lacking it are dropped and counted.
    coverage: dict[str, int] = {}
    for row in rows:
        for name in row:
            coverage[name] = coverage.get(name, 0) + 1
    threshold = min_coverage * len(rows)
    names = sorted(n for n, seen in coverage.items() if seen >= threshold)
    if not names:
        raise ValueError(
            f"no feature appears in {min_coverage:.0%} of rows; the cohort has "
            "nothing in common to train on"
        )
    complete = [i for i, r in enumerate(rows) if all(n in r for n in names)]

    values = np.array([[rows[i][n].value for n in names] for i in complete], dtype=float)
    return FeatureTable(
        feature_names=tuple(names),
        values=values,
        subjects=np.array([row_subjects[i] for i in complete]),
        labels=np.array([row_labels[i] for i in complete]),
        window_ids=tuple(row_windows[i] for i in complete),
        window_starts=np.array([row_starts[i] for i in complete], dtype=float),
        feature_set_version=_version_of(device),
        qc_policy_version=_policy_version(device, policy, chest_policy),
        dropped_incomplete=len(rows) - len(complete),
        qc_codes=codes,
    )
