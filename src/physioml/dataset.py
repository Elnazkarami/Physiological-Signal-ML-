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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from physioml.core.feature import Feature
from physioml.io.wesad import WESAD
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
            feature_set_version=meta["feature_set_version"],
            qc_policy_version=meta["qc_policy_version"],
            dropped_incomplete=meta["dropped_incomplete"],
            qc_codes=meta["qc_codes"],
        )


def build(
    archive: Path | str,
    *,
    subjects: list[str] | None = None,
    length_seconds: float = 60.0,
    stride_seconds: float = 5.0,
    policy: QCPolicy = DEFAULT_POLICY,
    min_coverage: float = 0.9,
    progress: bool = False,
) -> FeatureTable:
    """Read every subject, window it, run quality control, extract features.

    Subjects are processed one at a time and released, because the signals do
    not fit in memory together and the features do so comfortably.

    ``min_coverage`` is the share of rows a feature must appear in to become a
    column. Below it the feature is dropped; above it, the rows missing it are.
    """
    source = WESAD(archive)
    chosen = subjects or source.subjects()

    rows: list[dict[str, Feature]] = []
    row_subjects: list[str] = []
    row_labels: list[str] = []
    row_windows: list[str] = []
    codes: dict[str, int] = {}

    for subject_id in chosen:
        data = source.read(subject_id)
        for epoch in epochs(
            data, length_seconds=length_seconds, stride_seconds=stride_seconds
        ):
            if not epoch.labelled:
                continue
            verdict = assess(epoch, policy)
            for signal, found in verdict.codes.items():
                for code in found:
                    key = f"{signal}:{code}"
                    codes[key] = codes.get(key, 0) + 1

            features = extract(epoch, verdict, policy)
            if not features:
                continue
            rows.append({f.qualified_name: f for f in features})
            row_subjects.append(subject_id)
            row_labels.append(epoch.label or "")
            row_windows.append(next(iter(epoch.windows.values())).window_id)
        if progress:
            print(f"  {subject_id}: {len(rows)} rows so far", flush=True)
        del data

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
        feature_set_version=FEATURE_SET_VERSION,
        qc_policy_version=policy.version,
        dropped_incomplete=len(rows) - len(complete),
        qc_codes=codes,
    )
