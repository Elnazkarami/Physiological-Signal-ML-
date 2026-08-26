"""Scoring, chosen for a task where the classes are not balanced.

WESAD's stress windows are 22% of the labelled data, so plain accuracy is
misleading in the specific way that matters: a model that answers "not stressed"
every time scores 78% and has learned nothing. Every metric here is either
insensitive to that or reports it.

Two things are computed that a classification report usually omits, and both
are the point of the exercise.

**Calibration.** A probability that says 0.8 should be right about 80% of the
time. A model can rank well and still be badly calibrated, and a recovery or
stress score shown to a person is a probability being read literally, so the
Brier score and expected calibration error are reported alongside the ranking
metrics rather than instead of them.

**Per-subject spread.** A cohort mean hides that a model works for eleven people
and fails for four. For a wearable that is the whole question — someone is about
to put it on, and they are one person, not a mean — so the worst subject is
reported next to the average.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class Scores:
    """What one model achieved, per fold and across subjects."""

    balanced_accuracy: float
    f1_macro: float
    roc_auc: float | None
    pr_auc: float | None
    brier: float | None
    ece: float | None
    accuracy: float
    positive_rate: float
    """Share of the test set that is positive — the accuracy a constant
    predictor would beat, kept beside the metrics so it cannot be forgotten."""

    n: int
    per_subject: dict[str, float] = field(default_factory=dict)
    """Balanced accuracy for each held-out subject."""

    confusion: tuple[tuple[int, ...], ...] = ()

    @property
    def worst_subject(self) -> tuple[str, float] | None:
        if not self.per_subject:
            return None
        subject = min(self.per_subject, key=lambda s: self.per_subject[s])
        return subject, self.per_subject[subject]

    def summary(self) -> str:
        worst = self.worst_subject
        tail = f"  worst subject {worst[0]} {worst[1]:.3f}" if worst else ""
        auc = f"{self.roc_auc:.3f}" if self.roc_auc is not None else "n/a"
        return (
            f"balanced acc {self.balanced_accuracy:.3f}  F1 {self.f1_macro:.3f}  "
            f"AUC {auc}  n={self.n}{tail}"
        )


def score(
    truth: np.ndarray,
    predicted: np.ndarray,
    probability: np.ndarray | None = None,
    subjects: np.ndarray | None = None,
) -> Scores:
    """Every metric, from one set of predictions."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )

    truth = np.asarray(truth)
    predicted = np.asarray(predicted)
    binary = len(np.unique(truth)) <= 2

    roc = pr = brier = expected = None
    if probability is not None and binary and len(np.unique(truth)) == 2:
        roc = float(roc_auc_score(truth, probability))
        pr = float(average_precision_score(truth, probability))
        brier = float(brier_score_loss(truth, probability))
        expected = expected_calibration_error(truth, probability)

    per_subject: dict[str, float] = {}
    if subjects is not None:
        for subject in np.unique(subjects):
            rows = subjects == subject
            if len(np.unique(truth[rows])) < 2:
                # A held-out participant with one class present has no balanced
                # accuracy worth the name; reporting 0 or 1 would be noise.
                continue
            per_subject[str(subject)] = float(
                balanced_accuracy_score(truth[rows], predicted[rows])
            )

    return Scores(
        balanced_accuracy=float(balanced_accuracy_score(truth, predicted)),
        f1_macro=float(f1_score(truth, predicted, average="macro", zero_division=0)),
        roc_auc=roc,
        pr_auc=pr,
        brier=brier,
        ece=expected,
        accuracy=float(accuracy_score(truth, predicted)),
        positive_rate=float(np.mean(truth == 1)) if binary else float("nan"),
        n=int(truth.size),
        per_subject=per_subject,
        confusion=tuple(
            tuple(int(v) for v in row) for row in confusion_matrix(truth, predicted)
        ),
    )


def expected_calibration_error(
    truth: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    """How far stated confidence sits from observed frequency.

    Predictions are grouped by confidence and each group's mean probability is
    compared with the share of it that was actually positive, weighted by group
    size. Zero means a model claiming 70% is right 70% of the time.
    """
    truth = np.asarray(truth, dtype=float)
    probability = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in itertools.pairwise(edges):
        inside = (probability > low) & (probability <= high)
        if not inside.any():
            continue
        total += inside.mean() * abs(truth[inside].mean() - probability[inside].mean())
    return float(total)


def aggregate(folds: list[Scores]) -> dict[str, float]:
    """Across folds, reporting the spread rather than only the middle."""
    if not folds:
        return {}
    balanced = np.array([f.balanced_accuracy for f in folds])
    per_subject = {s: v for f in folds for s, v in f.per_subject.items()}
    summary = {
        "balanced_accuracy_mean": float(balanced.mean()),
        "balanced_accuracy_sd": float(balanced.std()),
        "balanced_accuracy_min": float(balanced.min()),
        "f1_macro_mean": float(np.mean([f.f1_macro for f in folds])),
        "n_total": float(sum(f.n for f in folds)),
    }
    for name, values in (
        ("roc_auc", [f.roc_auc for f in folds]),
        ("pr_auc", [f.pr_auc for f in folds]),
        ("brier", [f.brier for f in folds]),
        ("ece", [f.ece for f in folds]),
    ):
        present = [v for v in values if v is not None]
        if present:
            summary[f"{name}_mean"] = float(np.mean(present))
    if per_subject:
        worst = min(per_subject, key=lambda s: per_subject[s])
        summary["worst_subject_balanced_accuracy"] = per_subject[worst]
    return summary
