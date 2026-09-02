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
    kappa: float = 0.0
    """Cohen's kappa: agreement corrected for what chance would give.

    The metric sleep staging is reported in, and the reason it is here. Five
    stages are unevenly distributed -- N2 is 45% of a night and N1 is 6% -- so
    raw agreement flatters any model, and published automatic scoring is
    compared on kappa rather than accuracy."""

    per_class: dict[str, float] = field(default_factory=dict)
    """Recall for each class. On a five-stage problem the macro average hides
    which stage a model cannot see, and it is nearly always N1."""

    labels: tuple[str, ...] = ()
    """Class order, so the confusion matrix can be read."""

    per_subject: dict[str, float] = field(default_factory=dict)
    """Balanced accuracy for each held-out subject."""

    per_subject_auc: dict[str, float] = field(default_factory=dict)
    """Area under the ROC curve for each held-out subject.

    Beside the balanced accuracy because the two answer different questions and
    the difference is a diagnosis. Balanced accuracy asks whether the labels a
    model emits are right at the threshold it was scored at; AUC asks whether
    it ranks that person's stressed windows above their calm ones at all. A
    participant scoring 0.500 balanced accuracy and 0.900 AUC has not been
    failed by the model, they have been failed by the operating point -- their
    windows are ordered correctly and all of them sit on one side of the
    boundary. Reporting only the first calls both cases "chance"."""

    per_subject_stated: dict[str, float] = field(default_factory=dict)
    """Mean stated probability per subject, against which ``per_subject_rate``
    is the truth. Two numbers that make an over- or under-confident participant
    visible without a reliability plot."""

    per_subject_rate: dict[str, float] = field(default_factory=dict)

    per_subject_kappa: dict[str, float] = field(default_factory=dict)
    """Cohen's kappa per held-out subject.

    The per-participant form of the metric sleep staging is reported in, and
    what lets a difference between two montages carry an interval rather than
    being two point estimates set beside each other."""

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
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        recall_score,
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
    per_subject_auc: dict[str, float] = {}
    per_subject_stated: dict[str, float] = {}
    per_subject_rate: dict[str, float] = {}
    per_subject_kappa: dict[str, float] = {}
    if subjects is not None:
        for subject in np.unique(subjects):
            rows = subjects == subject
            if len(np.unique(truth[rows])) < 2:
                # A held-out participant with one class present has no balanced
                # accuracy worth the name; reporting 0 or 1 would be noise.
                continue
            name = str(subject)
            per_subject[name] = float(balanced_accuracy_score(truth[rows], predicted[rows]))
            per_subject_rate[name] = float(np.mean(truth[rows] == 1))
            per_subject_kappa[name] = float(cohen_kappa_score(truth[rows], predicted[rows]))
            if probability is not None and binary:
                per_subject_auc[name] = float(roc_auc_score(truth[rows], probability[rows]))
                per_subject_stated[name] = float(np.mean(probability[rows]))

    present = np.unique(np.concatenate([truth, predicted]))
    recalls = recall_score(truth, predicted, labels=present, average=None, zero_division=0)
    counts = np.array([np.sum(truth == c) for c in present], dtype=float)

    return Scores(
        balanced_accuracy=float(balanced_accuracy_score(truth, predicted)),
        f1_macro=float(f1_score(truth, predicted, average="macro", zero_division=0)),
        roc_auc=roc,
        pr_auc=pr,
        brier=brier,
        ece=expected,
        accuracy=float(accuracy_score(truth, predicted)),
        # What a constant predictor would score. For a 0/1 problem that is the
        # share of the positive class; for anything else it is the share of the
        # commonest one. Testing `binary` alone was not enough: two classes
        # named "W" and "N2" are binary, and `truth == 1` is false for every
        # row of them, which reported a rate of zero.
        positive_rate=_constant_rate(truth, counts),
        n=int(truth.size),
        kappa=float(cohen_kappa_score(truth, predicted)),
        per_class={str(c): float(r) for c, r in zip(present, recalls, strict=True)},
        labels=tuple(str(c) for c in present),
        per_subject=per_subject,
        per_subject_auc=per_subject_auc,
        per_subject_stated=per_subject_stated,
        per_subject_rate=per_subject_rate,
        per_subject_kappa=per_subject_kappa,
        confusion=tuple(
            tuple(int(v) for v in row)
            for row in confusion_matrix(truth, predicted, labels=present)
        ),
    )


def _constant_rate(truth: np.ndarray, counts: np.ndarray) -> float:
    """The accuracy the best constant answer would get."""
    if counts.sum() == 0:
        return 0.0
    labelled_one_zero = set(np.unique(truth).tolist()) <= {0, 1}
    if labelled_one_zero:
        return float(np.mean(truth == 1))
    return float(counts.max() / counts.sum())


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
    if probability.size == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index, (low, high) in enumerate(itertools.pairwise(edges)):
        # The first bin is closed at both ends. Every other is half-open on the
        # left, so a prediction sits in exactly one. Leaving the first open too
        # put predictions of exactly zero in no bin at all, and they were
        # dropped from the average rather than counted: a model answering "0.00
        # probability of stress" to every window of a set that is 22% stressed
        # scored a calibration error of 0.000. Isotonic regression clips to
        # [0, 1] and reaches the endpoint more often than an uncalibrated
        # model does, so the omission flattered calibration specifically.
        inside = (
            (probability >= low) & (probability <= high)
            if index == 0
            else (probability > low) & (probability <= high)
        )
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
        "kappa_mean": float(np.mean([f.kappa for f in folds])),
        "kappa_sd": float(np.std([f.kappa for f in folds])),
        "kappa_min": float(np.min([f.kappa for f in folds])),
        "accuracy_mean": float(np.mean([f.accuracy for f in folds])),
        "n_total": float(sum(f.n for f in folds)),
    }
    classes = sorted({c for f in folds for c in f.per_class})
    for name in classes:
        found = [f.per_class[name] for f in folds if name in f.per_class]
        if found:
            summary[f"recall_{name}"] = float(np.mean(found))
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
    per_auc = {s: v for f in folds for s, v in f.per_subject_auc.items()}
    if per_auc:
        summary["worst_subject_auc"] = min(per_auc.values())
        summary["per_subject_auc_mean"] = float(np.mean(list(per_auc.values())))
    per_kappa = {s: v for f in folds for s, v in f.per_subject_kappa.items()}
    if per_kappa:
        summary["worst_subject_kappa"] = min(per_kappa.values())
    return summary
