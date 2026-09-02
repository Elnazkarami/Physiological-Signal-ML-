"""Whether a difference between two configurations is one.

Most of the numbers in this project are differences: a device adds 0.010, a
sensor removes 0.054, calibration takes 0.019 off an error. Reported as bare
means they all read alike, and they are not alike — some are visible in every
participant and some are one person moving.

The unit of resampling here is the **participant**, not the window. Windows from
one person are not independent observations of anything, and an interval built
from them would be far too narrow: 8,057 rows look like a large sample and are
fifteen people. Fifteen is the sample size, and an interval that says so is
usually wide enough to be uncomfortable, which is the point.

Comparisons are **paired**. Both configurations are scored on the same
participants under the same folds, so the difference is taken within each
person before anything is averaged. That removes the between-participant
variance, which is the largest term here — subjects differ from each other far
more than configurations differ on a subject.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physioml.evaluation.run import Evaluation


@dataclass(frozen=True)
class PairedDifference:
    """One metric, two configurations, measured within each participant."""

    metric: str
    left: str
    right: str
    per_subject: dict[str, float]
    """``right`` minus ``left``, positive meaning the right one scored higher."""

    mean: float
    interval: tuple[float, float]
    """Percentile bootstrap over participants, 95% by default."""

    better: int
    worse: int
    tied: int

    @property
    def n(self) -> int:
        return len(self.per_subject)

    @property
    def crosses_zero(self) -> bool:
        """Whether the interval admits no difference at all."""
        return self.interval[0] <= 0.0 <= self.interval[1]

    def verdict(self) -> str:
        """A sentence that does not overstate what the interval supports."""
        direction = "higher" if self.mean > 0 else "lower"
        if self.crosses_zero:
            return (
                f"{self.right} is {abs(self.mean):.3f} {direction} on average, but the "
                f"interval [{self.interval[0]:+.3f}, {self.interval[1]:+.3f}] includes "
                f"zero: {self.better} of {self.n} participants improved. This is a "
                "description of the sample, not evidence of a difference."
            )
        return (
            f"{self.right} is {abs(self.mean):.3f} {direction}, interval "
            f"[{self.interval[0]:+.3f}, {self.interval[1]:+.3f}], improving "
            f"{self.better} of {self.n} participants."
        )

    def line(self) -> str:
        flag = "  " if not self.crosses_zero else " ~"
        return (
            f"{self.right:26} {self.mean:+7.3f}  "
            f"[{self.interval[0]:+.3f}, {self.interval[1]:+.3f}]  "
            f"{self.better:2d}/{self.n:2d} better{flag}"
        )


def _scores(evaluation: Evaluation, metric: str) -> dict[str, float]:
    if metric == "balanced_accuracy":
        return {s: v for f in evaluation.folds for s, v in f.per_subject.items()}
    if metric == "auc":
        return {s: v for f in evaluation.folds for s, v in f.per_subject_auc.items()}
    if metric == "kappa":
        return {s: v for f in evaluation.folds for s, v in f.per_subject_kappa.items()}
    raise ValueError(f"unknown per-subject metric {metric!r}")


def paired_difference(
    left: Evaluation,
    right: Evaluation,
    *,
    metric: str = "balanced_accuracy",
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedDifference:
    """Compare two evaluations participant by participant.

    Only participants both evaluations scored are used. A configuration that
    could not score somebody is not thereby better or worse on them -- that
    absence is a coverage question, reported separately, and folding it in here
    would let a pipeline improve its average by failing on hard cases.
    """
    a, b = _scores(left, metric), _scores(right, metric)
    shared = sorted(
        set(a) & set(b), key=lambda s: int("".join(filter(str.isdigit, s)) or 0)
    )
    if not shared:
        raise ValueError("the two evaluations share no scored participant")

    differences = {s: b[s] - a[s] for s in shared}
    values = np.array([differences[s] for s in shared])

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[draws].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    interval = (
        float(np.quantile(means, tail)),
        float(np.quantile(means, 1.0 - tail)),
    )

    return PairedDifference(
        metric=metric,
        left=left.model_name,
        right=right.model_name,
        per_subject=differences,
        mean=float(values.mean()),
        interval=interval,
        better=int(np.sum(values > 0)),
        worse=int(np.sum(values < 0)),
        tied=int(np.sum(values == 0)),
    )


def table(differences: list[PairedDifference], baseline: str) -> str:
    """Several comparisons against one reference, as a block."""
    lines = [
        f"against {baseline}, per participant:",
        f"{'configuration':26} {'mean':>7}  {'95% interval':>18}  improved",
        "-" * 74,
    ]
    lines.extend(d.line() for d in differences)
    lines.append("")
    lines.append("~ marks an interval that includes zero.")
    return "\n".join(lines)
