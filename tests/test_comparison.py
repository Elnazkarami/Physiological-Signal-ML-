"""Whether a difference between two configurations is one.

The failure guarded against is reading a mean difference as a result. Most of
these tests construct a case where the answer is known -- an effect present in
everybody, an effect present in one person, no effect at all -- and check that
the interval says so.
"""

from __future__ import annotations

import numpy as np
import pytest

from physioml.evaluation.comparison import paired_difference, table
from physioml.evaluation.metrics import Scores
from physioml.evaluation.run import Evaluation


def evaluation(name: str, per_subject: dict[str, float]) -> Evaluation:
    """An evaluation carrying only the per-subject scores under test."""
    folds = tuple(
        Scores(
            balanced_accuracy=value,
            f1_macro=value,
            roc_auc=None,
            pr_auc=None,
            brier=None,
            ece=None,
            accuracy=value,
            positive_rate=0.22,
            n=100,
            per_subject={subject: value},
            per_subject_auc={subject: value},
        )
        for subject, value in per_subject.items()
    )
    return Evaluation(name, "stress", ("a",), folds, ())


SUBJECTS = [f"S{n}" for n in range(2, 17)]


def test_an_effect_present_in_everybody_excludes_zero():
    left = evaluation("left", dict.fromkeys(SUBJECTS, 0.80))
    right = evaluation("right", dict.fromkeys(SUBJECTS, 0.85))
    found = paired_difference(left, right)
    assert found.mean == pytest.approx(0.05)
    assert not found.crosses_zero
    assert found.better == len(SUBJECTS)


def test_an_effect_carried_by_one_participant_does_not():
    """The case the mean cannot distinguish from the one above."""
    rng = np.random.default_rng(0)
    base = {s: float(rng.uniform(0.7, 0.9)) for s in SUBJECTS}
    moved = dict(base)
    moved[SUBJECTS[0]] += 0.75  # one person, a large jump

    found = paired_difference(evaluation("left", base), evaluation("right", moved))
    assert found.mean == pytest.approx(0.05, abs=0.01), "same mean as the test above"
    assert found.crosses_zero, "and it should not be believed"
    assert found.better == 1


def test_no_effect_reports_no_effect():
    rng = np.random.default_rng(1)
    base = {s: float(rng.uniform(0.7, 0.9)) for s in SUBJECTS}
    jitter = {s: v + float(rng.normal(0, 0.005)) for s, v in base.items()}
    found = paired_difference(evaluation("left", base), evaluation("right", jitter))
    assert found.crosses_zero


def test_the_comparison_is_paired_not_pooled():
    """Between-participant variance is the largest term; pairing removes it.

    Scores spread from 0.5 to 1.0 with a constant 0.02 gap. Unpaired, the
    spread swamps the gap. Paired, the gap is all that is left.
    """
    base = {s: 0.5 + 0.03 * i for i, s in enumerate(SUBJECTS)}
    shifted = {s: v + 0.02 for s, v in base.items()}
    found = paired_difference(evaluation("left", base), evaluation("right", shifted))
    assert found.interval[0] == pytest.approx(0.02, abs=1e-6)
    assert found.interval[1] == pytest.approx(0.02, abs=1e-6)
    assert not found.crosses_zero


def test_only_participants_both_configurations_scored_are_used():
    """Otherwise a pipeline improves its average by failing on hard cases."""
    left = evaluation("left", {"S2": 0.6, "S3": 0.6, "S16": 0.3})
    right = evaluation("right", {"S2": 0.7, "S3": 0.7})
    found = paired_difference(left, right)
    assert found.n == 2
    assert "S16" not in found.per_subject
    assert found.mean == pytest.approx(0.1)


def test_two_evaluations_with_nobody_in_common_are_refused():
    left = evaluation("left", {"S2": 0.6})
    right = evaluation("right", {"S3": 0.7})
    with pytest.raises(ValueError, match="share no scored participant"):
        paired_difference(left, right)


def test_the_interval_is_reproducible_from_its_seed():
    rng = np.random.default_rng(3)
    base = {s: float(rng.uniform(0.6, 0.9)) for s in SUBJECTS}
    moved = {s: v + float(rng.normal(0.02, 0.05)) for s, v in base.items()}
    left, right = evaluation("l", base), evaluation("r", moved)
    assert (
        paired_difference(left, right, seed=7).interval
        == paired_difference(left, right, seed=7).interval
    )


def test_the_verdict_refuses_to_call_an_interval_crossing_zero_a_difference():
    rng = np.random.default_rng(4)
    base = {s: float(rng.uniform(0.6, 0.9)) for s in SUBJECTS}
    moved = dict(base)
    moved[SUBJECTS[0]] += 0.6
    found = paired_difference(evaluation("l", base), evaluation("r", moved))
    assert "not evidence of a difference" in found.verdict()


def test_the_table_marks_which_intervals_include_zero():
    left = evaluation("left", dict.fromkeys(SUBJECTS, 0.80))
    real = evaluation("real", dict.fromkeys(SUBJECTS, 0.85))
    rng = np.random.default_rng(5)
    noise = evaluation("noise", {s: 0.80 + float(rng.normal(0, 0.05)) for s in SUBJECTS})
    text = table([paired_difference(left, real), paired_difference(left, noise)], "left")
    lines = [
        line for line in text.splitlines() if "better" in line and "improved" not in line
    ]
    assert lines[0].rstrip().endswith("better"), "a real difference is unmarked"
    assert lines[1].rstrip().endswith("~"), "and a noisy one is marked"


def test_an_unknown_metric_is_refused():
    left = evaluation("left", {"S2": 0.6})
    with pytest.raises(ValueError, match="unknown per-subject metric"):
        paired_difference(left, left, metric="vibes")
