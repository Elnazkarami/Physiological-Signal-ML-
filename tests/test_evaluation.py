"""Splits, metrics, and the guarantee that holds them together.

The thing being defended is that no participant appears on both sides of a
split. It is asserted for every strategy, and again at the end of a real
evaluation, because a leak introduced later would not otherwise show up as a
failure — it would show up as a better score.
"""

from __future__ import annotations

import numpy as np
import pytest

from physioml.dataset import FeatureTable
from physioml.evaluation.metrics import aggregate, expected_calibration_error, score
from physioml.evaluation.run import evaluate
from physioml.evaluation.splits import (
    group_k_fold,
    held_out_cohort,
    leave_one_subject_out,
)
from physioml.models.classical import MODELS

SUBJECTS = [f"S{n}" for n in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17)]


def table(rows_per_subject: int = 40, seed: int = 0) -> FeatureTable:
    """A synthetic cohort where the label is genuinely learnable."""
    rng = np.random.default_rng(seed)
    values, subjects, labels = [], [], []
    for subject in SUBJECTS:
        offset = rng.normal(0, 3)  # this participant's own baseline
        for i in range(rows_per_subject):
            stressed = i % 4 == 0
            values.append(
                [offset + rng.normal(2.0 if stressed else 0.0, 0.5), rng.normal(offset, 1)]
            )
            subjects.append(subject)
            labels.append("stress" if stressed else "baseline")
    return FeatureTable(
        feature_names=("a", "b"),
        values=np.array(values),
        subjects=np.array(subjects),
        labels=np.array(labels),
        window_ids=tuple(f"win-{i}" for i in range(len(values))),
        feature_set_version="test-1",
        qc_policy_version="test-1",
    )


# ── the guarantee ───────────────────────────────────────────────────────────


def test_leave_one_subject_out_holds_each_participant_out_once():
    folds = list(leave_one_subject_out(SUBJECTS))
    assert len(folds) == len(SUBJECTS)
    assert sorted(f.test_subjects[0] for f in folds) == sorted(SUBJECTS)


@pytest.mark.parametrize("folds", [2, 3, 5])
def test_group_k_fold_covers_everyone_exactly_once(folds):
    made = list(group_k_fold(SUBJECTS, folds=folds))
    assert len(made) == folds
    held = [s for f in made for s in f.test_subjects]
    assert sorted(held) == sorted(SUBJECTS), "each subject tested exactly once"


def test_no_strategy_puts_a_subject_on_both_sides():
    """The one error that inflates a score while looking like success."""
    strategies = [
        *leave_one_subject_out(SUBJECTS),
        *group_k_fold(SUBJECTS, folds=5),
        held_out_cohort(SUBJECTS),
    ]
    for split in strategies:
        shared = set(split.train_subjects) & set(split.test_subjects)
        assert not shared, f"{split.strategy} fold {split.fold} shares {shared}"


def test_the_row_masks_respect_the_subject_grouping():
    made = table()
    for split in leave_one_subject_out(made.subject_ids):
        train, test = split.mask(made.subjects)
        assert not set(made.subjects[train]) & set(made.subjects[test])
        assert train.size + test.size == len(made)


def test_a_split_that_leaves_nobody_to_train_on_is_refused():
    with pytest.raises(ValueError, match="nobody to train on"):
        held_out_cohort(SUBJECTS, test_fraction=1.0)


@pytest.mark.parametrize("folds", [1, 99])
def test_an_impossible_number_of_folds_is_refused(folds):
    with pytest.raises(ValueError, match="not usable"):
        list(group_k_fold(SUBJECTS, folds=folds))


def test_group_k_fold_is_reproducible_from_its_seed():
    assert [f.test_subjects for f in group_k_fold(SUBJECTS, seed=7)] == [
        f.test_subjects for f in group_k_fold(SUBJECTS, seed=7)
    ]


# ── metrics ─────────────────────────────────────────────────────────────────


def test_always_answering_the_common_class_scores_half():
    """The number every result has to beat, on a task that is 22% positive."""
    truth = np.array([0] * 78 + [1] * 22)
    always_negative = np.zeros(100, dtype=int)
    got = score(truth, always_negative)
    assert got.accuracy == pytest.approx(0.78)
    assert got.balanced_accuracy == pytest.approx(0.5)
    assert got.positive_rate == pytest.approx(0.22)


def test_a_perfect_classifier_scores_one():
    truth = np.array([0, 0, 1, 1])
    got = score(truth, truth, np.array([0.01, 0.02, 0.98, 0.99]))
    assert got.balanced_accuracy == pytest.approx(1.0)
    assert got.roc_auc == pytest.approx(1.0)
    assert got.brier < 0.01


def test_calibration_error_is_zero_when_confidence_matches_frequency():
    """Seventy per cent confident, right seventy per cent of the time."""
    truth = np.array([1] * 70 + [0] * 30)
    stated = np.full(100, 0.7)
    assert expected_calibration_error(truth, stated) == pytest.approx(0.0, abs=0.01)


def test_calibration_error_grows_with_overconfidence():
    truth = np.array([1] * 70 + [0] * 30)
    assert expected_calibration_error(truth, np.full(100, 0.99)) > 0.25


def test_the_worst_subject_is_reported_not_only_the_mean():
    truth = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    predicted = np.array([0, 1, 0, 1, 1, 0, 1, 0])  # perfect for A, wrong for B
    subjects = np.array(["A"] * 4 + ["B"] * 4)
    got = score(truth, predicted, subjects=subjects)
    assert got.per_subject["A"] == pytest.approx(1.0)
    assert got.per_subject["B"] == pytest.approx(0.0)
    assert got.worst_subject == ("B", 0.0)


def test_aggregate_reports_spread_as_well_as_centre():
    folds = [
        score(np.array([0, 1]), np.array([0, 1])),
        score(np.array([0, 1]), np.array([1, 0])),
    ]
    summary = aggregate(folds)
    assert summary["balanced_accuracy_mean"] == pytest.approx(0.5)
    assert summary["balanced_accuracy_min"] == pytest.approx(0.0)
    assert summary["balanced_accuracy_sd"] > 0


# ── end to end ──────────────────────────────────────────────────────────────


def test_an_evaluation_records_a_disjoint_run_for_every_fold():
    made = table()
    result = evaluate(
        made,
        MODELS["logistic"],
        leave_one_subject_out(made.subject_ids),
        model_name="logistic",
    )
    assert len(result.folds) == len(SUBJECTS)
    assert len(result.runs) == len(SUBJECTS)
    for run in result.runs:
        # TrainingRun refuses overlap at construction; this asserts the
        # evaluation actually produces runs rather than skipping them.
        assert not set(run.train_subjects) & set(run.test_subjects)
        assert run.feature_schema_version == "test-1"


def test_a_learnable_signal_is_learned_and_a_constant_model_is_not():
    made = table()
    folds = list(leave_one_subject_out(made.subject_ids))
    real = evaluate(made, MODELS["logistic"], folds, model_name="logistic")
    naive = evaluate(made, MODELS["majority"], folds, model_name="majority")

    assert naive.summary["balanced_accuracy_mean"] == pytest.approx(0.5, abs=0.01)
    assert real.summary["balanced_accuracy_mean"] > 0.8


def test_every_model_in_the_registry_runs():
    made = table(rows_per_subject=24)
    folds = list(group_k_fold(made.subject_ids, folds=3))
    for name, factory in MODELS.items():
        result = evaluate(made, factory, folds, model_name=name)
        assert len(result.folds) == 3, name
        assert 0.0 <= result.summary["balanced_accuracy_mean"] <= 1.0, name
