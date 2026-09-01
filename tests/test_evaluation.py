"""Splits, metrics, and the guarantee that holds them together.

The thing being defended is that no participant appears on both sides of a
split. It is asserted for every strategy, and again at the end of a real
evaluation, because a leak introduced later would not otherwise show up as a
failure — it would show up as a better score.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from physioml.dataset import FeatureTable
from physioml.evaluation.ablation import ablate
from physioml.evaluation.metrics import aggregate, expected_calibration_error, score
from physioml.evaluation.run import evaluate
from physioml.evaluation.splits import (
    group_k_fold,
    held_out_cohort,
    leave_one_subject_out,
)
from physioml.models.calibration import SubjectCalibrated, calibrated
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
        window_starts=np.tile(
            np.arange(rows_per_subject, dtype=float) * 30.0, len(SUBJECTS)
        ),
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


# ── ablation ────────────────────────────────────────────────────────────────


def informative_table(seed: int = 0) -> FeatureTable:
    """A cohort where the accelerometer knows the label and the skin does not."""
    rng = np.random.default_rng(seed)
    signal_names = ["acc_magnitude_mean", "acc_jerk_mean", "acc_x_sd"]
    noise_names = ["eda_mean", "eda_sd", "scl_mean"]
    values, subjects, labels = [], [], []
    for subject in SUBJECTS:
        for i in range(40):
            stressed = i % 4 == 0
            informative = rng.normal(2.0 if stressed else 0.0, 0.5, len(signal_names))
            values.append([*informative, *rng.normal(0, 1, len(noise_names))])
            subjects.append(subject)
            labels.append("stress" if stressed else "baseline")
    return FeatureTable(
        feature_names=(*signal_names, *noise_names),
        values=np.array(values),
        subjects=np.array(subjects),
        labels=np.array(labels),
        window_ids=tuple(f"win-{i}" for i in range(len(values))),
        window_starts=np.tile(np.arange(40, dtype=float) * 30.0, len(SUBJECTS)),
        feature_set_version="test-1",
        qc_policy_version="test-1",
    )


def test_selecting_columns_keeps_every_row_and_its_subject():
    made = table()
    picked = made.select(["b"])
    assert picked.feature_names == ("b",)
    assert picked.values.shape == (len(made), 1)
    assert np.array_equal(picked.subjects, made.subjects)
    assert np.array_equal(picked.values[:, 0], made.values[:, 1])


def test_selecting_a_feature_that_is_not_there_is_refused():
    with pytest.raises(KeyError, match="not in this table"):
        table().select(["a", "nonexistent"])


def test_selecting_nothing_is_refused():
    with pytest.raises(ValueError, match="no features"):
        table().select([])


def test_ablation_finds_the_signal_that_carries_the_label():
    made = informative_table()
    result = ablate(
        made,
        MODELS["logistic"],
        lambda: leave_one_subject_out(made.subject_ids),
        model_name="logistic",
    )
    assert result.signals == ("wrist EDA", "wrist ACC")

    alone = {s: e.summary["balanced_accuracy_mean"] for s, e in result.alone.items()}
    assert alone["wrist ACC"] > 0.8, "the informative signal should stand on its own"
    assert alone["wrist EDA"] == pytest.approx(0.5, abs=0.08), "noise should be near chance"

    assert result.ranked()[0][0] == "wrist ACC"
    assert result.contribution("wrist ACC") > 0.2
    assert result.contribution("wrist EDA") < 0.05


def test_each_ablation_gets_its_own_folds():
    """A generator of splits would be exhausted after the first evaluation."""
    made = informative_table()
    calls = 0

    def splits():
        nonlocal calls
        calls += 1
        return leave_one_subject_out(made.subject_ids)

    result = ablate(made, MODELS["logistic"], splits, model_name="logistic")
    assert calls == 1 + 2 * len(result.signals)
    for evaluation in (result.full, *result.alone.values(), *result.without.values()):
        assert len(evaluation.folds) == len(SUBJECTS), "a fold set was consumed twice"


def test_an_evaluation_records_the_features_it_was_fitted_on():
    made = informative_table()
    result = ablate(
        made,
        MODELS["logistic"],
        lambda: leave_one_subject_out(made.subject_ids),
        model_name="logistic",
    )
    assert result.alone["wrist ACC"].feature_names == (
        "acc_magnitude_mean",
        "acc_jerk_mean",
        "acc_x_sd",
    )
    assert "eda_mean" not in result.without["wrist EDA"].feature_names
    assert len(result.full.feature_names) == 6


def test_a_signal_absent_from_the_table_is_refused():
    made = informative_table()
    with pytest.raises(KeyError, match="no wrist TEMP features"):
        ablate(
            made,
            MODELS["logistic"],
            lambda: leave_one_subject_out(made.subject_ids),
            model_name="logistic",
            signals=["wrist TEMP"],
        )


# ── calibration ─────────────────────────────────────────────────────────────


def overconfident() -> object:
    """A model whose probabilities are pushed toward 0 and 1.

    Well separated and badly calibrated, which is the combination calibration
    exists for and the one a score alone will not show.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    class Sharpened(LogisticRegression):
        def predict_proba(self, X):
            p = super().predict_proba(X)[:, 1]
            sharp = np.clip(p * 3.0 - 1.0, 0.001, 0.999)
            return np.column_stack([1.0 - sharp, sharp])

    return Pipeline([("scale", StandardScaler()), ("model", Sharpened(max_iter=2000))])


def test_calibration_without_the_subject_of_each_row_is_refused():
    """The shortcut that makes a calibrator look better than it is."""
    made = table()
    model = SubjectCalibrated(MODELS["logistic"])
    with pytest.raises(ValueError, match="shares participants"):
        model.fit(made.values, made.binary("stress"))


def test_calibration_needs_more_than_one_training_subject():
    made = table()
    model = SubjectCalibrated(MODELS["logistic"])
    one = made.subjects == made.subject_ids[0]
    with pytest.raises(ValueError, match="at least two training subjects"):
        model.fit(made.values[one], made.binary("stress")[one], groups=made.subjects[one])


def test_an_unknown_calibration_method_is_refused():
    with pytest.raises(ValueError, match="unknown calibration method"):
        SubjectCalibrated(MODELS["logistic"], method="wishful")


@pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
def test_calibration_improves_a_confidently_wrong_model(method):
    made = table()
    folds = list(leave_one_subject_out(made.subject_ids))
    before = evaluate(made, overconfident, folds, model_name="raw")
    after = evaluate(
        made, calibrated(overconfident, method=method), folds, model_name=method
    )
    assert after.summary["ece_mean"] < before.summary["ece_mean"]
    assert after.summary["brier_mean"] < before.summary["brier_mean"]


def test_calibration_does_not_move_the_decision():
    """Probabilities are restated; the operating point is left alone."""
    made = table()
    folds = list(leave_one_subject_out(made.subject_ids))
    before = evaluate(made, overconfident, folds, model_name="raw")
    after = evaluate(made, calibrated(overconfident), folds, model_name="iso")
    assert after.summary["balanced_accuracy_mean"] == pytest.approx(
        before.summary["balanced_accuracy_mean"]
    )
    assert after.summary["f1_macro_mean"] == pytest.approx(before.summary["f1_macro_mean"])


def test_isotonic_calibration_barely_moves_the_ranking():
    """Nearly, but not exactly, preserved -- and the difference is real.

    Isotonic regression is monotone *non-decreasing*, which is not the same as
    strictly increasing: its flat regions map distinct scores onto one value
    and create ties, and ties change the area under the curve. Asserting
    equality here would be asserting something false, and the measured tables
    show AUC moving -- 0.954 to 0.951 for logistic regression on WESAD.
    """
    made = table()
    folds = list(leave_one_subject_out(made.subject_ids))
    before = evaluate(made, overconfident, folds, model_name="raw")
    after = evaluate(made, calibrated(overconfident), folds, model_name="iso")
    moved = abs(after.summary["roc_auc_mean"] - before.summary["roc_auc_mean"])
    assert moved < 0.05, "the ranking should be nearly, not exactly, preserved"


def test_isotonic_calibration_creates_ties_that_the_raw_scores_do_not_have():
    """The mechanism behind the previous test, asserted directly."""
    from physioml.models.calibration import SubjectCalibrated

    made = table()
    y = made.binary("stress")
    raw = overconfident()
    raw.fit(made.values, y)
    calibrator = SubjectCalibrated(overconfident)
    calibrator.fit(made.values, y, groups=made.subjects)

    before = raw.predict_proba(made.values)[:, 1]
    after = calibrator.predict_proba(made.values)[:, 1]
    assert len(np.unique(after)) < len(np.unique(before))


def test_a_calibrated_model_still_reports_two_columns_that_sum_to_one():
    made = table()
    model = SubjectCalibrated(MODELS["logistic"]).fit(
        made.values, made.binary("stress"), groups=made.subjects
    )
    probability = model.predict_proba(made.values)
    assert probability.shape == (len(made), 2)
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert ((probability >= 0.0) & (probability <= 1.0)).all()


def single_class_subject(table_in: FeatureTable, subject: str) -> FeatureTable:
    """The same table with one participant's positives relabelled away."""
    labels = table_in.labels.copy()
    labels[table_in.subjects == subject] = "baseline"
    return replace(table_in, labels=labels)


def test_a_fold_with_one_class_on_test_is_not_scored():
    """It was, and it moved the majority baseline off 0.500.

    Balanced accuracy is the mean of per-class recall, and an absent class has
    none. Scored anyway the fold returns 1.0 or 0.0 for whichever class is
    present, and the cohort mean moves for a reason unrelated to the model.
    """
    made = single_class_subject(table(), "S4")
    folds = list(leave_one_subject_out(made.subject_ids))
    result = evaluate(made, MODELS["majority"], folds, model_name="majority")

    assert "S4" in result.skipped
    assert len(result.folds) == len(SUBJECTS) - 1
    assert result.summary["balanced_accuracy_mean"] == pytest.approx(0.5)


def test_the_subjects_that_could_not_be_scored_are_reported():
    """A mean over fourteen subjects presented as fifteen is a quiet lie."""
    made = single_class_subject(table(), "S7")
    result = evaluate(
        made,
        MODELS["logistic"],
        leave_one_subject_out(made.subject_ids),
        model_name="logistic",
    )
    assert result.skipped == ("S7",)
    assert "S7" not in {s for fold in result.folds for s in fold.per_subject}


def test_an_evaluation_with_nothing_scorable_raises():
    made = table()
    labels = np.full(len(made), "baseline")
    with pytest.raises(ValueError, match="no fold could be scored"):
        evaluate(
            replace(made, labels=labels),
            MODELS["logistic"],
            leave_one_subject_out(made.subject_ids),
            model_name="logistic",
        )


# ── more than two classes ───────────────────────────────────────────────────


def staged(rows_per_subject: int = 200, seed: int = 0) -> FeatureTable:
    """A five-class cohort shaped like a night: one stage dominates."""
    rng = np.random.default_rng(seed)
    names = ["N1", "N2", "N3", "REM", "W"]
    shares = [0.06, 0.45, 0.15, 0.18, 0.16]
    values, subjects, labels, starts = [], [], [], []
    for subject in SUBJECTS:
        drawn = rng.choice(len(names), size=rows_per_subject, p=shares)
        for i, which in enumerate(drawn):
            values.append(rng.normal(float(which) * 1.5, 0.7, 2))
            subjects.append(subject)
            labels.append(names[which])
            starts.append(i * 30.0)
    return FeatureTable(
        feature_names=("a", "b"),
        values=np.array(values),
        subjects=np.array(subjects),
        labels=np.array(labels),
        window_ids=tuple(f"e{i}" for i in range(len(values))),
        window_starts=np.array(starts, dtype=float),
        feature_set_version="test-1",
        qc_policy_version="test-1",
    )


def test_agreement_by_chance_scores_zero_however_uneven_the_classes():
    """Kappa is why sleep staging is not reported in accuracy: a constant
    answer on a night that is 45% one stage is 45% accurate."""
    truth = np.array(["N2"] * 45 + ["W"] * 20 + ["REM"] * 18 + ["N3"] * 12 + ["N1"] * 5)
    constant = np.full(truth.size, "N2")
    got = score(truth, constant)
    assert got.accuracy == pytest.approx(0.45)
    assert got.kappa == pytest.approx(0.0, abs=1e-9)
    assert got.balanced_accuracy == pytest.approx(0.2)


def test_perfect_agreement_scores_one():
    truth = np.array(["N1", "N2", "N3", "REM", "W"] * 10)
    assert score(truth, truth).kappa == pytest.approx(1.0)


def test_the_share_of_the_commonest_class_is_what_a_constant_answer_scores():
    """Two classes named W and N2 are binary, and `truth == 1` is false for
    every row of them -- which reported a constant-answer rate of zero."""
    named = np.array(["N2"] * 45 + ["W"] * 55)
    assert score(named, named).positive_rate == pytest.approx(0.55)

    five = np.array(["N2"] * 45 + ["W"] * 20 + ["REM"] * 18 + ["N3"] * 12 + ["N1"] * 5)
    assert score(five, five).positive_rate == pytest.approx(0.45)

    # A 0/1 problem still reports the positive share, not the commonest.
    binary = np.array([0] * 78 + [1] * 22)
    assert score(binary, binary).positive_rate == pytest.approx(0.22)


def test_recall_is_reported_for_every_stage_not_only_the_average():
    """A macro average hides which stage a model cannot see, and it is
    nearly always N1."""
    truth = np.array(["N1"] * 10 + ["N2"] * 10)
    predicted = np.array(["N2"] * 10 + ["N2"] * 10)  # never says N1
    got = score(truth, predicted)
    assert got.per_class["N1"] == pytest.approx(0.0)
    assert got.per_class["N2"] == pytest.approx(1.0)
    assert got.labels == ("N1", "N2")


def test_the_confusion_matrix_can_be_read_because_its_order_is_recorded():
    truth = np.array(["W", "W", "N2", "N2"])
    predicted = np.array(["W", "N2", "N2", "N2"])
    got = score(truth, predicted)
    order = {name: i for i, name in enumerate(got.labels)}
    assert got.confusion[order["W"]][order["N2"]] == 1, "one wake epoch called N2"
    assert got.confusion[order["N2"]][order["N2"]] == 2


def test_two_class_metrics_are_absent_rather_than_wrong_on_five_classes():
    truth = np.array(["N1", "N2", "N3", "REM", "W"] * 4)
    got = score(truth, truth, np.linspace(0, 1, truth.size))
    assert got.roc_auc is None
    assert got.brier is None
    assert got.ece is None


def test_an_evaluation_can_score_the_labels_as_they_are():
    """positive=None: five stages, not one-versus-rest."""
    made = staged()
    folds = list(leave_one_subject_out(made.subject_ids))
    result = evaluate(made, MODELS["logistic"], folds, model_name="logistic", positive=None)
    assert result.summary["kappa_mean"] > 0.5
    assert set(result.folds[0].labels) == {"N1", "N2", "N3", "REM", "W"}
    assert "recall_N1" in result.summary


def test_a_constant_model_on_five_classes_scores_a_fifth_and_no_agreement():
    made = staged()
    folds = list(leave_one_subject_out(made.subject_ids))
    result = evaluate(made, MODELS["majority"], folds, model_name="majority", positive=None)
    assert result.summary["balanced_accuracy_mean"] == pytest.approx(0.2, abs=0.01)
    assert result.summary["kappa_mean"] == pytest.approx(0.0, abs=1e-6)


def test_an_ablation_reports_agreement_when_there_is_no_area_under_a_curve():
    made = staged(rows_per_subject=120)
    result = ablate(
        made,
        MODELS["logistic"],
        lambda: leave_one_subject_out(made.subject_ids),
        model_name="logistic",
        groups={"first": ("a",), "second": ("b",)},
        positive=None,
    )
    assert "kappa" in result.table()
    assert "AUC" not in result.table()
