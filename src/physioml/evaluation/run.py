"""Running a model across subject-wise folds and reporting what happened."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from physioml.core.registry import TrainingRun
from physioml.dataset import FeatureTable
from physioml.evaluation.metrics import Scores, aggregate, score
from physioml.evaluation.splits import Split


@dataclass(frozen=True, slots=True)
class Evaluation:
    """One model, over one set of folds."""

    model_name: str
    task: str
    feature_names: tuple[str, ...]
    """The columns this run was fitted on -- the subject of an ablation."""

    folds: tuple[Scores, ...]
    runs: tuple[TrainingRun, ...]
    """A training run per fold, carrying the split it was fitted under."""

    @property
    def summary(self) -> dict[str, float]:
        return aggregate(list(self.folds))

    def line(self) -> str:
        """One row of a comparison table."""
        s = self.summary

        def show(key: str) -> str:
            value = s.get(key)
            return f"{value:.3f}" if value is not None else "  -  "

        return (
            f"{self.model_name:18} "
            f"{s['balanced_accuracy_mean']:.3f} ±{s['balanced_accuracy_sd']:.3f}"
            f"   {show('f1_macro_mean')}"
            f"   {show('roc_auc_mean')}"
            f"   {show('pr_auc_mean')}"
            f"   {show('brier_mean')}"
            f"   {show('ece_mean')}"
            f"   {show('worst_subject_balanced_accuracy')}"
        )


def evaluate(
    table: FeatureTable,
    model_factory,
    splits: Iterable[Split],
    *,
    model_name: str,
    task: str = "stress",
    positive: str = "stress",
    dataset_version: str = "wesad-1",
) -> Evaluation:
    """Fit and score one model on every fold, recording the run behind each.

    A fresh model is built per fold. Reusing one would carry the previous fold's
    fitted state into the next, which is the same leak as sharing subjects,
    arrived at by a different route.
    """
    labels = table.binary(positive)
    folds: list[Scores] = []
    runs: list[TrainingRun] = []

    for split in splits:
        train_rows, test_rows = split.mask(table.subjects)
        if train_rows.size == 0 or test_rows.size == 0:
            continue
        y_train, y_test = labels[train_rows], labels[test_rows]
        if len(np.unique(y_train)) < 2:
            continue

        model = model_factory()
        model.fit(table.values[train_rows], y_train)
        predicted = model.predict(table.values[test_rows])
        probability = (
            model.predict_proba(table.values[test_rows])[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        folds.append(score(y_test, predicted, probability, table.subjects[test_rows]))
        runs.append(
            TrainingRun.create(
                task=task,
                dataset_version=dataset_version,
                split_strategy=split.strategy,
                train_subjects=split.train_subjects,
                test_subjects=split.test_subjects,
                feature_schema_version=table.feature_set_version,
                preprocessing_version=table.qc_policy_version,
                random_seed=split.seed,
                metrics={"balanced_accuracy": folds[-1].balanced_accuracy},
            )
        )

    return Evaluation(model_name, task, table.feature_names, tuple(folds), tuple(runs))
