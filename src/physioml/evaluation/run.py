"""Running a model across subject-wise folds and reporting what happened."""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

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

    skipped: tuple[str, ...] = ()
    """Subjects whose fold could not be scored, because one side of it held a
    single class. Carried rather than dropped: a cohort mean over fourteen
    subjects reported as though it were fifteen is a quiet lie."""

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
            f"{self.model_name:28} "
            f"{s['balanced_accuracy_mean']:.3f} ±{s['balanced_accuracy_sd']:.3f}"
            f"   {show('f1_macro_mean')}"
            f"   {show('roc_auc_mean')}"
            f"   {show('pr_auc_mean')}"
            f"   {show('brier_mean')}"
            f"   {show('ece_mean')}"
            f"   {show('worst_subject_balanced_accuracy')}"
        )


def _fit(model: Any, X: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> None:
    """Fit, telling the model who each row belongs to if it asks.

    A calibrator needs the grouping to hold participants out of its own inner
    split; a plain estimator does not take the argument at all. Asked by
    signature rather than by try/except, so a TypeError raised *inside* a fit
    is not swallowed and reported as "this model does not want groups".
    """
    if "groups" in inspect.signature(model.fit).parameters:
        model.fit(X, y, groups=subjects)
    else:
        model.fit(X, y)


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
    skipped: list[str] = []

    for split in splits:
        train_rows, test_rows = split.mask(table.subjects)
        if train_rows.size == 0 or test_rows.size == 0:
            continue
        y_train, y_test = labels[train_rows], labels[test_rows]
        if len(np.unique(y_train)) < 2:
            skipped.extend(split.test_subjects)
            continue
        if len(np.unique(y_test)) < 2:
            # Balanced accuracy is the mean of per-class recall, and a class
            # that is absent has none. Scored anyway, this fold returns the
            # recall of whichever class is present -- 1.0 or 0.0 -- and the
            # cohort mean moves for a reason that has nothing to do with the
            # model. It showed up as a majority-class baseline scoring 0.533
            # instead of exactly 0.500 on the fused table, where one subject
            # had lost every stress window to quality control.
            skipped.extend(split.test_subjects)
            continue

        model = model_factory()
        _fit(model, table.values[train_rows], y_train, table.subjects[train_rows])
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

    if not folds:
        raise ValueError(
            "no fold could be scored; every split had a single class on one side"
        )
    return Evaluation(
        model_name,
        task,
        table.feature_names,
        tuple(folds),
        tuple(runs),
        tuple(dict.fromkeys(skipped)),
    )
