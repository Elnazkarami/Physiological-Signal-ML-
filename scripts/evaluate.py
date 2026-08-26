"""Score every registered model on a feature table, subject by subject.

    python scripts/evaluate.py wesad_features.npz

Leave-one-subject-out by default. Per-subject balanced accuracy is printed for
every model, because the cohort mean is the number that hides a model which has
learned nothing about one person.

Requires the ml extra: ``pip install -e ".[ml]"``.
"""

from __future__ import annotations

import argparse

from physioml.dataset import FeatureTable
from physioml.evaluation.run import evaluate
from physioml.evaluation.splits import group_k_fold, leave_one_subject_out
from physioml.models.classical import MODELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", help="path to the .npz written by build_features.py")
    parser.add_argument("--positive", default="stress", help="positive class")
    parser.add_argument("--folds", type=int, help="use k-fold instead of leave-one-out")
    parser.add_argument("--models", nargs="*", help="limit to these model names")
    args = parser.parse_args()

    table = FeatureTable.load(args.table)
    print(table.summary())
    positive = table.binary(args.positive)
    share = positive.mean() * 100
    print(
        f"binary {args.positive} task: {positive.sum()} positive / {len(positive)} ({share:.1f}%)"
    )
    print(f"feature set {table.feature_set_version}, qc policy {table.qc_policy_version}\n")

    def splits():
        if args.folds:
            return group_k_fold(table.subject_ids, folds=args.folds)
        return leave_one_subject_out(table.subject_ids)

    header = (
        f"{'model':18} {'bal.acc':>13}   {'F1':>5}   {'AUC':>5}   "
        f"{'PR-AUC':>5}   {'Brier':>5}   {'ECE':>5}   {'worst':>5}"
    )
    print(header)
    print("-" * len(header))

    chosen = args.models or list(MODELS)
    results = {}
    for name in chosen:
        result = evaluate(
            table, MODELS[name], splits(), model_name=name, positive=args.positive
        )
        results[name] = result
        print(result.line(), flush=True)

    print("\nper-subject balanced accuracy")
    per_model = {}
    for name, result in results.items():
        scores: dict[str, float] = {}
        for fold in result.folds:
            scores.update(fold.per_subject)
        per_model[name] = scores
    print("subj  " + "  ".join(f"{n[:9]:>9}" for n in per_model))
    for subject in table.subject_ids:
        print(
            f"{subject:5} " + "  ".join(f"{per_model[n][subject]:9.3f}" for n in per_model)
        )


if __name__ == "__main__":
    main()
