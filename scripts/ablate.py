"""Ask which sensor a result is actually coming from.

    python scripts/ablate.py wesad_features.npz --model logistic

Scores the model on the full feature set, on each signal alone, and with each
signal removed. The two are different questions: a signal can score well alone
and cost nothing when dropped, because another one carries the same
information.

Requires the ml extra: ``pip install -e ".[ml]"``.
"""

from __future__ import annotations

import argparse

from physioml.dataset import FeatureTable
from physioml.evaluation.ablation import ablate
from physioml.evaluation.splits import group_k_fold, leave_one_subject_out
from physioml.models.classical import MODELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", help="path to the .npz written by build_features.py")
    parser.add_argument("--model", default="logistic", choices=sorted(MODELS))
    parser.add_argument("--positive", default="stress", help="positive class")
    parser.add_argument("--folds", type=int, help="use k-fold instead of leave-one-out")
    args = parser.parse_args()

    table = FeatureTable.load(args.table)
    print(table.summary())
    print(f"feature set {table.feature_set_version}, qc policy {table.qc_policy_version}\n")

    def splits():
        if args.folds:
            return group_k_fold(table.subject_ids, folds=args.folds)
        return leave_one_subject_out(table.subject_ids)

    result = ablate(
        table, MODELS[args.model], splits, model_name=args.model, positive=args.positive
    )
    print(result.table())
    print("\nbalanced accuracy lost when each signal is removed:")
    for signal, contribution in result.ranked():
        print(f"   {signal:5} {contribution:+.3f}")


if __name__ == "__main__":
    main()
