"""Fit on one cohort, score another.

    python scripts/external_validation.py sleep_all.npz --train SC --test ST

Every other evaluation here holds participants out of one dataset, which asks
whether a model transfers to a new person. This asks whether it transfers to a
new protocol, which is the harder question and the one a deployment faces.

Requires the ml extra: ``pip install -e ".[ml]"``.
"""

from __future__ import annotations

import argparse

from physioml.dataset import FeatureTable
from physioml.evaluation.run import evaluate
from physioml.evaluation.splits import held_out_cohort_split, leave_one_subject_out
from physioml.models.classical import MODELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", help="a table holding both cohorts")
    parser.add_argument("--train", default="SC", help="identifier prefix to fit on")
    parser.add_argument("--test", default="ST", help="identifier prefix to score")
    parser.add_argument("--model", default="random_forest", choices=sorted(MODELS))
    parser.add_argument("--positive", default="none")
    args = parser.parse_args()

    table = FeatureTable.load(args.table)
    print(table.summary())
    positive = None if args.positive == "none" else args.positive

    fitted = [s for s in table.subject_ids if s.startswith(args.train)]
    held = [s for s in table.subject_ids if s.startswith(args.test)]
    print(
        f"fitting on {len(fitted)} {args.train} participants, "
        f"scoring {len(held)} {args.test} participants\n"
    )

    split = held_out_cohort_split(
        table.subject_ids, train_prefix=args.train, test_prefix=args.test
    )
    across = evaluate(
        table,
        MODELS[args.model],
        [split],
        model_name=f"{args.train}->{args.test}",
        positive=positive,
        task="sleep_stage",
        dataset_version="sleep-edf",
    )

    # The comparison that gives it meaning: the same model, held out within the
    # cohort it was fitted on. A drop from one to the other is what the change
    # of protocol costs.
    inside_rows = table.select(list(table.feature_names))
    inside = evaluate(
        inside_rows,
        MODELS[args.model],
        leave_one_subject_out(fitted),
        model_name=f"within {args.train}",
        positive=positive,
        task="sleep_stage",
        dataset_version="sleep-edf",
    )

    header = f"{'evaluation':22} {'bal.acc':>9} {'kappa':>9} {'accuracy':>9}"
    print(header)
    print("-" * len(header))
    for name, result in (
        (f"within {args.train}", inside),
        (f"{args.train} → {args.test}", across),
    ):
        s = result.summary
        print(
            f"{name:22} {s['balanced_accuracy_mean']:9.3f} "
            f"{s.get('kappa_mean', float('nan')):9.3f} {s['accuracy_mean']:9.3f}"
        )

    fold = across.folds[0]
    if fold.per_class:
        print("\nper-stage recall on the held-out cohort:")
        for stage, recall in sorted(fold.per_class.items()):
            print(f"   {stage:4} {recall:.3f}")
    if fold.per_subject:
        worst = min(fold.per_subject, key=lambda s: fold.per_subject[s])
        print(f"\nworst held-out participant: {worst} at {fold.per_subject[worst]:.3f}")


if __name__ == "__main__":
    main()
