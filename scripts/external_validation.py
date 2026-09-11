"""Fit one model, score it in-domain and out-of-domain.

    python scripts/external_validation.py sleep_both.npz --train SC --test ST

Every other evaluation here holds participants out of one dataset, which asks
whether a model transfers to a new person. This asks whether it transfers to a
new protocol, which is the harder question and the one a deployment faces.

**One fitted model, two test sets.** Part of the fitted cohort is reserved as an
in-domain holdout and never trained on, so the in-domain and cross-cohort scores
come from the same model, the same training set and the same pooling. Comparing
a leave-one-subject-out mean against a single cross-cohort score would confound
the change of protocol with the training size, the aggregation and the stage
prevalences all at once.

Requires the ml extra: ``pip install -e ".[ml]"``.
"""

from __future__ import annotations

import argparse

import numpy as np

from physioml.dataset import FeatureTable
from physioml.evaluation.run import evaluate
from physioml.evaluation.splits import Split
from physioml.models.classical import MODELS
from physioml.neural.context import with_context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", help="a table holding both cohorts")
    parser.add_argument("--train", default="SC", help="identifier prefix to fit on")
    parser.add_argument("--test", default="ST", help="identifier prefix to score")
    parser.add_argument("--model", default="random_forest", choices=sorted(MODELS))
    parser.add_argument("--positive", default="none")
    parser.add_argument(
        "--holdout",
        type=float,
        default=0.25,
        help="share of the fitted cohort reserved as an in-domain test set",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context", action="store_true", help="add neighbouring epochs")
    args = parser.parse_args()

    table = FeatureTable.load(args.table)
    if args.context:
        table = with_context(table)
    print(table.summary())
    positive = None if args.positive == "none" else args.positive

    # One model, two test sets. The comparison only isolates the change of
    # protocol if the thing being compared is the same fitted model scored two
    # ways -- same training set, same estimator, same pooling. Comparing a
    # leave-one-out mean against a single pooled score confounds the protocol
    # with the training size and with how the metric was aggregated.
    fitted_pool = [s for s in table.subject_ids if s.startswith(args.train)]
    rng = np.random.default_rng(args.seed)
    shuffled = list(fitted_pool)
    rng.shuffle(shuffled)
    held_in = sorted(shuffled[: max(1, round(len(shuffled) * args.holdout))])
    train_on = sorted(set(fitted_pool) - set(held_in))
    held_out = [s for s in table.subject_ids if s.startswith(args.test)]

    print(f"fitting on {len(train_on)} {args.train} participants")
    print(f"scoring {len(held_in)} held-in {args.train} and {len(held_out)} {args.test}\n")

    in_domain = Split(
        train_subjects=tuple(train_on),
        test_subjects=tuple(held_in),
        strategy=f"in_domain_holdout:{args.train}",
        fold=0,
        seed=args.seed,
    )
    across = Split(
        train_subjects=tuple(train_on),
        test_subjects=tuple(held_out),
        strategy=f"cross_cohort:{args.train}->{args.test}",
        fold=0,
        seed=args.seed,
    )

    header = f"{'scored on':34} {'kappa':>7} {'accuracy':>9} {'bal.acc':>8}  participants"
    print(header)
    print("-" * len(header))
    results = {}
    for label, split in ((f"held-in {args.train}", in_domain), (f"{args.test}", across)):
        result = evaluate(
            table,
            MODELS[args.model],
            [split],
            model_name=label,
            positive=positive,
            task="sleep_stage",
            dataset_version="sleep-edf",
        )
        results[label] = result
        s = result.summary
        print(
            f"{label:34} {s.get('kappa_mean', float('nan')):7.3f} "
            f"{s['accuracy_mean']:9.3f} {s['balanced_accuracy_mean']:8.3f}"
            f"  {len(split.test_subjects):>12}"
        )

    print("\nthe same figures averaged over participants rather than pooled:")
    for label, result in results.items():
        per = {s: v for f in result.folds for s, v in f.per_subject_kappa.items()}
        if per:
            values = np.array(list(per.values()))
            worst = min(per, key=lambda s: per[s])
            print(
                f"  {label:32} kappa {values.mean():.3f} ±{values.std():.3f}"
                f"  worst {values.min():.3f} ({worst})"
            )
