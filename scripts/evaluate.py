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
from physioml.models.classical import MODELS, calibrated_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", help="path to the .npz written by build_features.py")
    parser.add_argument("--positive", default="stress", help="positive class")
    parser.add_argument("--folds", type=int, help="use k-fold instead of leave-one-out")
    parser.add_argument("--models", nargs="*", help="limit to these model names")
    parser.add_argument(
        "--calibrated",
        action="store_true",
        help="also offer each model with probabilities calibrated on held-out subjects",
    )
    args = parser.parse_args()

    table = FeatureTable.load(args.table)
    print(table.summary())
    multiclass = args.positive == "none"
    if multiclass:
        print(f"{len(table.counts())}-class task: {table.counts()}")
    else:
        positive = table.binary(args.positive)
        share = positive.mean() * 100
        print(
            f"binary {args.positive} task: {positive.sum()} positive / "
            f"{len(positive)} ({share:.1f}%)"
        )
    print(f"feature set {table.feature_set_version}, qc policy {table.qc_policy_version}\n")

    def splits():
        if args.folds:
            return group_k_fold(table.subject_ids, folds=args.folds)
        return leave_one_subject_out(table.subject_ids)

    # Area under the curve and the calibration measures are two-class
    # quantities. A five-stage problem is reported the way its field reports
    # it: agreement corrected for chance, and per-stage recall underneath.
    header = (
        f"{'model':28} {'bal.acc':>13}   {'kappa':>13}   {'acc':>5}   {'F1':>5}   "
        f"{'worst':>5}"
        if multiclass
        else f"{'model':28} {'bal.acc':>13}   {'F1':>5}   {'AUC':>5}   "
        f"{'PR-AUC':>5}   {'Brier':>5}   {'ECE':>5}   {'worst':>5}"
    )
    print(header)
    print("-" * len(header))

    available = dict(MODELS)
    if args.calibrated:
        available |= calibrated_models()
    chosen = args.models or list(available)
    results = {}
    for name in chosen:
        result = evaluate(
            table,
            available[name],
            splits(),
            model_name=name,
            positive=None if multiclass else args.positive,
        )
        results[name] = result
        if multiclass:
            s = result.summary
            print(
                f"{name:28} {s['balanced_accuracy_mean']:.3f} "
                f"±{s['balanced_accuracy_sd']:.3f}   {s['kappa_mean']:.3f} "
                f"±{s['kappa_sd']:.3f}   {s['accuracy_mean']:.3f}   "
                f"{s['f1_macro_mean']:.3f}   "
                f"{s.get('worst_subject_balanced_accuracy', float('nan')):.3f}",
                flush=True,
            )
        else:
            print(result.line(), flush=True)

    unscored = sorted({s for r in results.values() for s in r.skipped})
    if unscored:
        print(
            f"\nnot scored: {', '.join(unscored)} -- one side of the fold held a "
            "single class, so balanced accuracy is undefined there"
        )

    if multiclass:
        print("\nper-stage recall")
        for name, result in results.items():
            found = {
                k[len("recall_") :]: v
                for k, v in sorted(result.summary.items())
                if k.startswith("recall_")
            }
            shown = "  ".join(f"{k} {v:.3f}" for k, v in found.items())
            print(f"{name:28} {shown}")

    print("\nper-subject balanced accuracy, and the ranking underneath it")
    print(
        "  bal.acc is the label at the operating point; AUC is whether the windows\n"
        "  were ordered correctly at all. 0.500 beside a high AUC is a threshold\n"
        "  that sits off the end of that person's probabilities, not a model that\n"
        "  learned nothing about them."
    )
    per_model = {}
    for name, result in results.items():
        scores: dict[str, float] = {}
        for fold in result.folds:
            scores.update(fold.per_subject)
        per_model[name] = scores
    per_auc = {
        name: {s: v for fold in result.folds for s, v in fold.per_subject_auc.items()}
        for name, result in results.items()
    }
    print("subj  " + "  ".join(f"{n[:14]:>15}" for n in per_model))
    print("      " + "  ".join(f"{'bal.acc  AUC':>15}" for _ in per_model))
    for subject in table.subject_ids:
        if any(subject not in scores for scores in per_model.values()):
            continue  # not scored for at least one model; see "not scored" above
        cells = []
        for name in per_model:
            auc = per_auc[name].get(subject)
            shown = f"{auc:.3f}" if auc is not None else "  -  "
            cells.append(f"{per_model[name][subject]:7.3f}{shown:>8}")
        print(f"{subject:5} " + "  ".join(cells))


if __name__ == "__main__":
    main()
