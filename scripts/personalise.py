"""Measure what a short enrolment from the person being predicted for buys.

    python scripts/personalise.py wesad_features.npz --fraction 0.05

The model stays leave-one-subject-out throughout; only the calibrator sees the
person's own data. Enrolment windows and everything overlapping them are
removed from the evaluation, so the three columns -- uncalibrated, cohort
calibrated, personally calibrated -- are scored on identical rows.

Requires the ml extra: ``pip install -e ".[ml]"``.
"""

from __future__ import annotations

import argparse

from physioml.dataset import FeatureTable
from physioml.evaluation.personalisation import personalise
from physioml.models.classical import MODELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", help="path to the .npz written by build_features.py")
    parser.add_argument("--model", default="logistic", choices=sorted(MODELS))
    parser.add_argument(
        "--fraction",
        type=float,
        nargs="*",
        default=[0.05, 0.1, 0.2],
        help="share of each condition given to the calibrator",
    )
    parser.add_argument(
        "--strategy",
        default="per_condition",
        choices=("per_condition", "prospective", "blocks", "prefix"),
        help="where in the session the enrolment is taken from",
    )
    parser.add_argument("--method", default="isotonic", choices=("isotonic", "sigmoid"))
    args = parser.parse_args()

    table = FeatureTable.load(args.table)
    print(table.summary())
    print(f"model {args.model}, enrolment by {args.strategy}, {args.method}\n")

    for fraction in args.fraction:
        result = personalise(
            table,
            MODELS[args.model],
            fraction=fraction,
            strategy=args.strategy,
            method=args.method,
        )
        print(f"=== enrolment: {fraction:.0%} of each condition ===")
        print(result.table())
        if result.skipped:
            print(
                f"not scored: {', '.join(result.skipped)} -- the enrolment or the "
                "evaluation held a single class"
            )
        print()


if __name__ == "__main__":
    main()
