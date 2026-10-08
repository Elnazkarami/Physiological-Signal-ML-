"""Ask whether a difference between two feature sets is a difference.

    python scripts/compare.py wesad_features.npz --against ACC EDA

Pairs the two configurations participant by participant and reports a
percentile bootstrap interval over participants -- not over windows, which are
not independent observations of anything.

Requires the ml extra: ``pip install -e ".[ml]"``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from physioml.dataset import FeatureTable
from physioml.evaluation.ablation import channel_groups, device_groups, signal_groups
from physioml.evaluation.comparison import paired_difference, table
from physioml.evaluation.run import evaluate
from physioml.evaluation.splits import leave_one_subject_out
from physioml.models.classical import MODELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", help="path to the .npz written by build_features.py")
    parser.add_argument("--model", default="logistic", choices=sorted(MODELS))
    parser.add_argument(
        "--against",
        nargs="*",
        help="signal groups to remove one at a time; default is every group",
    )
    parser.add_argument(
        "--positive",
        default="stress",
        help="positive class, or 'none' to score every label as its own class",
    )
    parser.add_argument(
        "--metric",
        default="balanced_accuracy",
        choices=("balanced_accuracy", "auc", "kappa"),
    )
    parser.add_argument(
        "--by",
        default="signal",
        choices=("signal", "device", "channel"),
        help="what a group is: one sensor, one whole device, or one recording channel",
    )
    parser.add_argument(
        "--json",
        type=Path,
        metavar="PATH",
        help="also write the comparison as JSON, including every per-participant "
        "difference. What the figure in docs/images is drawn from, so that the "
        "figure is a view of a measurement rather than a second place numbers "
        "are typed in by hand.",
    )
    args = parser.parse_args()

    loaded = FeatureTable.load(args.table)
    print(loaded.summary())

    def run(columns, name):
        return evaluate(
            loaded.select(columns),
            MODELS[args.model],
            leave_one_subject_out(loaded.subject_ids),
            model_name=name,
            positive=None if args.positive == "none" else args.positive,
        )

    whole = run(list(loaded.feature_names), "everything")
    groups = {
        "signal": signal_groups,
        "device": device_groups,
        "channel": channel_groups,
    }[args.by]()
    chosen = args.against or [
        g for g, names in groups.items() if any(n in loaded.feature_names for n in names)
    ]

    found = []
    for group in chosen:
        remove = {n for n in groups.get(group, ()) if n in loaded.feature_names}
        if not remove:
            continue
        rest = [n for n in loaded.feature_names if n not in remove]
        if not rest:
            continue
        found.append(
            paired_difference(whole, run(rest, f"without {group}"), metric=args.metric)
        )

    print()
    print(table(found, f"everything ({args.metric})"))
    print()
    for difference in found:
        print(f"  {difference.verdict()}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "table": args.table,
                    "model": args.model,
                    "metric": args.metric,
                    "by": args.by,
                    "positive": args.positive,
                    "rows": len(loaded),
                    "subjects": len(loaded.subject_ids),
                    "baseline": whole.summary.get(f"{args.metric}_mean"),
                    "comparisons": [
                        {
                            "removed": d.right.removeprefix("without "),
                            "metric": d.metric,
                            "mean": d.mean,
                            "interval": list(d.interval),
                            "crosses_zero": d.crosses_zero,
                            "better": d.better,
                            "worse": d.worse,
                            "tied": d.tied,
                            "n": d.n,
                            "per_subject": d.per_subject,
                            "verdict": d.verdict(),
                        }
                        for d in found
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
