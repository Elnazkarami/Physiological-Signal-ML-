"""The temporal experiments, from one command.

    python scripts/sleep_experiment.py sleep_sc.npz --mode all

Three ways of using the structure of a night, on identical folds:

``single_epoch``  each epoch scored alone, which is what every classical model
                  here did before.
``context``       the neighbouring epochs handed over as extra columns.
``gru``           a recurrent network reading the whole night as a sequence.

Grouped five-fold rather than leave-one-subject-out, because the recurrent mode
fits one network per fold and seventy-six of those on a CPU is days. Every mode
uses the same folds, so the comparison between them is unaffected by that.

``--mode gru`` needs the deep extra: ``pip install -e ".[deep]"``.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

from physioml.dataset import FeatureTable
from physioml.evaluation.comparison import paired_difference
from physioml.evaluation.comparison import table as difference_table
from physioml.evaluation.run import evaluate, manifest
from physioml.evaluation.splits import group_k_fold
from physioml.models.classical import MODELS
from physioml.neural.context import with_context

MODES = ("single_epoch", "context", "gru")


def _commit() -> str:
    """The revision this was run at, so a manifest can be placed in time."""
    try:
        found = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent,
        )
        return found.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _versions() -> dict[str, str]:
    found = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "sklearn", "torch"):
        try:
            found[name] = __import__(name).__version__
        except ImportError:
            found[name] = "absent"
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", help="a sleep feature table")
    parser.add_argument("--mode", default="all", choices=(*MODES, "all"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="random_forest", choices=sorted(MODELS))
    parser.add_argument("--epochs", type=int, default=15, help="gru training passes")
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--manifest", help="write folds, versions and scores to JSON")
    args = parser.parse_args()

    base = FeatureTable.load(args.table)
    print(base.summary())
    chosen = list(MODES) if args.mode == "all" else [args.mode]

    def folds():
        return group_k_fold(base.subject_ids, folds=args.folds, seed=args.seed)

    def build(mode: str):
        if mode == "gru":
            from physioml.models.sequence import gru

            return base, lambda: gru(
                epochs=args.epochs, hidden=args.hidden, layers=1, seed=args.seed
            )
        if mode == "context":
            return with_context(base), MODELS[args.model]
        return base, MODELS[args.model]

    runs, timings = {}, {}
    print(f"\n{'mode':16} {'kappa':>14} {'accuracy':>9} {'bal.acc':>8} {'seconds':>8}")
    print("-" * 60)
    for mode in chosen:
        table, factory = build(mode)
        started = time.time()
        result = evaluate(
            table,
            factory,
            folds(),
            model_name=mode,
            positive=None,
            task="sleep_stage",
            dataset_version=Path(args.table).stem,
        )
        timings[mode] = time.time() - started
        runs[mode] = result
        s = result.summary
        print(
            f"{mode:16} {s['kappa_mean']:.3f} ±{s['kappa_sd']:.3f}   "
            f"{s['accuracy_mean']:9.3f} {s['balanced_accuracy_mean']:8.3f} "
            f"{timings[mode]:8.0f}",
            flush=True,
        )
        print(
            "    per-stage:",
            {
                k[len("recall_") :]: round(v, 3)
                for k, v in sorted(s.items())
                if k.startswith("recall_")
            },
            flush=True,
        )

    if len(runs) > 1 and "single_epoch" in runs:
        others = [m for m in chosen if m != "single_epoch"]
        print()
        print(
            difference_table(
                [
                    paired_difference(runs["single_epoch"], runs[m], metric="kappa")
                    for m in others
                ],
                "single_epoch (kappa)",
            )
        )
        if {"context", "gru"} <= set(runs):
            head = paired_difference(runs["context"], runs["gru"], metric="kappa")
            print("\ngru against context: " + head.verdict())

    if args.manifest:
        record = {
            "table": args.table,
            "commit": _commit(),
            "versions": _versions(),
            "folds": args.folds,
            "seed": args.seed,
            "hyperparameters": {
                "model": args.model,
                "epochs": args.epochs,
                "hidden": args.hidden,
            },
            "seconds": timings,
            "runs": {m: manifest(r) for m, r in runs.items()},
        }
        Path(args.manifest).write_text(json.dumps(record, indent=2, sort_keys=True))
        print(f"\nmanifest written to {args.manifest}")


if __name__ == "__main__":
    main()
