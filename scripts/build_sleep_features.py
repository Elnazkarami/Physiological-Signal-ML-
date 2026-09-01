"""Build a feature table from a directory of Sleep-EDF recordings.

    python scripts/build_sleep_features.py ~/sleep-edf sleep_features.npz

Each PSG file is paired with its hypnogram, trimmed to the sleep period plus a
margin, and cut into the 30-second epochs the scoring uses. Both nights of a
participant are one subject.

Requires the signal extra: ``pip install -e ".[signal]"``.
"""

from __future__ import annotations

import argparse
import time

from physioml.neural.dataset import build_sleep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="directory holding SC4*.edf files")
    parser.add_argument("output", nargs="?", default="sleep_features.npz")
    parser.add_argument("--subjects", nargs="*", help="limit to these subject ids")
    parser.add_argument(
        "--margin",
        type=float,
        default=30.0,
        help="minutes of recording either side of the sleep period to keep",
    )
    args = parser.parse_args()

    started = time.time()
    table = build_sleep(
        args.directory,
        subjects=args.subjects,
        margin_minutes=args.margin,
        progress=True,
    )
    table.save(args.output)

    print(f"\n{table.summary()}")
    print(f"dropped incomplete rows: {table.dropped_incomplete}")
    print(f"trimmed epochs: {table.qc_codes.get('trimmed_epochs', 0)}")
    print(f"unscored epochs: {table.qc_codes.get('unscored_epochs', 0)}")
    print(f"written to {args.output} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
