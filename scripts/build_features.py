"""Build a feature table from the WESAD archive.

    python scripts/build_features.py ~/Downloads/WESAD.zip wesad_features.npz

The archive is streamed, never extracted: it is 2.1 GB compressed and roughly
17 GB unpacked, and nothing here needs it on disk twice.

Requires the signal extra: ``pip install -e ".[signal]"``.
"""

from __future__ import annotations

import argparse
import time

from physioml.dataset import build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="path to WESAD.zip")
    parser.add_argument("output", nargs="?", default="wesad_features.npz")
    parser.add_argument("--length", type=float, default=60.0, help="window seconds")
    parser.add_argument("--stride", type=float, default=5.0, help="stride seconds")
    parser.add_argument("--subjects", nargs="*", help="limit to these subject ids")
    args = parser.parse_args()

    started = time.time()
    table = build(
        args.archive,
        length_seconds=args.length,
        stride_seconds=args.stride,
        subjects=args.subjects,
        progress=True,
    )
    table.save(args.output)

    print(f"\n{table.summary()}")
    print(f"dropped incomplete rows: {table.dropped_incomplete}")
    print(f"qc codes: {table.qc_codes}")
    print(f"written to {args.output} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
