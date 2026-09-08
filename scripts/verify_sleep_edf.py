"""Check a Sleep-EDF directory: which nights are complete, which are not.

    python scripts/verify_sleep_edf.py ~/Downloads/sleep-edf
    python scripts/verify_sleep_edf.py ~/Downloads/sleep-edf --repair-list missing.txt

The dataset is fetched over a rate-limited connection and a partial download
leaves a file that exists, has a plausible size, and is not a recording. Two
weaker checks are not enough on their own: counting files misses the truncated
ones, and checking file size misses them too, because a half-downloaded night
is megabytes of perfectly good bytes. The only reliable test is to parse the
header and compare the records it declares against the records that are there,
which is what the reader does anyway.

``--repair-list`` writes the names that need re-fetching, ready to feed back to
curl.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from physioml.io.edf import EDF, EDFError

NAME = re.compile(
    r"^SC4(?P<subject>\d\d)(?P<night>\d)[A-Z0-9]{2}-(?P<kind>PSG|Hypnogram)\.edf$"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="directory holding SC4*.edf files")
    parser.add_argument("--repair-list", help="write the names needing re-fetch here")
    parser.add_argument("--night", type=int, choices=(1, 2), help="check one night only")
    args = parser.parse_args()

    directory = Path(args.directory).expanduser()
    if not directory.is_dir():
        raise SystemExit(f"{directory} is not a directory")

    whole: dict[tuple[str, str], dict[str, Path]] = {}
    broken: list[tuple[Path, str]] = []
    for path in sorted(directory.glob("SC4*.edf")):
        found = NAME.match(path.name)
        if found is None:
            continue
        if args.night and int(found["night"]) != args.night:
            continue
        key = (found["subject"], found["night"])
        try:
            EDF(path)
        except EDFError as exc:
            broken.append((path, str(exc).split(";")[-1].strip()))
            continue
        whole.setdefault(key, {})[found["kind"]] = path

    paired = {k: v for k, v in whole.items() if {"PSG", "Hypnogram"} <= set(v)}
    unpaired = {k: v for k, v in whole.items() if k not in paired}

    by_night = Counter(night for _, night in paired)
    print(f"{directory}")
    print(f"  usable nights:   {len(paired)}", end="")
    if by_night:
        print(
            "  (" + ", ".join(f"night {n}: {c}" for n, c in sorted(by_night.items())) + ")"
        )
    else:
        print()
    print(f"  subjects:        {len({s for s, _ in paired})}")

    if unpaired:
        print(f"  missing a pair:  {len(unpaired)}")
        for (subject, night), found in sorted(unpaired.items())[:10]:
            absent = {"PSG", "Hypnogram"} - set(found)
            print(f"      SC4{subject}{night}: no {', '.join(sorted(absent))}")
    if broken:
        print(f"  unreadable:      {len(broken)}")
        for path, why in broken[:10]:
            print(f"      {path.name}: {why}")

    if args.repair_list:
        names = [path.name for path, _ in broken]
        Path(args.repair_list).write_text("\n".join(names))
        print(f"\n  {len(names)} name(s) written to {args.repair_list}")

    if broken or unpaired:
        raise SystemExit(1)
    print("\n  every night is complete and paired.")


if __name__ == "__main__":
    main()
