"""Draw the ablation as a figure, from the JSON the comparison wrote.

    python scripts/compare.py wesad_features.npz --json docs/data/wesad_ablation.json
    python scripts/figure_ablation.py docs/data/wesad_ablation.json \
        --output docs/images/wesad-ablation.png

Reads a measurement and draws it. It does not compute anything, which is the
point: a figure that recomputed its own numbers would be a second place they
could differ from the ones in the text, and a figure with numbers typed into it
by hand would be a third.

What it shows is what the ablation is for -- not which sensor scores highest,
but which ones carry something the rest of the array does not, and how much of
that survives being measured across fifteen participants rather than pooled
over windows. Two of the four intervals include zero. Those are drawn the same
size as the others and marked, because an interval that includes zero still
bounds the effect -- it says the direction is uncertain, not that there is
nothing there -- and a bar without its interval cannot say either.

Requires matplotlib: ``pip install -e ".[figures]"``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

INK = "#1b1b1b"
MUTED = "#8a8a8a"
ESTABLISHED = "#b03030"
UNCERTAIN = "#9a9a9a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="JSON from scripts/compare.py --json")
    parser.add_argument("--output", type=Path, required=True, help="PNG to write")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    found = json.loads(args.source.read_text())
    # Largest cost first, so the eye starts at the sensor that matters most.
    rows = sorted(found["comparisons"], key=lambda c: c["mean"])
    names = [c["removed"].replace("wrist ", "") for c in rows]

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(9.6, 0.62 * len(rows) + 1.85))
    y = np.arange(len(rows))[::-1]

    ax.axvline(0.0, color=INK, lw=1.0, zorder=1)

    for position, comparison in zip(y, rows, strict=True):
        colour = UNCERTAIN if comparison["crosses_zero"] else ESTABLISHED

        # Every participant, so the interval is visibly a summary of fifteen
        # paired differences rather than a decoration on a bar.
        per_subject = list(comparison["per_subject"].values())
        jitter = rng.uniform(-0.16, 0.16, len(per_subject))
        ax.scatter(
            per_subject,
            position + jitter,
            s=14,
            color=MUTED,
            alpha=0.55,
            zorder=2,
            linewidths=0,
        )

        low, high = comparison["interval"]
        ax.plot([low, high], [position, position], color=colour, lw=2.6, zorder=3)
        for edge in (low, high):
            ax.plot(
                [edge, edge],
                [position - 0.1, position + 0.1],
                color=colour,
                lw=2.6,
                zorder=3,
            )
        ax.scatter(
            [comparison["mean"]],
            [position],
            s=62,
            color=colour,
            zorder=4,
            edgecolor="white",
            linewidths=1.0,
        )

        ax.annotate(
            f"{comparison['mean']:+.3f}  [{low:+.3f}, {high:+.3f}]"
            + ("\nincludes zero" if comparison["crosses_zero"] else ""),
            xy=(1.0, position),
            xycoords=("axes fraction", "data"),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=UNCERTAIN if comparison["crosses_zero"] else INK,
            annotation_clip=False,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([f"without {n}" for n in names], fontsize=10)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(
        "change in balanced accuracy when the sensor is removed\n"
        "(paired within participant; negative means the sensor was carrying something)",
        fontsize=9,
    )
    subtitle = (
        f"{found['model']}, leave-one-participant-out, "
        f"{found['subjects']} participants, {found['rows']:,} windows"
    )
    if found.get("baseline"):
        subtitle += f"\nall {len(rows) and ''}features together: " + (
            f"{found['baseline']:.3f} balanced accuracy"
        )
    ax.set_title(
        "What each wrist sensor contributes to WESAD stress detection\n" + subtitle,
        fontsize=10.5,
        loc="left",
        pad=12,
    )
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e6e6e6", lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    fig.text(
        0.055,
        0.035,
        "Grey dots are individual participants. Bars are 95% percentile bootstrap "
        "intervals over participants,\nnot over windows: overlapping 60-second "
        "windows are not independent observations. Red marks an interval\nthat "
        "excludes zero; grey marks one that includes it, where the direction of "
        "the effect remains uncertain.",
        fontsize=7.6,
        color=MUTED,
        va="bottom",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.135, right=0.775, top=0.775, bottom=0.30)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.18)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
