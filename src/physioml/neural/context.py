"""Giving a model the epochs either side of the one it is scoring.

A human scorer does not judge a 30-second epoch alone. They read the ones
before and after: an epoch is N1 partly because what precedes it was wake and
what follows is N2, and the same spectrum in a different neighbourhood is a
different stage. Every model in this repository sees one epoch in isolation,
and N1 -- the stage defined almost entirely by being a transition -- is the one
they all fail on.

This adds the neighbourhood back, as ordinary columns, so the same classical
models can use it and the same evaluation can measure whether it helped.

**It is not leakage.** The neighbouring epochs belong to the same participant,
and participants are held out whole, so nothing from a scored subject reaches
the model that scores them. What would be leakage is taking context across a
subject boundary, and the boundaries are respected: an epoch at the edge of a
recording is padded with itself rather than with somebody else's sleep.

**It is also not free.** Context multiplies the columns, and a table with five
times the features takes considerably longer to fit. A subset of the features
is carried rather than all of them -- the ones a scorer actually reads across
epochs, which is the spectral shape and the two supporting channels, not every
percentile.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from physioml.dataset import FeatureTable

CONTEXT_PREFIX = "ctx_"

#: What is worth carrying from a neighbouring epoch. A scorer reading the
#: epochs around this one is looking at the spectral balance, the eye movement
#: and the muscle tone -- not at the ninety-fifth percentile of the amplitude.
DEFAULT_CARRIED = (
    "delta_rel",
    "theta_rel",
    "alpha_rel",
    "sigma_rel",
    "beta_rel",
    "entropy",
    "edge95",
    "hjorth_mobility",
)

DEFAULT_CHANNEL_CARRIED = ("eog_slow_rel", "eog_amplitude_sd", "chin_emg_rms")


def carried_columns(
    names: Sequence[str],
    suffixes: Sequence[str] = DEFAULT_CARRIED,
    exact: Sequence[str] = DEFAULT_CHANNEL_CARRIED,
) -> list[str]:
    """Which columns to carry from neighbouring epochs."""
    chosen = [n for n in names if any(n.endswith(f"_{s}") for s in suffixes)]
    chosen += [n for n in names if n in exact]
    return sorted(set(chosen), key=list(names).index)


def with_context(
    table: FeatureTable,
    *,
    offsets: Sequence[int] = (-2, -1, 1, 2),
    columns: Sequence[str] | None = None,
    smooth: int = 2,
    epoch_seconds: float = 30.0,
    tolerance: float = 1.0,
) -> FeatureTable:
    """The same table, with each row carrying its neighbours' features.

    ``offsets`` are epochs relative to the row: -1 is the epoch before it.
    ``smooth`` adds a centred rolling mean over that many epochs either side,
    which is what a scorer's sense of "the trend here" amounts to.

    Rows are grouped by participant and ordered by time, and the edges of each
    recording are padded by repeating the first and last epoch. Padding with
    zeros would tell the model that sleep begins with a flat spectrum, which is
    not a neutral statement -- it is the signature of a disconnected electrode.
    """
    if np.isnan(table.window_starts).any():
        raise ValueError(
            "this table has no window times; context needs to know which epoch "
            "precedes which, and row order is not a timeline"
        )

    carried = list(columns) if columns is not None else carried_columns(table.feature_names)
    if not carried:
        raise ValueError("no columns to carry as context")
    index = {name: i for i, name in enumerate(table.feature_names)}
    picked = np.array([index[n] for n in carried], dtype=int)

    added: list[str] = []
    for offset in offsets:
        tag = f"prev{-offset}" if offset < 0 else f"next{offset}"
        added += [f"{CONTEXT_PREFIX}{tag}_{n}" for n in carried]
    if smooth:
        added += [f"{CONTEXT_PREFIX}mean{smooth}_{n}" for n in carried]

    values = np.zeros((len(table), len(added)), dtype=float)
    for subject in np.unique(table.subjects):
        rows = np.flatnonzero(table.subjects == subject)
        order = rows[np.argsort(table.window_starts[rows], kind="stable")]
        times = table.window_starts[order]
        # A break wherever the next row is not the next epoch: a dropped epoch
        # mid-night, or the boundary between two nights of the same person.
        breaks = np.flatnonzero(np.abs(np.diff(times) - epoch_seconds) > tolerance) + 1

        for run in np.split(order, breaks):
            if run.size == 0:
                continue
            block = table.values[np.ix_(run, picked)]
            n = block.shape[0]

            written = 0
            for offset in offsets:
                # Clipping repeats this run's own edge epoch, which is the
                # mildest honest statement about what lies outside it.
                shifted = block[np.clip(np.arange(n) + offset, 0, n - 1)]
                values[run, written : written + len(carried)] = shifted
                written += len(carried)
            if smooth:
                window = 2 * smooth + 1
                padded = np.vstack(
                    [
                        np.repeat(block[:1], smooth, axis=0),
                        block,
                        np.repeat(block[-1:], smooth, axis=0),
                    ]
                )
                cumulative = np.cumsum(padded, axis=0)
                cumulative = np.vstack([np.zeros((1, block.shape[1])), cumulative])
                rolling = (cumulative[window:] - cumulative[:-window]) / window
                values[run, written : written + len(carried)] = rolling

    return replace(
        table,
        feature_names=(*table.feature_names, *added),
        values=np.hstack([table.values, values]),
    )
