"""Carrying the neighbouring epochs into the row being scored.

A scorer reads the epochs either side; every model here sees one epoch alone,
and N1 -- the stage defined by being a transition -- is the one they all fail
on. These tests are mostly about the two ways that can go wrong: taking context
across a participant boundary, and inventing something at the edges.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from physioml.dataset import FeatureTable
from physioml.neural.context import CONTEXT_PREFIX, carried_columns, with_context

SUBJECTS = ["S1", "S2"]


def table(per_subject: int = 8) -> FeatureTable:
    """Each row's value is its own index, so a shift is visible by eye."""
    values, subjects, labels, starts = [], [], [], []
    for offset, subject in enumerate(SUBJECTS):
        for i in range(per_subject):
            values.append([float(i + 100 * offset), float(i)])
            subjects.append(subject)
            labels.append("N2")
            starts.append(i * 30.0)
    return FeatureTable(
        feature_names=("fpz_delta_rel", "fpz_amplitude_p95"),
        values=np.array(values),
        subjects=np.array(subjects),
        labels=np.array(labels),
        window_ids=tuple(f"w{i}" for i in range(len(values))),
        window_starts=np.array(starts),
        feature_set_version="t",
        qc_policy_version="t",
    )


def column(made: FeatureTable, name: str) -> np.ndarray:
    return made.values[:, made.feature_names.index(name)]


def test_the_previous_epoch_arrives_as_a_column():
    made = with_context(table(), offsets=(-1,), smooth=0)
    got = column(made, f"{CONTEXT_PREFIX}prev1_fpz_delta_rel")
    first = made.subjects == "S1"
    # Row i carries row i-1's value; the first row carries its own.
    assert list(got[first]) == [0, 0, 1, 2, 3, 4, 5, 6]


def test_the_next_epoch_arrives_too():
    made = with_context(table(), offsets=(1,), smooth=0)
    got = column(made, f"{CONTEXT_PREFIX}next1_fpz_delta_rel")
    first = made.subjects == "S1"
    assert list(got[first]) == [1, 2, 3, 4, 5, 6, 7, 7]


def test_context_never_crosses_a_participant_boundary():
    """The one thing here that would be leakage, and the one thing a naive
    shift over a stacked array does automatically."""
    made = with_context(table(), offsets=(-1, -2), smooth=0)
    second = made.subjects == "S2"
    for name in (
        f"{CONTEXT_PREFIX}prev1_fpz_delta_rel",
        f"{CONTEXT_PREFIX}prev2_fpz_delta_rel",
    ):
        carried = column(made, name)[second]
        assert (carried >= 100).all(), "S2 must never see S1's epochs"


def test_the_edges_repeat_rather_than_inventing_a_flat_epoch():
    """Padding with zeros would tell the model sleep begins with a flat
    spectrum, which is the signature of a disconnected electrode."""
    made = with_context(table(), offsets=(-2,), smooth=0)
    got = column(made, f"{CONTEXT_PREFIX}prev2_fpz_delta_rel")
    first = made.subjects == "S1"
    assert list(got[first])[:3] == [0, 0, 0]
    assert 0 not in list(
        column(made, f"{CONTEXT_PREFIX}prev2_fpz_delta_rel")[made.subjects == "S2"]
    )


def test_the_rolling_mean_is_centred_on_the_row():
    made = with_context(table(), offsets=(), smooth=1)
    got = column(made, f"{CONTEXT_PREFIX}mean1_fpz_delta_rel")
    first = made.subjects == "S1"
    # Row 3 averages rows 2, 3, 4 → 3.0; the edges repeat.
    assert got[first][3] == pytest.approx(3.0)
    assert got[first][0] == pytest.approx((0 + 0 + 1) / 3)


def test_rows_are_ordered_by_time_not_by_position():
    """A table built from two nights is not in time order within a subject."""
    made = table()
    shuffled = replace(
        made,
        values=made.values[::-1].copy(),
        subjects=made.subjects[::-1].copy(),
        labels=made.labels[::-1].copy(),
        window_starts=made.window_starts[::-1].copy(),
        window_ids=tuple(reversed(made.window_ids)),
    )
    got = with_context(shuffled, offsets=(-1,), smooth=0)
    rows = got.subjects == "S1"
    pairs = sorted(
        zip(
            got.window_starts[rows],
            column(got, f"{CONTEXT_PREFIX}prev1_fpz_delta_rel")[rows],
            strict=True,
        )
    )
    assert [v for _, v in pairs] == [0, 0, 1, 2, 3, 4, 5, 6]


def test_only_the_useful_columns_are_carried():
    """Context multiplies the columns; carrying every percentile would cost
    more to fit than the neighbourhood is worth."""
    made = table()
    assert carried_columns(made.feature_names) == ["fpz_delta_rel"]
    got = with_context(made, offsets=(-1, 1), smooth=0)
    assert len(got.feature_names) == 2 + 2


def test_the_original_columns_are_untouched():
    made = table()
    got = with_context(made)
    assert got.feature_names[: len(made.feature_names)] == made.feature_names
    assert np.array_equal(got.values[:, : made.values.shape[1]], made.values)


def test_a_table_with_no_times_is_refused():
    made = table()
    without = replace(made, window_starts=np.full(len(made), np.nan))
    with pytest.raises(ValueError, match="no window times"):
        with_context(without)
