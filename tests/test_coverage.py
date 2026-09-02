"""Counting what a pipeline can answer for, beside how well it answers.

The failure this guards against is not a wrong score. It is a right score over
a cohort quietly smaller than the one the reader has in mind.
"""

from __future__ import annotations

import numpy as np

from physioml.dataset import FeatureTable
from physioml.evaluation.coverage import (
    by_condition_table,
    common_subjects,
    compare,
    coverage_of,
)

SUBJECTS = ["S2", "S3", "S4"]


def made(counts: dict[str, dict[str, int]]) -> FeatureTable:
    """A table from {subject: {condition: rows}}."""
    subjects, labels = [], []
    for subject, conditions in counts.items():
        for condition, n in conditions.items():
            subjects += [subject] * n
            labels += [condition] * n
    rows = len(subjects)
    return FeatureTable(
        feature_names=("a",),
        values=np.zeros((rows, 1)),
        subjects=np.array(subjects),
        labels=np.array(labels),
        window_ids=tuple(f"w{i}" for i in range(rows)),
        window_starts=np.arange(rows, dtype=float) * 30.0,
        feature_set_version="test",
        qc_policy_version="test",
    )


def test_it_counts_rows_by_subject_and_condition():
    table = made(
        {"S2": {"stress": 10, "baseline": 30}, "S3": {"stress": 5, "baseline": 15}}
    )
    found = coverage_of(table, "wrist")
    assert found.rows == 60
    assert found.by_subject == {"S2": 40, "S3": 20}
    assert found.by_condition == {"baseline": 45, "stress": 15}
    assert found.by_subject_condition[("S2", "stress")] == 10


def test_a_subject_with_no_positive_windows_is_not_scorable():
    """The case that matters: they are absent from the result, not scored
    badly in it, so the mean is over a smaller cohort than it appears."""
    table = made(
        {
            "S2": {"stress": 10, "baseline": 30},
            "S3": {"stress": 5, "baseline": 15},
            "S16": {"baseline": 40},
        }
    )
    found = coverage_of(table, "chest")
    assert set(found.scorable()) == {"S2", "S3"}
    assert found.missing() == ("S16",)


def test_a_subject_with_only_positive_windows_is_not_scorable_either():
    table = made({"S2": {"stress": 10, "baseline": 30}, "S4": {"stress": 20}})
    assert coverage_of(table, "x").missing() == ("S4",)


def test_the_common_subset_is_what_two_configurations_share():
    """Comparing on anything else also measures who each one dropped."""
    wrist = made({s: {"stress": 5, "baseline": 15} for s in SUBJECTS})
    chest = made(
        {
            "S2": {"stress": 5, "baseline": 15},
            "S3": {"baseline": 20},
            "S4": {"stress": 5, "baseline": 15},
        }
    )
    shared = common_subjects([coverage_of(wrist, "w"), coverage_of(chest, "c")])
    assert shared == ["S2", "S4"]


def test_the_comparison_names_who_is_missing_rather_than_only_counting():
    wrist = made({s: {"stress": 5, "baseline": 15} for s in SUBJECTS})
    chest = made(
        {
            "S2": {"stress": 5, "baseline": 15},
            "S3": {"baseline": 20},
            "S4": {"baseline": 20},
        }
    )
    text = compare([coverage_of(wrist, "wrist only"), coverage_of(chest, "wrist + chest")])
    assert "S3" in text and "S4" in text
    assert "wrist only" in text and "wrist + chest" in text


def test_retention_is_reported_against_a_reference_table():
    wrist = made(
        {"S2": {"stress": 10, "baseline": 10}, "S16": {"stress": 10, "baseline": 10}}
    )
    chest = made({"S2": {"stress": 10, "baseline": 10}, "S16": {"baseline": 10}})
    text = by_condition_table(coverage_of(chest, "c"), coverage_of(wrist, "w"))
    lines = {line.split()[0]: line for line in text.splitlines() if line[:1] == "S"}
    assert "0.00" in lines["S16"], "no stress windows retained"
    assert "1.00" in lines["S2"]


def test_an_empty_comparison_does_not_raise():
    assert common_subjects([]) == []
