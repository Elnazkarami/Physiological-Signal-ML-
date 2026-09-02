"""A short enrolment from the person being predicted for.

The claim under test is narrow: giving a calibrator a few labelled minutes of
someone's own data lowers the error in what the model then states about them.
The traps are wider, and most of these tests are about the traps -- chiefly
that these windows overlap by 55 of their 60 seconds, so an enrolment drawn at
random from a person's rows sits almost on top of the evaluation set and any
improvement it shows is an echo of itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from physioml.dataset import FeatureTable
from physioml.evaluation.personalisation import (
    WINDOW_SECONDS,
    covered_minutes,
    enrolment,
    personalise,
)
from physioml.models.classical import MODELS

SUBJECTS = [f"S{n}" for n in range(2, 12)]
STRIDE = 5.0


def cohort(rows_per_subject: int = 720, seed: int = 0) -> FeatureTable:
    """A cohort whose participants disagree about where the boundary sits.

    Conditions run in blocks, as a protocol does: four stretches of which one
    is the positive class. Interleaving the labels row by row would make every
    condition one row long, which is not a session and would let an enrolment
    sample the same minute it is scored on.

    Each subject carries an offset the model cannot see, so a single global
    decision function is over-confident for some of them and under-confident
    for others -- the situation cohort calibration narrows and cannot close.
    """
    rng = np.random.default_rng(seed)
    values, subjects, labels, starts = [], [], [], []
    block = rows_per_subject // 4
    for index, subject in enumerate(SUBJECTS):
        offset = (index - len(SUBJECTS) / 2) * 0.6
        for i in range(rows_per_subject):
            stressed = i // block == 2
            values.append([offset + rng.normal(1.6 if stressed else 0.0, 1.0)])
            subjects.append(subject)
            labels.append("stress" if stressed else "baseline")
            starts.append(i * STRIDE)
    return FeatureTable(
        feature_names=("x",),
        values=np.array(values),
        subjects=np.array(subjects),
        labels=np.array(labels),
        window_ids=tuple(f"w{i}" for i in range(len(values))),
        window_starts=np.array(starts, dtype=float),
        feature_set_version="test-1",
        qc_policy_version="test-1",
    )


# ── choosing the enrolment ──────────────────────────────────────────────────


@pytest.mark.parametrize("strategy", ["blocks", "prefix"])
def test_no_evaluation_window_shares_signal_with_an_enrolment_window(strategy):
    """The whole experiment rests on this."""
    starts = np.arange(720, dtype=float) * STRIDE
    enrol, evaluate = enrolment(starts, 0.1, strategy=strategy)
    assert enrol.size and evaluate.size
    for when in starts[evaluate]:
        assert np.abs(starts[enrol] - when).min() >= WINDOW_SECONDS


@pytest.mark.parametrize("strategy", ["blocks", "prefix"])
def test_the_two_sets_never_share_a_row(strategy):
    starts = np.arange(720, dtype=float) * STRIDE
    enrol, evaluate = enrolment(starts, 0.2, strategy=strategy)
    assert not set(enrol.tolist()) & set(evaluate.tolist())


def test_blocks_are_spread_across_the_session_and_prefix_sits_at_the_start():
    starts = np.arange(720, dtype=float) * STRIDE
    spread, _ = enrolment(starts, 0.12, strategy="blocks", blocks=4)
    prefix, _ = enrolment(starts, 0.12, strategy="prefix")
    assert starts[spread].max() > starts.max() * 0.7
    assert starts[prefix].max() < starts.max() * 0.2


def test_scattering_single_windows_would_consume_the_session():
    """Why enrolment comes in blocks. Each window costs a window length of
    exclusion on each side, so twenty-four of them eat twenty minutes."""
    starts = np.arange(240, dtype=float) * STRIDE
    _, evaluation = enrolment(starts, 0.1, strategy="blocks", blocks=24)
    assert evaluation.size == 0


def test_covered_time_is_the_union_not_the_sum_of_window_lengths():
    """Windows five seconds apart cover a minute and five seconds between
    them, not two minutes."""
    from physioml.evaluation.personalisation import covered_minutes

    assert covered_minutes(np.array([0.0, 5.0])) == pytest.approx(65.0 / 60.0)
    assert covered_minutes(np.array([0.0])) == pytest.approx(1.0)
    assert covered_minutes(np.array([0.0, 600.0])) == pytest.approx(2.0)
    assert covered_minutes(np.array([])) == 0.0


def test_a_table_without_window_times_is_refused():
    """Rather than treating row order as a timeline."""
    with pytest.raises(ValueError, match="no window times"):
        enrolment(np.full(50, np.nan), 0.1)


@pytest.mark.parametrize("fraction", [0.0, 1.0, 1.5, -0.1])
def test_an_impossible_fraction_is_refused(fraction):
    with pytest.raises(ValueError, match="between 0 and 1"):
        enrolment(np.arange(50, dtype=float), fraction)


def test_an_unknown_strategy_is_refused():
    with pytest.raises(ValueError, match="unknown enrolment strategy"):
        enrolment(np.arange(50, dtype=float), 0.1, strategy="vibes")


# ── what the enrolment buys ─────────────────────────────────────────────────


def test_a_personal_calibrator_beats_a_cohort_one_on_its_own_subject():
    made = cohort()
    result = personalise(made, MODELS["logistic"], fraction=0.2)
    found = result.summary()
    assert found["personal_ece_mean"] < found["cohort_ece_mean"]
    assert found["cohort_ece_mean"] <= found["uncalibrated_ece_mean"]


def test_personalisation_narrows_the_spread_across_people():
    """The measure the cohort calibrator could only halve."""
    made = cohort()
    found = personalise(made, MODELS["logistic"], fraction=0.2).summary()
    assert found["personal_ece_worst"] < found["uncalibrated_ece_worst"]


def test_calibration_does_not_move_the_decision():
    """All three are scored on one set of predictions; only the stated
    probability differs, so accuracy must be identical across the three."""
    made = cohort()
    result = personalise(made, MODELS["logistic"], fraction=0.2)
    for row in result.subjects:
        assert row.uncalibrated.balanced_accuracy == pytest.approx(
            row.personal.balanced_accuracy
        )
        assert row.uncalibrated.balanced_accuracy == pytest.approx(
            row.cohort.balanced_accuracy
        )


def test_every_scored_subject_reports_how_much_of_itself_it_was_given():
    made = cohort()
    result = personalise(made, MODELS["logistic"], fraction=0.2)
    for row in result.subjects:
        assert row.enrolment_rows > 0
        assert row.evaluation_rows > row.enrolment_rows
        # Not one minute per window: they overlap by 55 seconds each, and
        # summing their lengths reports more enrolment than the session ran for.
        assert row.enrolment_minutes < row.enrolment_rows


def test_an_enrolment_that_misses_the_positive_class_calibrates_nothing():
    """A finding, not a preference. A protocol runs its conditions in blocks,
    so the opening tenth of a session is one condition -- and four blocks
    spread evenly across it can miss the stress episode entirely. Personalising
    a stress model needs labelled stress from that person."""
    blocks = cohort()
    with pytest.raises(ValueError, match="no subject had a usable enrolment"):
        personalise(blocks, MODELS["logistic"], fraction=0.1, strategy="prefix")

    # Whether evenly spaced blocks find the stress episode depends on where it
    # happens to fall, so this asserts only that per-condition enrolment does
    # not depend on that alignment.
    by_condition = personalise(blocks, MODELS["logistic"], fraction=0.2)
    assert len(by_condition.subjects) == len(SUBJECTS)


def test_a_subject_whose_enrolment_holds_one_class_is_named_not_dropped():
    made = cohort()
    labels = made.labels.copy()
    labels[made.subjects == "S5"] = "baseline"
    from dataclasses import replace

    result = personalise(replace(made, labels=labels), MODELS["logistic"], fraction=0.2)
    assert "S5" in result.skipped
    assert "S5" not in {row.subject for row in result.subjects}


# ── the only order a deployment could use ───────────────────────────────────


def test_a_prospective_enrolment_never_looks_forward():
    """Every enrolment window precedes every evaluation window."""
    made = cohort()
    for subject in SUBJECTS[:3]:
        rows = made.subjects == subject
        starts = made.window_starts[rows]
        labels = made.labels[rows]
        enrol, evaluate = enrolment(starts, 0.2, strategy="prospective", labels=labels)
        assert enrol.size and evaluate.size
        assert starts[enrol].max() < starts[evaluate].min()


def test_a_prospective_enrolment_leaves_a_window_length_gap():
    made = cohort()
    rows = made.subjects == SUBJECTS[0]
    starts, labels = made.window_starts[rows], made.labels[rows]
    enrol, evaluate = enrolment(starts, 0.2, strategy="prospective", labels=labels)
    assert starts[evaluate].min() - starts[enrol].max() > WINDOW_SECONDS


def test_a_prospective_enrolment_waits_for_both_classes():
    """Otherwise it has nothing to calibrate, and the comparison with the
    retrospective version would be unfair rather than informative."""
    made = cohort()
    rows = made.subjects == SUBJECTS[0]
    starts, labels = made.window_starts[rows], made.labels[rows]
    enrol, _ = enrolment(starts, 0.01, strategy="prospective", labels=labels)
    assert len(set(labels[enrol])) == 2


def test_a_prospective_enrolment_costs_the_time_it_waits_through():
    """The distinction the retrospective table hides: for a prefix, labelled
    signal and elapsed session are the same number."""
    made = cohort()
    rows = made.subjects == SUBJECTS[0]
    starts, labels = made.window_starts[rows], made.labels[rows]
    prefix, _ = enrolment(starts, 0.05, strategy="prospective", labels=labels)
    blocks, _ = enrolment(starts, 0.05, strategy="per_condition", labels=labels)
    assert covered_minutes(starts[prefix]) > covered_minutes(starts[blocks]) * 2


def test_prospective_enrolment_needs_the_labels():
    with pytest.raises(ValueError, match="prospective enrolment needs the label"):
        enrolment(np.arange(100, dtype=float) * 5.0, 0.2, strategy="prospective")
