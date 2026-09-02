"""Giving the calibrator a little of the person it is predicting for.

Cohort calibration [fitted on held-out subjects] narrows the spread of stated
confidence across people and cannot close it. The measurement that says so:
every WESAD participant's true stress share is between 0.21 and 0.24, fixed by
the protocol, while the model's average stated probability runs from 0.187 for
one subject to 0.475 for another. A single global calibrator cannot correct
both, because one is wrong in a direction specific to that person.

The obvious remedy is a short enrolment -- a few labelled minutes from the
person themselves, before the model is trusted on them. This module measures
what that buys, against the two things it has to beat: no calibration, and
cohort calibration on the same evaluation rows.

**The model is never trained on the enrolment.** It stays leave-one-subject-out
throughout. Only the calibrator sees the person's own data, which is what makes
this a statement about calibration rather than about fine-tuning.

**Enrolment windows and evaluation windows may not overlap.** These windows are
60 seconds long at a 5-second stride, so consecutive rows share 55 seconds of
signal. A calibration set drawn at random from a person's rows would sit almost
on top of the evaluation set and report a personalisation result that is mostly
an echo. Every strategy here separates the two in time by at least one window
length, and a test asserts it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from physioml.dataset import FeatureTable
from physioml.evaluation.metrics import Scores, expected_calibration_error, score
from physioml.evaluation.splits import leave_one_subject_out
from physioml.models.calibration import SubjectCalibrated, _logit

WINDOW_SECONDS = 60.0
"""Assumed window length, and therefore the minimum separation between an
enrolment window and an evaluation one."""


@dataclass(frozen=True)
class PersonalScores:
    """One subject, scored three ways on the same rows."""

    subject: str
    enrolment_rows: int
    evaluation_rows: int
    enrolment_minutes: float
    uncalibrated: Scores
    cohort: Scores
    personal: Scores
    prevalence: Scores
    """A constant probability, fitted on the enrolment labels alone.

    The baseline that separates calibration from arithmetic. This protocol
    fixes each participant's stress share near 22%, so a calibrator that has
    learned only "answer 0.22 every time" would look well calibrated without
    having learned anything about the person. If personal calibration cannot
    beat this, it has not earned its enrolment."""

    def line(self) -> str:
        return (
            f"{self.subject:5} {self.enrolment_rows:5}  {self.enrolment_minutes:6.1f}   "
            f"{self.uncalibrated.ece:6.3f}  {self.cohort.ece:6.3f}  "
            f"{self.prevalence.ece:6.3f}  {self.personal.ece:6.3f}"
        )


@dataclass(frozen=True)
class Personalisation:
    """What a short enrolment did to the stated confidence, per subject."""

    fraction: float
    strategy: str
    subjects: tuple[PersonalScores, ...]
    skipped: tuple[str, ...] = ()
    """Subjects whose enrolment could not be used, with the reason folded into
    the count: an enrolment holding a single class has nothing to calibrate."""

    def summary(self) -> dict[str, float]:
        def spread(values: list[float]) -> tuple[float, float, float]:
            array = np.array(values)
            return float(array.mean()), float(array.std()), float(array.max())

        rows = self.subjects
        found: dict[str, float] = {}
        for name, pick in (
            ("uncalibrated", lambda s: s.uncalibrated),
            ("cohort", lambda s: s.cohort),
            ("personal", lambda s: s.personal),
            ("prevalence", lambda s: s.prevalence),
        ):
            mean, sd, worst = spread([pick(s).ece for s in rows])
            found[f"{name}_ece_mean"] = mean
            found[f"{name}_ece_sd"] = sd
            found[f"{name}_ece_worst"] = worst
            found[f"{name}_brier_mean"] = float(np.mean([pick(s).brier for s in rows]))
            found[f"{name}_balanced_accuracy_mean"] = float(
                np.mean([pick(s).balanced_accuracy for s in rows])
            )
        found["enrolment_minutes_mean"] = float(
            np.mean([s.enrolment_minutes for s in rows])
        )
        return found

    def table(self) -> str:
        header = (
            f"{'subj':5} {'rows':>5}  {'enrol m':>7}   "
            f"{'no cal':>6}  {'cohort':>6}  {'preval':>6}  {'person':>6}"
        )
        lines = [header, "-" * len(header)]
        lines.extend(s.line() for s in self.subjects)
        found = self.summary()
        lines.append("-" * len(header))
        lines.append(
            f"{'mean':5} {'':5}  {found['enrolment_minutes_mean']:7.1f}   "
            f"{found['uncalibrated_ece_mean']:6.3f}  {found['cohort_ece_mean']:6.3f}  "
            f"{found['prevalence_ece_mean']:6.3f}  {found['personal_ece_mean']:6.3f}"
        )
        lines.append(
            f"{'sd':5} {'':5}  {'':7}   "
            f"{found['uncalibrated_ece_sd']:6.3f}  {found['cohort_ece_sd']:6.3f}  "
            f"{found['prevalence_ece_sd']:6.3f}  {found['personal_ece_sd']:6.3f}"
        )
        lines.append(
            f"{'worst':5} {'':5}  {'':7}   "
            f"{found['uncalibrated_ece_worst']:6.3f}  {found['cohort_ece_worst']:6.3f}  "
            f"{found['prevalence_ece_worst']:6.3f}  {found['personal_ece_worst']:6.3f}"
        )
        return "\n".join(lines)


# ── choosing the enrolment ───────────────────────────────────────────────────


def enrolment(
    starts: np.ndarray,
    fraction: float,
    *,
    strategy: str = "per_condition",
    blocks: int = 4,
    labels: np.ndarray | None = None,
    window_seconds: float = WINDOW_SECONDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Which of one subject's rows to calibrate on, and which to score on.

    Enrolment is taken in **contiguous blocks**, not as scattered windows, and
    that is a structural necessity rather than a convenience. Every enrolment
    window forces a window-length exclusion on each side of it, so a scattered
    window costs two minutes of session to buy one. Twenty-four of them across
    a twenty-minute recording consume the recording: the first version of this
    function returned an empty evaluation set for every subject, which is the
    honest arithmetic of the idea rather than a bug in it. A block costs the
    same two minutes however long it is.

    Four strategies, and the differences between them are findings rather
    than options.

    ``prospective`` fits on a prefix of the session and scores the remainder,
    which is the only one of these a deployment could actually perform: nothing
    the calibrator learned came from after the moment it was fitted.

    ``per_condition`` takes the beginning of each condition the session passes
    through. It is the default because it is the only one that reliably works:
    a calibrator needs examples of both classes, and on a protocol that runs
    its conditions in blocks the other two can miss the one that matters.

    ``blocks`` spreads a few evenly across the session, which is what a
    schedule that samples somebody periodically would give. On a blocked
    protocol it can land entirely in the wrong conditions -- four blocks
    covering a tenth of a WESAD session miss the stress episode completely.

    ``prefix`` takes one block at the beginning, which is what enrolling
    somebody before a session starts would give. On a blocked protocol that is
    one condition, with nothing to calibrate a two-class probability from.

    The practical statement all three make together: personalising a stress
    model needs labelled stress from that person. Time on the device is not
    enough, and neither is a lot of it.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be between 0 and 1, got {fraction}")
    if np.isnan(starts).any():
        raise ValueError(
            "this table has no window times; it was built before they were "
            "recorded, and an enrolment cannot be separated in time without them"
        )
    if starts.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    if strategy == "prospective":
        if labels is None:
            raise ValueError(
                "prospective enrolment needs the label of each row, to find the "
                "point by which every condition has been seen at least once"
            )
        return _prospective(starts, labels, fraction, window_seconds)
    if strategy == "per_condition":
        if labels is None:
            raise ValueError(
                "per_condition enrolment needs the label of each row; it takes "
                "the start of every condition the session passes through"
            )
        return _per_condition(starts, labels, fraction, window_seconds)
    if strategy == "prefix":
        count = 1
    elif strategy == "blocks":
        count = max(int(blocks), 1)
    else:
        raise ValueError(f"unknown enrolment strategy {strategy!r}")

    first, last = float(starts.min()), float(starts.max())
    duration = last - first
    if duration <= 0:
        return np.zeros(0, dtype=int), np.arange(starts.size)

    per_block = duration * fraction / count
    segment = duration / count
    spans = [(first + i * segment, first + i * segment + per_block) for i in range(count)]

    in_block = np.zeros(starts.size, dtype=bool)
    near_block = np.zeros(starts.size, dtype=bool)
    for begin, finish in spans:
        in_block |= (starts >= begin) & (starts <= finish)
        near_block |= (starts > begin - window_seconds) & (starts < finish + window_seconds)

    enrol = np.flatnonzero(in_block)
    evaluation = np.flatnonzero(~near_block)
    return enrol, evaluation


def covered_minutes(starts: np.ndarray, window_seconds: float = WINDOW_SECONDS) -> float:
    """How much wall-clock time a set of windows actually covers.

    Not the count of windows. These overlap by 55 of their 60 seconds, so
    adding their lengths reports an enrolment of 107 minutes for a session that
    only ran for 60 -- and the cost of the method is exactly the number this
    has to get right.
    """
    if starts.size == 0:
        return 0.0
    ordered = np.sort(starts)
    total = 0.0
    begin, finish = float(ordered[0]), float(ordered[0]) + window_seconds
    for when in ordered[1:]:
        if when <= finish:
            finish = max(finish, float(when) + window_seconds)
        else:
            total += finish - begin
            begin, finish = float(when), float(when) + window_seconds
    return (total + finish - begin) / 60.0


def _prospective(
    starts: np.ndarray,
    labels: np.ndarray,
    fraction: float,
    window_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Everything before a cut, against everything after it.

    The honest version of an enrolment, and the one a deployment could
    actually perform: the calibrator sees a prefix of the session and is
    scored on the remainder, so nothing it learned came from after the moment
    it was fitted.

    The cut is placed at the earliest point by which every condition has been
    seen, plus enough of the session to reach ``fraction``. That is what makes
    the comparison with ``per_condition`` meaningful rather than unfair -- both
    calibrators get examples of both classes; only this one is forbidden to
    look forward. It is also the number a deployment cares about, and it is
    usually much larger than the labelled minutes: waiting for a condition that
    occurs late means waiting through everything before it.
    """
    order = np.argsort(starts, kind="stable")
    ordered_starts = starts[order]
    ordered_labels = labels[order]

    first, last = float(ordered_starts[0]), float(ordered_starts[-1])
    duration = last - first
    if duration <= 0:
        return np.zeros(0, dtype=int), np.arange(starts.size)

    # The earliest moment at which both classes have appeared.
    seen: set[str] = set()
    complete = None
    for position, label in enumerate(ordered_labels):
        seen.add(str(label))
        if len(seen) >= len(set(ordered_labels.tolist())):
            complete = float(ordered_starts[position])
            break
    if complete is None:
        return np.zeros(0, dtype=int), np.arange(starts.size)

    cut = max(complete, first + duration * fraction)
    enrol = np.flatnonzero(starts <= cut)
    evaluation = np.flatnonzero(starts > cut + window_seconds)
    return enrol, evaluation


def _per_condition(
    starts: np.ndarray,
    labels: np.ndarray,
    fraction: float,
    window_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The opening slice of every contiguous condition in the session."""
    order = np.argsort(starts, kind="stable")
    ordered_labels = labels[order]
    ordered_starts = starts[order]

    boundaries = np.flatnonzero(ordered_labels[1:] != ordered_labels[:-1]) + 1
    runs = np.split(np.arange(order.size), boundaries)

    in_block = np.zeros(starts.size, dtype=bool)
    near_block = np.zeros(starts.size, dtype=bool)
    for run in runs:
        if run.size == 0:
            continue
        begin = float(ordered_starts[run[0]])
        finish = begin + (float(ordered_starts[run[-1]]) - begin) * fraction
        in_block |= (starts >= begin) & (starts <= finish)
        near_block |= (starts > begin - window_seconds) & (starts < finish + window_seconds)

    return np.flatnonzero(in_block), np.flatnonzero(~near_block)


# ── the experiment ───────────────────────────────────────────────────────────


def _fit_personal(probability: np.ndarray, truth: np.ndarray, method: str) -> Any:
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
            probability, truth
        )
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression().fit(_logit(probability).reshape(-1, 1), truth)


def _apply(calibrator: Any, probability: np.ndarray, method: str) -> np.ndarray:
    if method == "isotonic":
        return np.clip(calibrator.predict(probability), 0.0, 1.0)
    return np.clip(
        calibrator.predict_proba(_logit(probability).reshape(-1, 1))[:, 1], 0.0, 1.0
    )


def personalise(
    table: FeatureTable,
    model_factory: Callable[[], Any],
    *,
    fraction: float = 0.1,
    strategy: str = "per_condition",
    blocks: int = 4,
    method: str = "isotonic",
    positive: str = "stress",
    subjects: Iterable[str] | None = None,
) -> Personalisation:
    """Score each subject uncalibrated, cohort-calibrated, and personalised.

    All three are scored on exactly the same rows -- the ones left after the
    enrolment windows and everything overlapping them are removed -- so the
    three numbers differ only by what calibrated them.
    """
    labels = table.binary(positive)
    chosen = list(subjects) if subjects is not None else table.subject_ids
    scored: list[PersonalScores] = []
    skipped: list[str] = []

    for split in leave_one_subject_out(table.subject_ids):
        subject = split.test_subjects[0]
        if subject not in chosen:
            continue
        train_rows, subject_rows = split.mask(table.subjects)
        if len(np.unique(labels[train_rows])) < 2:
            skipped.append(subject)
            continue

        starts = table.window_starts[subject_rows]
        enrol_local, evaluate_local = enrolment(
            starts,
            fraction,
            strategy=strategy,
            blocks=blocks,
            labels=table.labels[subject_rows],
        )
        enrol_rows = subject_rows[enrol_local]
        evaluate_rows = subject_rows[evaluate_local]

        y_enrol = labels[enrol_rows]
        y_evaluate = labels[evaluate_rows]
        if len(np.unique(y_enrol)) < 2 or len(np.unique(y_evaluate)) < 2:
            # An enrolment of one class calibrates nothing, and an evaluation
            # of one class cannot be scored. Both are reported, not patched.
            skipped.append(subject)
            continue

        base = model_factory()
        base.fit(table.values[train_rows], labels[train_rows])
        raw_evaluate = base.predict_proba(table.values[evaluate_rows])[:, 1]
        predicted = base.predict(table.values[evaluate_rows])

        cohort_model = SubjectCalibrated(model_factory, method=method)
        cohort_model.fit(
            table.values[train_rows],
            labels[train_rows],
            groups=table.subjects[train_rows],
        )
        cohort_evaluate = cohort_model.predict_proba(table.values[evaluate_rows])[:, 1]

        personal = _fit_personal(
            base.predict_proba(table.values[enrol_rows])[:, 1], y_enrol, method
        )
        personal_evaluate = _apply(personal, raw_evaluate, method)

        # The arithmetic baseline: the share of the enrolment that was
        # positive, stated for every window regardless of the signal.
        constant = np.full(y_evaluate.size, float(np.mean(y_enrol)))

        who = table.subjects[evaluate_rows]
        scored.append(
            PersonalScores(
                subject=subject,
                enrolment_rows=int(enrol_rows.size),
                evaluation_rows=int(evaluate_rows.size),
                enrolment_minutes=covered_minutes(table.window_starts[enrol_rows]),
                uncalibrated=score(y_evaluate, predicted, raw_evaluate, who),
                cohort=score(y_evaluate, predicted, cohort_evaluate, who),
                personal=score(y_evaluate, predicted, personal_evaluate, who),
                prevalence=score(y_evaluate, predicted, constant, who),
            )
        )

    if not scored:
        raise ValueError("no subject had a usable enrolment")
    return Personalisation(fraction, strategy, tuple(scored), tuple(skipped))


def calibration_gap(table: FeatureTable, probability: np.ndarray, positive: str) -> float:
    """Convenience: the expected calibration error of one probability vector."""
    return expected_calibration_error(table.binary(positive), probability)
