"""Holding participants apart.

There is one way to be badly wrong here and it does not announce itself. Windows
cut from one recording share that participant's physiology — their resting heart
rate, their baseline skin conductance, the way their particular wrist sits
against the sensor. A model shown windows from the same person in training and
in test can identify the person and infer the state from that, and it will score
extremely well while having learned nothing about stress.

With a five-second stride the problem is acute: consecutive windows overlap by
55 seconds, so a random split puts near-duplicate rows on both sides. The
resulting number is not merely optimistic, it is measuring something else.

So every split here groups by subject, and :class:`~physioml.core.registry.TrainingRun`
refuses a run whose splits share one. Nothing in this module can produce a split
that violates that, and a test asserts it across every strategy.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Split:
    """One train/test division, named by the subjects on each side."""

    strategy: str
    fold: int
    train_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]
    seed: int | None = None

    def mask(self, subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Row indices for train and test, given a column of subject ids."""
        train = np.isin(subjects, np.array(self.train_subjects))
        test = np.isin(subjects, np.array(self.test_subjects))
        return np.flatnonzero(train), np.flatnonzero(test)

    def __str__(self) -> str:
        return (
            f"{self.strategy} fold {self.fold}: "
            f"{len(self.train_subjects)} train / {len(self.test_subjects)} test "
            f"({', '.join(self.test_subjects)})"
        )


def leave_one_subject_out(subject_ids: list[str]) -> Iterator[Split]:
    """One fold per participant, each held out in turn.

    The most demanding arrangement and the one that answers the question a
    wearable actually poses: this model has never seen this person, and someone
    is about to put it on.
    """
    ordered = sorted(subject_ids, key=_numeric)
    if len(ordered) < 2:
        raise ValueError("leave-one-subject-out needs at least two subjects")
    for fold, held in enumerate(ordered):
        yield Split(
            strategy="leave_one_subject_out",
            fold=fold,
            train_subjects=tuple(s for s in ordered if s != held),
            test_subjects=(held,),
        )


def group_k_fold(subject_ids: list[str], folds: int = 5, seed: int = 0) -> Iterator[Split]:
    """``folds`` disjoint groups of participants.

    Cheaper than leaving each one out, and coarser: with fifteen subjects and
    five folds each estimate rests on three people, so the spread between folds
    says as much about which three as about the model.
    """
    ordered = sorted(subject_ids, key=_numeric)
    if folds < 2 or folds > len(ordered):
        raise ValueError(
            f"{folds} folds is not usable with {len(ordered)} subjects; "
            "it must be between 2 and the number of subjects"
        )
    shuffled = list(ordered)
    np.random.default_rng(seed).shuffle(shuffled)
    groups = [shuffled[i::folds] for i in range(folds)]
    for fold, held in enumerate(groups):
        yield Split(
            strategy="group_k_fold",
            fold=fold,
            train_subjects=tuple(s for s in ordered if s not in held),
            test_subjects=tuple(sorted(held, key=_numeric)),
            seed=seed,
        )


def held_out_cohort(
    subject_ids: list[str], test_fraction: float = 0.3, seed: int = 0
) -> Split:
    """A single division, for when a cohort is set aside once and not revisited.

    Returns one split rather than an iterator, deliberately: this is the shape
    used when a test set is meant to be touched once, and an iterator invites a
    loop over it.
    """
    ordered = sorted(subject_ids, key=_numeric)
    n_test = max(1, round(len(ordered) * test_fraction))
    if n_test >= len(ordered):
        raise ValueError("the held-out cohort would leave nobody to train on")
    shuffled = list(ordered)
    np.random.default_rng(seed).shuffle(shuffled)
    held = sorted(shuffled[:n_test], key=_numeric)
    return Split(
        strategy="held_out_cohort",
        fold=0,
        train_subjects=tuple(s for s in ordered if s not in held),
        test_subjects=tuple(held),
        seed=seed,
    )


def _numeric(subject_id: str) -> int:
    digits = "".join(c for c in subject_id if c.isdigit())
    return int(digits) if digits else 0
