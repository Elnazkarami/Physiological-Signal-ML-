"""How often a pipeline can answer at all, beside how well it answers.

Excluding a participant whose signal quality control rejected is honest and
insufficient. The performance table that results answers one question -- *when
the inputs are available, how good is the prediction* -- and quietly drops the
other, which for anything deployed is at least as important: *how often are they
available?*

The two come apart on this data. Adding a chest strap to a wrist band raises the
cohort score slightly. It also loses one participant's entire positive class to
an amplifier that clipped during the stress condition, so for that person the
chest-dependent pipeline produces no usable prediction at all. A comparison that
reports the first number and not the second recommends the strap.

Nothing here judges. It counts what survived, per participant and per condition,
and leaves the trade visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from physioml.dataset import FeatureTable


@dataclass(frozen=True)
class Coverage:
    """What a feature table retained, and from whom."""

    name: str
    rows: int
    subjects: tuple[str, ...]
    by_subject: dict[str, int] = field(default_factory=dict)
    by_condition: dict[str, int] = field(default_factory=dict)
    by_subject_condition: dict[tuple[str, str], int] = field(default_factory=dict)

    def scorable(self, positive: str = "stress") -> tuple[str, ...]:
        """Subjects with both classes present, which is what a fold needs.

        A participant holding only one class cannot yield a balanced accuracy,
        so they are not merely scored badly -- they are absent from the result,
        and the mean is over a smaller cohort than the reader assumes.
        """
        return tuple(
            s
            for s in self.subjects
            if 0 < self.by_subject_condition.get((s, positive), 0) < self.by_subject[s]
        )

    def missing(self, positive: str = "stress") -> tuple[str, ...]:
        return tuple(s for s in self.subjects if s not in self.scorable(positive))


def coverage_of(table: FeatureTable, name: str) -> Coverage:
    """Count what one table holds, without scoring anything."""
    subjects = tuple(table.subject_ids)
    by_subject = {s: int(np.sum(table.subjects == s)) for s in subjects}
    labels = sorted(set(table.labels.tolist()))
    by_condition = {c: int(np.sum(table.labels == c)) for c in labels}
    by_subject_condition = {
        (s, c): int(np.sum((table.subjects == s) & (table.labels == c)))
        for s in subjects
        for c in labels
    }
    return Coverage(
        name=name,
        rows=len(table),
        subjects=subjects,
        by_subject=by_subject,
        by_condition=by_condition,
        by_subject_condition=by_subject_condition,
    )


def compare(coverages: list[Coverage], *, positive: str = "stress") -> str:
    """A table of what each configuration can answer for, and for whom."""
    width = max(len(c.name) for c in coverages) + 2
    lines = [
        f"{'configuration':{width}} {'rows':>7} {'subjects':>9} {'scorable':>9} "
        f"{positive:>8} {'missing':>20}",
        "-" * (width + 56),
    ]
    for found in coverages:
        scorable = found.scorable(positive)
        missing = found.missing(positive)
        lines.append(
            f"{found.name:{width}} {found.rows:7d} {len(found.subjects):9d} "
            f"{len(scorable):9d} {found.by_condition.get(positive, 0):8d} "
            f"{', '.join(missing) or '—':>20}"
        )
    return "\n".join(lines)


def common_subjects(coverages: list[Coverage], *, positive: str = "stress") -> list[str]:
    """Participants every configuration can be scored on.

    The only cohort on which two configurations can be compared without the
    comparison also measuring who each of them dropped.
    """
    if not coverages:
        return []
    shared = set(coverages[0].scorable(positive))
    for found in coverages[1:]:
        shared &= set(found.scorable(positive))
    return sorted(shared, key=lambda s: int("".join(filter(str.isdigit, s)) or 0))


def by_condition_table(found: Coverage, reference: Coverage) -> str:
    """Retention per participant and condition, against a reference table."""
    conditions = sorted(reference.by_condition)
    lines = [
        f"{'subj':5} " + "  ".join(f"{c[:9]:>9}" for c in conditions),
        "-" * (5 + 11 * len(conditions)),
    ]
    for subject in reference.subjects:
        cells = []
        for condition in conditions:
            kept = found.by_subject_condition.get((subject, condition), 0)
            whole = reference.by_subject_condition.get((subject, condition), 0)
            cells.append(f"{kept / whole:9.2f}" if whole else f"{'—':>9}")
        lines.append(f"{subject:5} " + "  ".join(cells))
    return "\n".join(lines)
