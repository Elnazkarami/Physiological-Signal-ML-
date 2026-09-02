"""Turning scored nights into the table a sleep classifier is trained on.

The same shape as the peripheral builder and the same rules -- one row per
epoch, the subject kept on the row, features dropped rather than imputed when
quality control rejects the signal they come from -- with three differences
that belong to this dataset.

Epochs are not chosen. A hypnogram is scored in 30-second epochs and the
windows are those epochs, so there is no stride to pick and no overlap to worry
about: consecutive rows share no signal at all. That removes the trap the
peripheral tables carry, where a random split puts almost the same minute on
both sides of it.

Both nights of a subject are one subject. Sleep-EDF records most participants
twice, and treating the two nights as two people would put the same person on
both sides of every split -- the leak the whole evaluation is built to prevent,
arriving through the file naming.

The trimming is reported. A Sleep Cassette recording runs about twenty hours
around a night's sleep, and how much was cut away decides what the class
balance is; a table that does not say cannot be compared with one built
differently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from physioml.core.feature import Feature
from physioml.dataset import FeatureTable
from physioml.io.edf import EDFError
from physioml.io.sleep_edf import EPOCH_SECONDS, SleepEDF, SleepEDFError
from physioml.neural.features import (
    CHANNEL_PREFIX,
    FEATURE_SET,
    FEATURE_SET_VERSION,
    eeg_features,
    emg_features,
    eog_features,
)
from physioml.neural.qc import CHECKS, DEFAULT_EEG_POLICY, EEGPolicy


def _extract(label: str, samples: np.ndarray, rate: float) -> dict[str, float]:
    if label in CHANNEL_PREFIX:
        return eeg_features(samples, rate, CHANNEL_PREFIX[label])
    if label.startswith("EOG"):
        return eog_features(samples, rate)
    if label.startswith("EMG"):
        return emg_features(samples, rate)
    return {}


def build_sleep(
    directory: Path | str,
    *,
    subjects: list[str] | None = None,
    nights: tuple[int, ...] = (1, 2),
    margin_minutes: float = 30.0,
    policy: EEGPolicy = DEFAULT_EEG_POLICY,
    min_coverage: float = 0.9,
    progress: bool = False,
) -> FeatureTable:
    """Read every scored night, epoch it, run quality control, extract features."""
    source = SleepEDF(directory)
    available = source.nights()
    wanted = set(subjects) if subjects is not None else None

    rows: list[dict[str, Feature]] = []
    row_subjects: list[str] = []
    row_labels: list[str] = []
    row_windows: list[str] = []
    row_starts: list[float] = []
    codes: dict[str, int] = {}
    trimmed = 0
    unscored = 0

    unreadable: list[str] = []
    for subject_id, night in available:
        if night not in nights or (wanted is not None and subject_id not in wanted):
            continue
        try:
            record = source.read(subject_id, night, margin_minutes=margin_minutes)
        except (EDFError, SleepEDFError) as exc:
            # One unreadable night should not cost the other seventy-seven. A
            # truncated download, a file with no scoring, a recording missing
            # the channels asked for: all are reasons to skip that night and
            # say so, not to lose an hour of work at the end of it.
            unreadable.append(f"{subject_id}n{night}: {exc}")
            if progress:
                print(f"  {subject_id} night {night}: skipped — {exc}", flush=True)
            continue
        trimmed += record.trimmed_epochs
        unscored += record.unscored_epochs

        for index, stage in enumerate(record.stages):
            if not stage:
                continue  # nobody scored this epoch
            found: dict[str, Feature] = {}
            for label, samples in record.signals.items():
                rate = record.rates[label]
                begin = round(index * EPOCH_SECONDS * rate)
                end = round((index + 1) * EPOCH_SECONDS * rate)
                slice_ = samples[begin:end]
                if slice_.size == 0:
                    continue

                check = CHECKS.get(label)
                reasons = check(slice_, rate, policy) if check else []
                for code in reasons:
                    key = f"{label}:{code}"
                    codes[key] = codes.get(key, 0) + 1
                if [c for c in reasons if c not in policy.warn_only]:
                    continue

                window = f"{subject_id}n{night}e{index}"
                for name, value in _extract(label, slice_, rate).items():
                    if not np.isfinite(value):
                        continue
                    found[name] = Feature.create(
                        subject_id=subject_id,
                        name=name,
                        value=float(value),
                        unit=None,
                        feature_set=FEATURE_SET,
                        feature_set_version=FEATURE_SET_VERSION,
                        source_window_ids=(window,),
                        transform_id=f"{FEATURE_SET}@{FEATURE_SET_VERSION}",
                    )
            if not found:
                continue
            rows.append(found)
            row_subjects.append(subject_id)
            row_labels.append(str(stage))
            row_windows.append(f"{subject_id}n{night}e{index}")
            # Offset by night so the two nights of one subject do not appear to
            # be the same hours twice, which any split made in time would then
            # interleave.
            row_starts.append(
                record.offset_seconds + index * EPOCH_SECONDS + (night - 1) * 86400.0
            )
        if progress:
            print(f"  {subject_id} night {night}: {len(rows)} rows so far", flush=True)
        del record

    if not rows:
        raise ValueError("no scored epoch produced any features")

    coverage: dict[str, int] = {}
    for row in rows:
        for name in row:
            coverage[name] = coverage.get(name, 0) + 1
    threshold = min_coverage * len(rows)
    names = sorted(n for n, seen in coverage.items() if seen >= threshold)
    if not names:
        raise ValueError(
            f"no feature appears in {min_coverage:.0%} of epochs; the cohort has "
            "nothing in common to train on"
        )
    complete = [i for i, r in enumerate(rows) if all(n in r for n in names)]

    codes["trimmed_epochs"] = trimmed
    codes["unscored_epochs"] = unscored
    if unreadable:
        codes["unreadable_nights"] = len(unreadable)
    return FeatureTable(
        feature_names=tuple(names),
        values=np.array([[rows[i][n].value for n in names] for i in complete], dtype=float),
        subjects=np.array([row_subjects[i] for i in complete]),
        labels=np.array([row_labels[i] for i in complete]),
        window_ids=tuple(row_windows[i] for i in complete),
        window_starts=np.array([row_starts[i] for i in complete], dtype=float),
        feature_set_version=f"{FEATURE_SET}-{FEATURE_SET_VERSION}",
        qc_policy_version=policy.version,
        dropped_incomplete=len(rows) - len(complete),
        qc_codes=codes,
    )
