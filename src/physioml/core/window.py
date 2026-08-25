"""A bounded slice of a recording, and what quality control made of it.

Windowing is where a continuous signal becomes countable things a model can be
trained on, so it is also where most silent damage happens. Two rules here
guard against it.

**A window is never discarded, only labelled.** Quality control assigns a
status and reason codes; it does not filter. A pipeline that drops bad windows
quietly reports a clean dataset and a model that was trained on an unstated
subset, and no downstream number reveals it. Keeping rejected windows means
"how much of this subject survived QC" is answerable, and it is usually the
first question worth asking about a disappointing result.

**A window remembers its preprocessing.** Filtering, referencing and resampling
change what the samples mean, so a window carries the identifier of the run that
produced it. Change the filter and you get new windows with new identifiers
rather than the same windows meaning something different.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from physioml.core.provenance import content_id, utc


class QCStatus(str, Enum):
    """What quality control concluded about a window."""

    VALID = "valid"
    WARNING = "warning"
    """Usable, with a caveat recorded. Included in training by default."""

    REJECTED = "rejected"
    """Not usable for the stated purpose. Retained, never deleted."""

    @property
    def usable(self) -> bool:
        return self is not QCStatus.REJECTED


@dataclass(frozen=True, slots=True)
class SignalWindow:
    """One window or epoch. Construct via :meth:`create`."""

    recording_id: str
    subject_id: str
    start_sample: int
    end_sample: int
    """Exclusive, so ``end_sample - start_sample`` is the length."""

    start_time: datetime
    sampling_rate_hz: float
    channel_ids: tuple[str, ...] = ()
    preprocessing_run_id: str = ""
    qc_status: QCStatus = QCStatus.VALID
    """A verdict about the window, not part of its identity — see
    :meth:`rejected`."""

    qc_reason_codes: tuple[str, ...] = ()
    """Why QC reached its conclusion. Codes rather than prose so they can be
    counted across a cohort."""

    source_fact_ids: tuple[str, ...] = ()
    label: str | None = None
    """Ground truth for this window, when the dataset provides it."""

    window_id: str = ""

    @classmethod
    def create(cls, **kwargs: Any) -> SignalWindow:
        kwargs["start_time"] = utc(kwargs.get("start_time"))
        if isinstance(kwargs.get("qc_status"), str):
            kwargs["qc_status"] = QCStatus(kwargs["qc_status"])
        window = cls(**kwargs)
        if window.end_sample <= window.start_sample:
            raise ValueError(
                f"window ends at {window.end_sample} and starts at "
                f"{window.start_sample}; a window must contain samples"
            )
        if window.qc_status is QCStatus.REJECTED and not window.qc_reason_codes:
            raise ValueError("a rejected window must record why it was rejected")
        return replace(window, window_id=window._identity())

    def _identity(self) -> str:
        return content_id(
            "win",
            {
                "recording_id": self.recording_id,
                "start_sample": self.start_sample,
                "end_sample": self.end_sample,
                "channel_ids": list(self.channel_ids),
                "preprocessing_run_id": self.preprocessing_run_id,
            },
        )

    @property
    def n_samples(self) -> int:
        return self.end_sample - self.start_sample

    @property
    def duration_seconds(self) -> float:
        return self.n_samples / self.sampling_rate_hz

    @property
    def end_time(self) -> datetime:
        return self.start_time + timedelta(seconds=self.duration_seconds)

    def rejected(self, *codes: str) -> SignalWindow:
        """The same window, marked rejected. Returns a new object.

        The identifier does **not** change, and that is deliberate. A window's
        identity is the physical slice — this recording, these samples, this
        preprocessing run — while QC is a verdict *about* that slice, which a
        better artifact detector may revise later.

        Keeping the identifier stable is what makes cascade invalidation
        possible: "window W was rejected, so every feature naming W is stale"
        is answerable. Had rejection minted a new identifier, the features
        computed while the window was still considered good would point at an
        identifier nothing has any more, and the link would have to be
        rediscovered by content.
        """
        return SignalWindow.create(
            recording_id=self.recording_id,
            subject_id=self.subject_id,
            start_sample=self.start_sample,
            end_sample=self.end_sample,
            start_time=self.start_time,
            sampling_rate_hz=self.sampling_rate_hz,
            channel_ids=self.channel_ids,
            preprocessing_run_id=self.preprocessing_run_id,
            qc_status=QCStatus.REJECTED,
            qc_reason_codes=tuple(codes),
            source_fact_ids=self.source_fact_ids,
            label=self.label,
        )

    def __str__(self) -> str:
        return (
            f"{self.subject_id} [{self.start_sample}:{self.end_sample}] "
            f"{self.duration_seconds:g}s {self.qc_status.value}"
        )
