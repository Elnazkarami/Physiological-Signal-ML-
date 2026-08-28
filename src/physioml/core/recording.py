"""A single continuous acquisition from one device.

A recording is a *pointer* to signal data, not the data itself. The samples of
an hour of 256 Hz EEG do not belong in a provenance record, and putting them
there would make the lineage graph unusable for the thing it exists to do. What
is kept is enough to find the data again (``source_uri``), enough to know it has
not changed since (``source_hash``), and enough to interpret it (rate, channels,
units, device).

``source_fact_ids`` is the join back to CDFS: the canonical observations that
say this subject, in this study, produced this recording. It is what lets a
prediction eventually trace to a clinical record rather than only to a file.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any

from physioml.core.provenance import content_id, utc


class Modality(str, Enum):
    """What kind of signal a recording carries."""

    EEG = "eeg"
    BVP = "bvp"
    """Blood-volume pulse, from photoplethysmography."""

    EDA = "eda"
    """Electrodermal activity."""

    ECG = "ecg"
    TEMP = "temp"
    ACC = "acc"
    """Accelerometry."""

    RESP = "resp"

    EOG = "eog"
    """Electro-oculography: eye movement, which is what distinguishes REM
    sleep from the other stages and is scored alongside the EEG for exactly
    that reason."""

    EMG = "emg"
    """Electromyography: muscle activity. Submental in sleep scoring, where
    atonia is the other half of the REM definition; trapezius in stress
    protocols, where it reads tension."""
    SPO2 = "spo2"

    @property
    def is_neural(self) -> bool:
        return self is Modality.EEG


@dataclass(frozen=True, slots=True)
class Recording:
    """One acquisition. Construct via :meth:`create`."""

    study_id: str
    subject_id: str
    modality: Modality
    sampling_rate_hz: float
    start_time: datetime
    duration_seconds: float
    channels: tuple[str, ...]
    """Channel labels as the device reported them, before normalisation.

    The originals are kept because a normalisation that turns out to be wrong
    has to be re-doable, and because a label the mapper did not recognise is
    evidence about the device rather than noise to discard."""

    units: tuple[str, ...] = ()
    """Unit per channel, in channel order. Empty when the source does not say —
    which is itself worth recording, since a missing unit is a QC finding and
    not a default."""

    device_name: str = ""
    device_model: str = ""
    source_uri: str = ""
    source_hash: str = ""
    """Digest of the signal file. A recording whose file has changed underneath
    it is a different recording, and this is how that is noticed."""

    source_fact_ids: tuple[str, ...] = ()
    """CDFS facts this acquisition belongs to."""

    metadata: dict[str, Any] = field(default_factory=dict)
    recording_id: str = ""

    @classmethod
    def create(cls, **kwargs: Any) -> Recording:
        kwargs["start_time"] = utc(kwargs.get("start_time"))
        if isinstance(kwargs.get("modality"), str):
            kwargs["modality"] = Modality(kwargs["modality"])
        recording = cls(**kwargs)
        if recording.units and len(recording.units) != len(recording.channels):
            raise ValueError(
                f"{len(recording.units)} units for {len(recording.channels)} channels; "
                "a unit per channel or none at all"
            )
        if recording.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        return replace(recording, recording_id=recording._identity())

    def _identity(self) -> str:
        return content_id(
            "rec",
            {
                "study_id": self.study_id,
                "subject_id": self.subject_id,
                "modality": self.modality.value,
                "sampling_rate_hz": self.sampling_rate_hz,
                "start_time": self.start_time,
                "duration_seconds": self.duration_seconds,
                "channels": list(self.channels),
                "device_name": self.device_name,
                "device_model": self.device_model,
                "source_uri": self.source_uri,
                "source_hash": self.source_hash,
            },
        )

    @property
    def n_samples(self) -> int:
        return round(self.duration_seconds * self.sampling_rate_hz)

    def channel_index(self, label: str) -> int:
        try:
            return self.channels.index(label)
        except ValueError:
            raise KeyError(f"{self.recording_id} has no channel {label!r}") from None

    def __str__(self) -> str:
        return (
            f"{self.subject_id}/{self.modality.value} "
            f"{len(self.channels)}ch @{self.sampling_rate_hz:g}Hz "
            f"{self.duration_seconds:g}s"
        )
