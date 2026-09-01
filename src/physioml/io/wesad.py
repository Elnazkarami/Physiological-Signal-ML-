"""Reading WESAD, without unpacking it.

WESAD is fifteen subjects of chest and wrist physiology, about 17 GB unpacked
from a 2 GB archive. This adapter reads a subject straight out of the zip and
never writes it to disk, because a pipeline that requires 17 GB of scratch space
before it can compute anything is one people run once.

The archive's per-subject ``.pkl`` is the synchronised, labelled form of the
same signals held raw in ``_respiban.txt`` and ``_E4_Data.zip``. It is the only
member read.

**On unpickling.** A pickle can execute arbitrary code while loading, so this
module refuses to run one: :class:`_ArraysOnly` permits exactly the handful of
constructors a numpy array needs to rebuild and blocks everything else. The
dataset is a published one from a known source, and that is still not a reason
to hand it the interpreter.
"""

from __future__ import annotations

import hashlib
import pickle
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from physioml.core.recording import Modality, Recording

#: Protocol conditions, as WESAD codes them. 0 is transition between blocks;
#: 5-7 are the recovery periods the study guide says to ignore.
CONDITIONS: dict[int, str] = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}

#: The label stream is sampled with the chest device.
LABEL_HZ = 700.0

#: Empatica E4, the wrist device. Rates are fixed by the hardware.
WRIST_HZ: dict[str, float] = {"ACC": 32.0, "BVP": 64.0, "EDA": 4.0, "TEMP": 4.0}
CHEST_HZ = 700.0

#: WESAD's signal names to the modalities this package models.
#: Every signal WESAD stores, and what it is. There is no default: a name that
#: is not here is a signal this reader does not understand, and guessing at one
#: is how the chest electromyogram became accelerometry -- silently, with a
#: recording that then claimed to be something it was not for the rest of its
#: life in the provenance chain.
MODALITIES: dict[str, Modality] = {
    "ACC": Modality.ACC,
    "BVP": Modality.BVP,
    "EDA": Modality.EDA,
    "TEMP": Modality.TEMP,
    "Temp": Modality.TEMP,
    "ECG": Modality.ECG,
    "EMG": Modality.EMG,
    "Resp": Modality.RESP,
}


def modality_of(name: str) -> Modality:
    """What a WESAD signal name is, or an error naming what is known."""
    try:
        return MODALITIES[name]
    except KeyError:
        raise WESADError(
            f"unknown signal {name!r}; this reader understands {sorted(MODALITIES)}"
        ) from None


def _digest(array: np.ndarray) -> str:
    """A checksum of the samples themselves.

    A recording's identity is built from its metadata -- subject, device, rate,
    duration -- and every one of those can stay the same while the samples
    change. Without this, a corrected export and the export it corrects are the
    same recording, and nothing downstream can tell that anything moved.
    """
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


#: WESAD carries no acquisition timestamps. Recordings are placed on a fixed
#: epoch so identifiers are reproducible; nothing downstream reads wall-clock
#: meaning into them, and a real deployment would take the time from the device.
EPOCH = datetime(2017, 1, 1, tzinfo=UTC)

_ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("builtins", "bytearray"),
    ("builtins", "bytes"),
    ("_codecs", "encode"),
}


class _ArraysOnly(pickle.Unpickler):
    """An unpickler that can rebuild numpy arrays and nothing else."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"refusing to construct {module}.{name} while unpickling WESAD; "
            "this reader rebuilds arrays and does not run code"
        )


class WESADError(RuntimeError):
    """The archive is not shaped the way this reader expects."""


@dataclass(frozen=True, slots=True)
class SubjectData:
    """One subject's signals, labels, and the recordings describing them."""

    subject_id: str
    signals: dict[str, np.ndarray]
    """Signal name to samples, wrist device unless ``chest`` was requested."""

    labels: np.ndarray
    """Protocol condition per sample at :data:`LABEL_HZ`, as WESAD codes it."""

    recordings: dict[str, Recording]
    """One per signal, keyed by the same names as ``signals``."""

    @property
    def duration_seconds(self) -> float:
        return len(self.labels) / LABEL_HZ


class WESAD:
    """A WESAD archive, addressed by subject."""

    def __init__(self, archive: Path | str, *, study_id: str = "WESAD") -> None:
        self.archive = Path(archive)
        self.study_id = study_id
        if not self.archive.is_file():
            raise WESADError(f"no archive at {self.archive}")

    def subjects(self) -> list[str]:
        """Subject identifiers present, in numeric order."""
        with zipfile.ZipFile(self.archive) as bundle:
            found = {
                name.split("/")[1]
                for name in bundle.namelist()
                if name.endswith(".pkl") and name.count("/") == 2
            }
        if not found:
            raise WESADError(f"{self.archive} holds no subject pickles")
        return sorted(found, key=lambda s: int(s.lstrip("S")))

    def read(self, subject_id: str, *, device: str = "wrist") -> SubjectData:
        """One subject, streamed from the archive.

        ``device`` is ``wrist`` (Empatica E4) or ``chest`` (RespiBAN). The wrist
        is the default because it is the one a person would plausibly wear
        outside a laboratory, which is the setting this package is aimed at.
        """
        if device not in ("wrist", "chest"):
            raise WESADError(f"device must be 'wrist' or 'chest', not {device!r}")

        member = f"WESAD/{subject_id}/{subject_id}.pkl"
        with zipfile.ZipFile(self.archive) as bundle:
            try:
                handle = bundle.open(member)
            except KeyError:
                raise WESADError(
                    f"{subject_id} is not in {self.archive.name}; "
                    f"present: {', '.join(self.subjects())}"
                ) from None
            with handle:
                raw = _ArraysOnly(handle, encoding="latin1").load()

        try:
            signals = {k: np.asarray(v) for k, v in raw["signal"][device].items()}
            labels = np.asarray(raw["label"]).ravel()
        except KeyError as exc:
            raise WESADError(f"{member} is missing {exc}") from exc

        rates = WRIST_HZ if device == "wrist" else dict.fromkeys(signals, CHEST_HZ)
        duration = len(labels) / LABEL_HZ
        recordings = {
            name: Recording.create(
                study_id=self.study_id,
                subject_id=subject_id,
                modality=modality_of(name),
                sampling_rate_hz=rates[name],
                start_time=EPOCH,
                duration_seconds=duration,
                channels=_channels(name, array),
                device_name="Empatica E4" if device == "wrist" else "RespiBAN",
                source_uri=f"{self.archive.name}!{member}",
                source_hash=_digest(array),
                metadata={"wesad_signal": name, "device": device},
            )
            for name, array in signals.items()
        }
        return SubjectData(subject_id, signals, labels, recordings)


def _channels(name: str, array: np.ndarray) -> tuple[str, ...]:
    """Channel labels for a WESAD signal.

    Accelerometry is three axes; everything else is one. The axis labels are
    invented here because WESAD does not name them, and inventing them
    explicitly is better than three columns called 0, 1 and 2.
    """
    if array.ndim == 2 and array.shape[1] == 3:
        return (f"{name}_x", f"{name}_y", f"{name}_z")
    return (name,)


def condition_of(labels: np.ndarray) -> str | None:
    """The protocol condition a stretch of label samples belongs to.

    ``None`` when the stretch spans more than one condition, or falls in a
    period WESAD says to ignore. Windows are not assigned a majority label:
    a window straddling the boundary between rest and stress is not 60% rest,
    it is unusable, and training on it teaches the model the transition.
    """
    unique = np.unique(labels)
    if unique.size != 1:
        return None
    return CONDITIONS.get(int(unique[0]))
