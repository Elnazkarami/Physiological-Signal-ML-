"""The provenance spine: what was recorded, windowed, computed and predicted.

Deliberately free of scientific dependencies. Everything here is plain Python,
so the chain of evidence — recording to window to feature to prediction, and
the training run behind it — can be constructed and tested without installing
numpy, and CI asserts that it stays that way.

The computation lives elsewhere. :mod:`physioml.peripheral` and
:mod:`physioml.neural` produce features, :mod:`physioml.models` produce
predictions, and both express their results in these types.
"""

from physioml.core.feature import Feature, FeatureVector
from physioml.core.invalidation import Invalidated, invalidated_by
from physioml.core.prediction import Prediction
from physioml.core.provenance import content_id, utc
from physioml.core.recording import Modality, Recording
from physioml.core.registry import ModelArtifact, SchemaMismatch, TrainingRun
from physioml.core.window import QCStatus, SignalWindow

__all__ = [
    "Feature",
    "FeatureVector",
    "Invalidated",
    "Modality",
    "ModelArtifact",
    "Prediction",
    "QCStatus",
    "Recording",
    "SchemaMismatch",
    "SignalWindow",
    "TrainingRun",
    "content_id",
    "invalidated_by",
    "utc",
]
