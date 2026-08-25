"""A model's output, and everything needed to account for it.

The plan's central question is whether a prediction can be traced back to the
sensor windows that produced it. That is only answerable if the prediction
carries both halves of its history at the moment it is made:

* **where the data came from** — the feature ids, and through them the windows,
  the preprocessing run, the recording and the CDFS facts;
* **what produced it** — the model name and version, and the training run, and
  through that the split, the seed, the hyperparameters and the code commit.

Both are required here rather than encouraged. A prediction missing either is
refused at construction, because a prediction that cannot be accounted for is
worse than no prediction: it will be believed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from physioml.core.provenance import content_id, utc


@dataclass(frozen=True, slots=True)
class Prediction:
    """One model output for one window. Construct via :meth:`create`."""

    study_id: str
    subject_id: str
    task: str
    window_start: datetime
    window_end: datetime

    model_name: str
    model_version: str
    training_run_id: str
    feature_set_version: str
    feature_ids: tuple[str, ...]
    source_window_ids: tuple[str, ...]

    predicted_class: str | None = None
    predicted_value: float | None = None
    probability: float | None = None
    """Confidence in the predicted class, after calibration if any was applied."""

    uncertainty: float | None = None
    created_at: datetime | None = None
    prediction_id: str = ""

    @classmethod
    def create(cls, **kwargs: Any) -> Prediction:
        kwargs["window_start"] = utc(kwargs.get("window_start"))
        kwargs["window_end"] = utc(kwargs.get("window_end"))
        kwargs["created_at"] = utc(kwargs.get("created_at"))
        for key in ("feature_ids", "source_window_ids"):
            kwargs[key] = tuple(kwargs.get(key) or ())
        prediction = cls(**kwargs)
        prediction._require_accountable()
        return replace(prediction, prediction_id=prediction._identity())

    def _require_accountable(self) -> None:
        if self.predicted_class is None and self.predicted_value is None:
            raise ValueError("a prediction must predict something")
        if self.window_end <= self.window_start:
            raise ValueError("window_end must follow window_start")
        missing = [
            name
            for name in (
                "model_name",
                "model_version",
                "training_run_id",
                "feature_set_version",
            )
            if not getattr(self, name)
        ]
        if missing:
            raise ValueError(
                f"prediction is not accountable: missing {', '.join(missing)}. "
                "A prediction that cannot name what produced it will still be "
                "believed, which is why this is refused rather than warned about."
            )
        if not self.source_window_ids:
            raise ValueError(
                "prediction has no source windows; it cannot be traced to the "
                "signal it was made from"
            )
        if self.probability is not None and not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability {self.probability} is not in [0, 1]")

    def _identity(self) -> str:
        return content_id(
            "pred",
            {
                "study_id": self.study_id,
                "subject_id": self.subject_id,
                "task": self.task,
                "window_start": self.window_start,
                "window_end": self.window_end,
                "predicted_class": self.predicted_class,
                "predicted_value": self.predicted_value,
                "model_name": self.model_name,
                "model_version": self.model_version,
                "training_run_id": self.training_run_id,
                "feature_set_version": self.feature_set_version,
                "feature_ids": sorted(self.feature_ids),
            },
        )

    @property
    def outcome(self) -> str | float:
        """What was predicted — a class label, or a number.

        :meth:`create` guarantees one of the two is present; the final branch
        states that guarantee in code rather than leaving it to a silenced
        type error.
        """
        if self.predicted_class is not None:
            return self.predicted_class
        if self.predicted_value is not None:
            return self.predicted_value
        raise ValueError(f"prediction {self.prediction_id} predicts nothing")

    def __str__(self) -> str:
        confidence = f" p={self.probability:.2f}" if self.probability is not None else ""
        return (
            f"{self.subject_id} {self.task}={self.outcome}{confidence} "
            f"[{self.model_name}@{self.model_version}]"
        )
