"""What a training run was, what a model expects, and refusing the mismatch.

Two registries, and one rule that makes them worth having.

A :class:`TrainingRun` records the conditions a model was produced under —
which subjects were in which split, the strategy that put them there, the seed,
the hyperparameters, the code commit. A :class:`ModelArtifact` records what the
resulting model expects to be fed.

The rule is that inference is **refused** when the features on offer do not
match what the model was trained on. Silently accepting them is the failure
this is designed against: a model scored on a differently-ordered or
differently-versioned feature vector returns confident, plausible, wrong
numbers, and nothing downstream looks unusual. A refusal is noisy and correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from physioml.core.feature import FeatureVector
from physioml.core.provenance import content_id, utc
from physioml.core.recording import Modality


class SchemaMismatch(ValueError):
    """Raised when features do not match what a model expects."""


@dataclass(frozen=True, slots=True)
class TrainingRun:
    """The conditions one model was trained under."""

    task: str
    dataset_version: str
    split_strategy: str
    """``leave_one_subject_out``, ``group_k_fold``, ``held_out_cohort``."""

    train_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]
    validation_subjects: tuple[str, ...] = ()
    feature_schema_version: str = ""
    preprocessing_version: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    random_seed: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    code_commit: str = ""
    environment: dict[str, str] = field(default_factory=dict)
    created_at: datetime | None = None
    training_run_id: str = ""

    @classmethod
    def create(cls, **kwargs: Any) -> TrainingRun:
        for key in ("train_subjects", "test_subjects", "validation_subjects"):
            kwargs[key] = tuple(kwargs.get(key) or ())
        kwargs["created_at"] = utc(kwargs.get("created_at"))
        run = cls(**kwargs)
        run._require_disjoint()
        return replace(run, training_run_id=run._identity())

    def _require_disjoint(self) -> None:
        """No subject may appear in more than one split.

        Windows from one participant share that participant's physiology, so a
        subject spanning train and test lets the model recognise the person
        rather than the state. The resulting score is high and meaningless, and
        it is the single most common way a physiological classifier is wrong.
        """
        splits = {
            "train": set(self.train_subjects),
            "validation": set(self.validation_subjects),
            "test": set(self.test_subjects),
        }
        for left, right in (
            ("train", "test"),
            ("train", "validation"),
            ("validation", "test"),
        ):
            shared = splits[left] & splits[right]
            if shared:
                raise ValueError(
                    f"subjects {sorted(shared)} appear in both {left} and {right}. "
                    "Subject-level leakage makes the reported score a measure of "
                    "subject recognition, not of the task."
                )
        if not splits["train"] or not splits["test"]:
            raise ValueError("a training run needs both train and test subjects")

    def _identity(self) -> str:
        return content_id(
            "trun",
            {
                "task": self.task,
                "dataset_version": self.dataset_version,
                "split_strategy": self.split_strategy,
                "train_subjects": sorted(self.train_subjects),
                "validation_subjects": sorted(self.validation_subjects),
                "test_subjects": sorted(self.test_subjects),
                "feature_schema_version": self.feature_schema_version,
                "preprocessing_version": self.preprocessing_version,
                "hyperparameters": self.hyperparameters,
                "random_seed": self.random_seed,
                "code_commit": self.code_commit,
            },
        )

    @property
    def all_subjects(self) -> frozenset[str]:
        return frozenset(
            {*self.train_subjects, *self.validation_subjects, *self.test_subjects}
        )


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """A trained model, and the shape of what it accepts."""

    model_name: str
    model_version: str
    task: str
    training_run_id: str
    expected_features: tuple[str, ...]
    """Qualified feature names, in the order the model was trained on."""

    feature_schema_version: str
    expected_modalities: tuple[Modality, ...] = ()
    artifact_hash: str = ""
    calibration: str = ""
    """How probabilities were calibrated, if they were. Empty means raw model
    output, which should not be reported as a probability."""

    created_at: datetime | None = None
    model_id: str = ""

    @classmethod
    def create(cls, **kwargs: Any) -> ModelArtifact:
        kwargs["expected_features"] = tuple(kwargs.get("expected_features") or ())
        kwargs["expected_modalities"] = tuple(
            Modality(m) if isinstance(m, str) else m
            for m in (kwargs.get("expected_modalities") or ())
        )
        kwargs["created_at"] = utc(kwargs.get("created_at"))
        model = cls(**kwargs)
        if not model.expected_features:
            raise ValueError("a model must declare the features it expects")
        return replace(model, model_id=model._identity())

    def _identity(self) -> str:
        return content_id(
            "model",
            {
                "model_name": self.model_name,
                "model_version": self.model_version,
                "task": self.task,
                "training_run_id": self.training_run_id,
                "expected_features": list(self.expected_features),
                "feature_schema_version": self.feature_schema_version,
                "artifact_hash": self.artifact_hash,
            },
        )

    def accepts(self, vector: FeatureVector) -> None:
        """Raise unless this vector is what the model was trained on.

        Checks order as well as membership. A vector with the right names in the
        wrong order scores without complaint and is wrong in a way no metric
        reveals.
        """
        if vector.feature_set_version != self.feature_schema_version:
            raise SchemaMismatch(
                f"{self.model_name}@{self.model_version} expects feature schema "
                f"{self.feature_schema_version!r}, vector is "
                f"{vector.feature_set_version!r}"
            )
        if vector.names != self.expected_features:
            missing = [n for n in self.expected_features if n not in vector.names]
            extra = [n for n in vector.names if n not in self.expected_features]
            if missing or extra:
                raise SchemaMismatch(
                    f"{self.model_name}@{self.model_version} feature mismatch — "
                    f"missing {missing or 'none'}, unexpected {extra or 'none'}"
                )
            raise SchemaMismatch(
                f"{self.model_name}@{self.model_version} received the expected "
                "features in a different order; scoring would silently misalign "
                "every column"
            )

    def __str__(self) -> str:
        return f"{self.model_name}@{self.model_version} ({self.task})"
