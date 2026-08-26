"""The models, smallest first.

The plan's hierarchy, and the reason for its order: a linear model on engineered
features establishes whether the features carry the signal at all. If logistic
regression reaches a useful score, a deeper architecture is answering a question
that was already answered. If it does not, a deeper architecture on the same
features usually finds the same nothing, more expensively.

Every model is wrapped in a pipeline that scales inside the fold. That is not
tidiness. Fitting a scaler on the whole dataset lets the test subjects'
distribution influence the transform applied to the training data, which is
leakage — quieter than sharing subjects between splits, and it inflates scores
the same way.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def logistic() -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0),
            ),
        ]
    )


def linear_svm() -> Any:
    """A linear support vector machine, calibrated so it can report probability.

    An uncalibrated SVM's decision function is not a probability, and reporting
    it as one would make every calibration number below meaningless.

    This inner calibration is stratified, not grouped: sklearn's
    ``CalibratedClassifierCV`` is not given the subject of each row, so the
    same participants sit on both sides of its ``cv=3`` split. It is here to
    turn a decision function into something bounded, not to make a trustworthy
    probability, and this model's ECE should be read with that in mind.
    :class:`~physioml.models.calibration.SubjectCalibrated` is the one that
    holds participants out.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                CalibratedClassifierCV(
                    LinearSVC(class_weight="balanced", random_state=0), cv=3
                ),
            ),
        ]
    )


def random_forest() -> Any:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=0,
                ),
            ),
        ]
    )


def gradient_boosting() -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", HistGradientBoostingClassifier(max_iter=200, random_state=0)),
        ]
    )


def majority() -> Any:
    """Always the commonest class.

    Included as a model rather than mentioned in a footnote, so every table has
    the number a result has to beat sitting in it. On a task that is 22%
    positive this scores 78% accuracy and 0.5 balanced accuracy, which is the
    clearest possible statement of why accuracy is not reported alone.
    """
    from sklearn.dummy import DummyClassifier

    return DummyClassifier(strategy="most_frequent")


#: The same models with their probabilities calibrated on held-out subjects.
#:
#: Kept separate rather than folded into ``MODELS`` so a comparison table can
#: put a model beside its calibrated self, which is the only way to see what
#: calibration did.
def _calibrated_models() -> dict[str, Callable[[], Any]]:
    from physioml.models.calibration import calibrated

    return {
        f"{name}+isotonic": calibrated(factory)
        for name, factory in MODELS.items()
        if name != "majority"  # a constant model has nothing to calibrate
    }


MODELS: dict[str, Callable[[], Any]] = {
    "majority": majority,
    "logistic": logistic,
    "linear_svm": linear_svm,
    "random_forest": random_forest,
    "gradient_boosting": gradient_boosting,
}


def calibrated_models() -> dict[str, Callable[[], Any]]:
    """Calibrated counterparts of every model that predicts a probability."""
    return _calibrated_models()
