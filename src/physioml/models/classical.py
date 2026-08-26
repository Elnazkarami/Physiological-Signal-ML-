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


MODELS: dict[str, Callable[[], Any]] = {
    "majority": majority,
    "logistic": logistic,
    "linear_svm": linear_svm,
    "random_forest": random_forest,
    "gradient_boosting": gradient_boosting,
}
