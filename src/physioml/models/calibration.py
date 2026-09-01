"""Making a stated probability mean what it says.

A model that separates two classes well can still be badly wrong about how
confident it is. On WESAD every model here reaches an expected calibration
error between 0.086 and 0.117 uncalibrated: a window it calls 90% stressed is
really about 80%. For a ranking that does not matter. For a number a person
reads, or a threshold someone sets from it, it is the whole thing.

**Calibration is fitted on held-out subjects, like everything else.** The usual
shortcut -- an inner stratified split -- puts the same participants in the fit
set and the calibration set, and a calibrator learns that participant's
particular confidence rather than the model's general overconfidence. It comes
out looking better calibrated than it is, which is the specific failure this
module exists to avoid. So the inner split groups by subject, using the same
splitter the outer evaluation uses.

**Calibration changes the probability, not the decision.** ``predict`` is the
base model's, untouched, so the reported labels keep the base model's decision
rule rather than thresholding the calibrated probability at 0.5. What is left
in a before-and-after table is the effect on the stated confidence alone;
moving the operating point at the same time would mix two changes into one
column.

The ranking is *nearly* preserved rather than exactly. Isotonic regression is
monotone non-decreasing, and its flat regions map distinct scores onto a single
value: those ties change the area under the curve slightly. On WESAD it moves
from 0.954 to 0.951. Anything that claimed AUC was untouched would be claiming
something the tables themselves contradict.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from physioml.evaluation.splits import group_k_fold


class SubjectCalibrated:
    """A classifier whose probabilities are calibrated on held-out subjects.

    Fitted in two passes over the training data. The first collects
    out-of-fold probabilities -- every row scored by a model that did not see
    that row's participant -- and fits the calibrator on those. The second
    refits the base model on all of the training data, because the calibrator
    is what needed the held-out scores, not the model.
    """

    def __init__(
        self,
        base_factory: Callable[[], Any],
        *,
        method: str = "isotonic",
        inner_folds: int = 5,
        random_state: int = 0,
    ) -> None:
        if method not in ("isotonic", "sigmoid"):
            raise ValueError(f"unknown calibration method {method!r}")
        self.base_factory = base_factory
        self.method = method
        self.inner_folds = inner_folds
        self.random_state = random_state

    # ── fitting ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None):
        if groups is None:
            # Falling back to a stratified split here would still produce a
            # model, and a better-looking calibration number than the method
            # deserves. Refusing is the point.
            raise ValueError(
                "SubjectCalibrated needs the subject of each row; without it the "
                "calibration set shares participants with the fit set and the "
                "result overstates how well calibrated the model is"
            )
        groups = np.asarray(groups)
        subjects = sorted(set(groups.tolist()))
        folds = min(self.inner_folds, len(subjects))
        if folds < 2:
            raise ValueError(
                f"calibration needs at least two training subjects, got {len(subjects)}"
            )

        held_out = np.full(len(y), np.nan)
        for split in group_k_fold(subjects, folds=folds, seed=self.random_state):
            inner_train, inner_test = split.mask(groups)
            if inner_train.size == 0 or inner_test.size == 0:
                continue
            if len(np.unique(y[inner_train])) < 2:
                continue
            model = self.base_factory()
            model.fit(X[inner_train], y[inner_train])
            held_out[inner_test] = model.predict_proba(X[inner_test])[:, 1]

        scored = ~np.isnan(held_out)
        if scored.sum() == 0 or len(np.unique(y[scored])) < 2:
            raise ValueError("no usable out-of-fold probabilities to calibrate on")

        self.calibrator_ = self._fit_calibrator(held_out[scored], y[scored])
        self.base_ = self.base_factory()
        self.base_.fit(X, y)
        self.classes_ = getattr(self.base_, "classes_", np.array([0, 1]))
        return self

    def _fit_calibrator(self, probability: np.ndarray, y: np.ndarray) -> Any:
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            # Clipping out of bounds rather than extrapolating: a probability
            # beyond the range seen in training has no evidence behind it.
            return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
                probability, y
            )

        from sklearn.linear_model import LogisticRegression

        # Platt scaling: a logistic fitted on the log-odds of the raw score.
        # On the probability itself the fit is forced through a shape logistic
        # regression cannot make, and the correction comes out too weak.
        return LogisticRegression().fit(_logit(probability).reshape(-1, 1), y)

    # ── predicting ───────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.base_.predict_proba(X)[:, 1]
        if self.method == "isotonic":
            adjusted = self.calibrator_.predict(raw)
        else:
            adjusted = self.calibrator_.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]
        adjusted = np.clip(adjusted, 0.0, 1.0)
        return np.column_stack([1.0 - adjusted, adjusted])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """The base model's decision, unchanged.

        Calibration restates confidence; it does not move the threshold. Doing
        both at once would leave a before-and-after table unable to say which
        change produced which difference.
        """
        return self.base_.predict(X)


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def calibrated(
    base_factory: Callable[[], Any], *, method: str = "isotonic"
) -> Callable[[], SubjectCalibrated]:
    """A factory producing a subject-calibrated version of a model."""

    def make() -> SubjectCalibrated:
        return SubjectCalibrated(base_factory, method=method)

    return make
