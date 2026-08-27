"""Which sensor is actually carrying the signal.

A single score on all features says a model works. It does not say what it is
working *from*, and on a wrist device that distinction decides what the result
means. A stress classifier that turns out to be reading the accelerometer has
learned that stressed people in a laboratory protocol move differently, which
is a real effect and not a physiological measurement — and it would fail the
moment the protocol changed. The ablation is what separates those two claims.

Two questions, and they are not the same one:

*Alone* — how far does one sensor get on its own? This is the direct measure of
how much a modality carries.

*Without* — how much is lost when one sensor is removed from the full set? This
is what the sensor contributes that nothing else already supplies. A modality
can score well alone and cost nothing when dropped, because another one is
carrying the same information; that pair of numbers is the finding, and either
number alone is misleading.

Every ablation reuses the same splits as the full model, so the comparison is
across feature sets and not across folds.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from physioml.dataset import FeatureTable
from physioml.evaluation.run import Evaluation, evaluate
from physioml.evaluation.splits import Split
from physioml.peripheral.chest import CHEST_FEATURES_BY_SIGNAL
from physioml.peripheral.features import FEATURES_BY_SIGNAL


def signal_groups() -> dict[str, tuple[str, ...]]:
    """Every signal of both devices, chest names prefixed to stay distinct."""
    groups = {f"wrist {k}": v for k, v in FEATURES_BY_SIGNAL.items()}
    groups.update({f"chest {k}": v for k, v in CHEST_FEATURES_BY_SIGNAL.items()})
    return groups


def device_groups() -> dict[str, tuple[str, ...]]:
    """The two devices, each as one group.

    The question a per-signal ablation cannot answer: not which sensor carries
    the most, but whether the second piece of hardware is worth wearing. A
    chest strap and a wrist band are not two features, they are two decisions.
    """
    wrist = tuple(n for names in FEATURES_BY_SIGNAL.values() for n in names)
    chest = tuple(n for names in CHEST_FEATURES_BY_SIGNAL.values() for n in names)
    return {"wrist": wrist, "chest": chest}


SplitSource = Callable[[], Iterable[Split]]
"""Called once per ablation, because a generator of splits is consumed by use."""


@dataclass(frozen=True)
class Ablation:
    """One model, scored on the full feature set and on subsets of it."""

    model_name: str
    full: Evaluation
    alone: dict[str, Evaluation]
    without: dict[str, Evaluation]
    signals: tuple[str, ...]

    def _score(self, run: Evaluation) -> float:
        return run.summary["balanced_accuracy_mean"]

    @property
    def baseline(self) -> float:
        return self._score(self.full)

    def contribution(self, signal: str) -> float:
        """Balanced accuracy lost when this signal is removed from the whole.

        Positive means the model is worse without it. Negative means the
        features were costing the model something, which happens and is worth
        seeing rather than clipping to zero.
        """
        return self.baseline - self._score(self.without[signal])

    def ranked(self) -> list[tuple[str, float]]:
        """Signals ordered by what they contribute, most first."""
        return sorted(
            ((s, self.contribution(s)) for s in self.signals),
            key=lambda pair: pair[1],
            reverse=True,
        )

    def table(self) -> str:
        width = max(22, max(len(s) for s in self.signals) + 9)
        header = (
            f"{'features':{width}} {'n':>3}   {'bal.acc':>13}   {'AUC':>5}   "
            f"{'worst':>5}   {'vs all':>7}"
        )
        lines = [header, "-" * len(header)]

        def row(name: str, run: Evaluation) -> str:
            summary = run.summary
            delta = self._score(run) - self.baseline
            return (
                f"{name:{width}} {len(run.feature_names):>3}   "
                f"{summary['balanced_accuracy_mean']:.3f} "
                f"±{summary['balanced_accuracy_sd']:.3f}   "
                f"{summary['roc_auc_mean']:.3f}   "
                f"{summary['balanced_accuracy_min']:.3f}   "
                f"{delta:+7.3f}"
            )

        lines.append(row("all signals", self.full))
        lines.append("")
        for signal in self.signals:
            lines.append(row(f"{signal} alone", self.alone[signal]))
        lines.append("")
        for signal in self.signals:
            lines.append(row(f"without {signal}", self.without[signal]))
        return "\n".join(lines)


def ablate(
    table: FeatureTable,
    model_factory: Callable[[], object],
    splits: SplitSource,
    *,
    model_name: str,
    groups: Mapping[str, Sequence[str]] | None = None,
    signals: Sequence[str] | None = None,
    positive: str = "stress",
) -> Ablation:
    """Score a model on the full feature set, on each signal, and without each.

    ``splits`` is a callable rather than a sequence so that every ablation gets
    the same folds freshly generated. Passing the folds themselves would work
    once and then silently score every later subset on an exhausted iterator.
    """
    using = signal_groups() if groups is None else groups
    present = {
        signal: [n for n in names if n in table.feature_names]
        for signal, names in using.items()
    }
    chosen = (
        tuple(signals) if signals is not None else tuple(s for s, n in present.items() if n)
    )
    for signal in chosen:
        if not present.get(signal):
            raise KeyError(f"no {signal} features in this table")

    def run(subset: FeatureTable, name: str) -> Evaluation:
        return evaluate(subset, model_factory, splits(), model_name=name, positive=positive)

    full = run(table, model_name)
    alone: dict[str, Evaluation] = {}
    without: dict[str, Evaluation] = {}
    for signal in chosen:
        alone[signal] = run(table.select(present[signal]), f"{model_name}/{signal}")
        rest = [n for n in table.feature_names if n not in present[signal]]
        if not rest:
            raise ValueError(f"removing {signal} would leave no features")
        without[signal] = run(table.select(rest), f"{model_name}/-{signal}")

    return Ablation(
        model_name=model_name, full=full, alone=alone, without=without, signals=chosen
    )
