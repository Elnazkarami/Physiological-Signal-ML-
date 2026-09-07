"""A model that reads a night rather than 96,000 unrelated epochs.

Every other model here scores one epoch in isolation. A human scorer does not:
they read the epochs before and after, and stage N1 is defined almost entirely
by being a transition between the two things around it. That is a structural
gap rather than a capacity one, which is why it is worth a different *kind* of
model instead of a bigger version of the same one.

The order matters and is deliberate. Simple models established that the
engineered features carry signal at all; a context-window version established
how much of the gap closes by handing a classical model its neighbours as extra
columns. Only after both is a recurrent model justified, and it should be
measured against them rather than against the single-epoch baseline it will
obviously beat.

**A recording is one sequence, and participants are held out whole.** The unit
of batching is a night, not an epoch, so nothing from a held-out participant
reaches the model that scores them -- the leak this project has guarded against
throughout, arriving through a different door.

**It is bidirectional, and that is a scope decision.** A scorer reads forwards
and backwards through a night that has already happened, so a model that scores
retrospectively may do the same. A model scoring in real time may not, and
nothing here should be read as a claim about that setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SequenceConfig:
    """Everything that decides what this model is, in one hashable place."""

    hidden: int = 64
    layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_nights: int = 8
    patience: int = 5
    """Stop when the held-out fraction of the *training* participants has not
    improved for this many passes. Never the test participant: that would be
    selecting on the thing being measured."""

    validation_fraction: float = 0.15
    seed: int = 0
    version: str = "gru-1.0"

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class SequenceClassifier:
    """A recurrent stage classifier, fitted on whole nights.

    Presents the same ``fit``/``predict``/``predict_proba`` surface the
    classical models do, so the same evaluation, ablation and comparison
    machinery works on it without special cases.
    """

    def __init__(self, config: SequenceConfig | None = None) -> None:
        self.config = config or SequenceConfig()
        self.classes_: np.ndarray = np.array([])
        self._model: Any = None
        self._mean: np.ndarray = np.array([])
        self._scale: np.ndarray = np.array([])

    # ── the network ──────────────────────────────────────────────────────────

    def _build(self, n_features: int, n_classes: int) -> Any:
        import torch
        from torch import nn

        cfg = self.config
        torch.manual_seed(cfg.seed)

        class Recurrent(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rnn = nn.GRU(
                    n_features,
                    cfg.hidden,
                    num_layers=cfg.layers,
                    batch_first=True,
                    dropout=cfg.dropout if cfg.layers > 1 else 0.0,
                    bidirectional=cfg.bidirectional,
                )
                width = cfg.hidden * (2 if cfg.bidirectional else 1)
                self.drop = nn.Dropout(cfg.dropout)
                self.out = nn.Linear(width, n_classes)

            def forward(self, x):
                hidden, _ = self.rnn(x)
                return self.out(self.drop(hidden))

        return Recurrent()

    # ── fitting ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None = None,
        order: np.ndarray | None = None,
    ):
        """Fit on whole recordings.

        ``groups`` is the participant of each row and ``order`` their time, both
        required: a sequence model without them would be reading rows in
        whatever order the table happens to hold, which for a night of sleep is
        not a sequence at all.
        """
        import torch
        from torch import nn

        if groups is None or order is None:
            raise ValueError(
                "a sequence model needs the participant and time of each row; "
                "without them the rows are not a sequence, they are a pile"
            )

        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        self.classes_ = np.unique(y)
        lookup = {c: i for i, c in enumerate(self.classes_)}

        # Scaling is fitted here, inside the fold, like every other model.
        self._mean = X.mean(axis=0)
        self._scale = X.std(axis=0)
        self._scale[self._scale == 0] = 1.0

        nights = self._nights(X, y, groups, order, lookup)
        subjects = sorted({s for s, _, _ in nights})
        rng.shuffle(subjects)
        held = max(1, int(len(subjects) * cfg.validation_fraction))
        validation = set(subjects[:held])

        train = [(a, b) for s, a, b in nights if s not in validation]
        check = [(a, b) for s, a, b in nights if s in validation]
        if not train:
            train, check = [(a, b) for _, a, b in nights], []

        self._model = self._build(X.shape[1], len(self.classes_))
        optimiser = torch.optim.AdamW(
            self._model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        # Stages are unevenly distributed and N1 is the rarest; without this the
        # model buys accuracy by never predicting it.
        counts = np.array([np.sum(y == c) for c in self.classes_], dtype=float)
        weight = torch.tensor((counts.sum() / (len(counts) * counts)), dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=weight)

        best, waited, best_state = np.inf, 0, None
        for _ in range(cfg.epochs):
            self._model.train()
            rng.shuffle(train)
            for start in range(0, len(train), cfg.batch_nights):
                for features, labels in train[start : start + cfg.batch_nights]:
                    optimiser.zero_grad()
                    logits = self._model(features.unsqueeze(0))[0]
                    loss = loss_fn(logits, labels)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self._model.parameters(), 5.0)
                    optimiser.step()

            if not check:
                continue
            self._model.eval()
            with torch.no_grad():
                total = sum(
                    float(loss_fn(self._model(f.unsqueeze(0))[0], lab)) for f, lab in check
                ) / len(check)
            if total < best - 1e-4:
                best, waited = total, 0
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
            else:
                waited += 1
                if waited >= cfg.patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        return self

    def _nights(self, X, y, groups, order, lookup):
        """One tensor per participant-night, in time order."""
        import torch

        made = []
        scaled = (X - self._mean) / self._scale
        for subject in np.unique(groups):
            rows = np.flatnonzero(groups == subject)
            when = order[rows]
            # A participant may have more than one recording; a gap far larger
            # than an epoch is a different night and a different sequence.
            ordered = rows[np.argsort(when, kind="stable")]
            times = order[ordered]
            splits = np.flatnonzero(np.diff(times) > 3600.0) + 1
            for piece in np.split(ordered, splits):
                if piece.size < 2:
                    continue
                made.append(
                    (
                        str(subject),
                        torch.tensor(scaled[piece], dtype=torch.float32),
                        torch.tensor([lookup[v] for v in y[piece]], dtype=torch.long),
                    )
                )
        return made

    # ── predicting ───────────────────────────────────────────────────────────

    def predict_proba(
        self,
        X: np.ndarray,
        groups: np.ndarray | None = None,
        order: np.ndarray | None = None,
    ) -> np.ndarray:
        import torch

        if self._model is None:
            raise ValueError("this model has not been fitted")
        if groups is None:
            groups = np.zeros(len(X), dtype=int)
        if order is None:
            order = np.arange(len(X), dtype=float)

        out = np.zeros((len(X), len(self.classes_)))
        self._model.eval()
        scaled = (X - self._mean) / self._scale
        with torch.no_grad():
            for subject in np.unique(groups):
                rows = np.flatnonzero(groups == subject)
                ordered = rows[np.argsort(order[rows], kind="stable")]
                times = order[ordered]
                splits = np.flatnonzero(np.diff(times) > 3600.0) + 1
                for piece in np.split(ordered, splits):
                    if piece.size == 0:
                        continue
                    features = torch.tensor(scaled[piece], dtype=torch.float32)
                    logits = self._model(features.unsqueeze(0))[0]
                    out[piece] = torch.softmax(logits, dim=1).numpy()
        return out

    def predict(
        self,
        X: np.ndarray,
        groups: np.ndarray | None = None,
        order: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.classes_[
            np.argmax(self.predict_proba(X, groups=groups, order=order), axis=1)
        ]


def gru(**overrides: Any) -> SequenceClassifier:
    """A recurrent classifier with the default configuration."""
    return SequenceClassifier(SequenceConfig(**overrides))


SEQUENCE_MODELS: dict[str, Any] = {"gru": gru}
