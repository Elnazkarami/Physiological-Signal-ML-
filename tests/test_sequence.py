"""A model that reads a night rather than a pile of epochs.

The tests that matter here are about sequence integrity: that a recording is
kept in time order, that two participants are never joined into one sequence,
and that a model given the order can use information a single-epoch model
cannot.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch is not installed")

from physioml.models.sequence import SequenceClassifier, SequenceConfig, gru  # noqa: E402

SUBJECTS = ["S1", "S2", "S3", "S4"]


def sequences(per_subject: int = 60, seed: int = 0):
    """A task solvable only from context.

    The feature is pure noise. The label is a repeating block structure in
    time, so an epoch tells you nothing on its own and its position in the
    night tells you everything.
    """
    rng = np.random.default_rng(seed)
    X, y, groups, order = [], [], [], []
    for subject in SUBJECTS:
        for i in range(per_subject):
            X.append(rng.normal(0, 1, 3))
            y.append("A" if (i // 10) % 2 == 0 else "B")
            groups.append(subject)
            order.append(i * 30.0)
    return (
        np.array(X),
        np.array(y),
        np.array(groups),
        np.array(order, dtype=float),
    )


def test_a_sequence_model_needs_the_participant_and_the_time():
    """Without them the rows are not a sequence, they are a pile."""
    X, y, groups, _ = sequences()
    with pytest.raises(ValueError, match="not a sequence"):
        gru(epochs=1).fit(X, y)
    with pytest.raises(ValueError, match="not a sequence"):
        gru(epochs=1).fit(X, y, groups=groups)


def test_it_learns_a_structure_only_the_order_reveals():
    """The feature is noise; the label is a block pattern in time. A model
    scoring epochs independently cannot beat chance here."""
    X, y, groups, order = sequences()
    # A brisker learning rate than the default: this toy has three noise
    # features and four short nights, and the default is tuned for a real one.
    model = gru(epochs=40, hidden=32, layers=1, patience=40, seed=1, learning_rate=3e-3)
    model.fit(X, y, groups=groups, order=order)
    predicted = model.predict(X, groups=groups, order=order)
    assert float(np.mean(predicted == y)) > 0.75


def test_two_participants_are_never_one_sequence():
    """Concatenating them would let one person's night end inside another's."""
    X, y, groups, order = sequences(per_subject=20)
    model = gru(epochs=1, hidden=8, layers=1)
    model.fit(X, y, groups=groups, order=order)
    pieces = model._nights(
        X, y, groups, order, {c: i for i, c in enumerate(model.classes_)}
    )
    assert len(pieces) == len(SUBJECTS)
    for _, features, labels in pieces:
        assert features.shape[0] == 20
        assert labels.shape[0] == 20


def test_a_gap_starts_a_new_sequence():
    """A participant with two nights has two sequences, not one long one with
    a discontinuity in the middle."""
    X, y, groups, order = sequences(per_subject=20)
    second = order.copy()
    second[groups == "S1"] += np.where(np.arange(20) >= 10, 86400.0, 0.0)
    model = gru(epochs=1, hidden=8, layers=1)
    model.fit(X, y, groups=groups, order=second)
    pieces = model._nights(
        X, y, groups, second, {c: i for i, c in enumerate(model.classes_)}
    )
    assert len(pieces) == len(SUBJECTS) + 1


def test_rows_are_scored_in_time_order_whatever_order_they_arrive_in():
    X, y, groups, order = sequences(per_subject=30, seed=3)
    model = gru(epochs=10, hidden=16, layers=1, patience=10, seed=2)
    model.fit(X, y, groups=groups, order=order)

    shuffle = np.random.default_rng(0).permutation(len(X))
    straight = model.predict(X, groups=groups, order=order)
    jumbled = model.predict(X[shuffle], groups=groups[shuffle], order=order[shuffle])
    assert np.array_equal(straight[shuffle], jumbled)


def test_probabilities_are_a_distribution():
    X, y, groups, order = sequences(per_subject=20)
    model = gru(epochs=2, hidden=8, layers=1).fit(X, y, groups=groups, order=order)
    probability = model.predict_proba(X, groups=groups, order=order)
    assert probability.shape == (len(X), 2)
    assert np.allclose(probability.sum(axis=1), 1.0)


def test_predicting_before_fitting_is_refused():
    X, _, groups, order = sequences(per_subject=5)
    with pytest.raises(ValueError, match="not been fitted"):
        SequenceClassifier().predict(X, groups=groups, order=order)


def test_the_configuration_is_recorded_in_full():
    """A model artifact has to be able to say what it was."""
    found = SequenceConfig().as_dict()
    for key in ("hidden", "layers", "bidirectional", "seed", "version"):
        assert key in found


def test_a_dropped_epoch_starts_a_new_sequence():
    """Splitting only on a large gap treats a missing epoch as uninterrupted
    sleep, which is exactly the discontinuity a recurrent model reads through."""
    X, y, groups, order = sequences(per_subject=20)
    with_gap = order.copy()
    # Remove the 30 s step between rows 9 and 10 of S1 by pushing the rest on.
    with_gap[(groups == "S1") & (order >= 300.0)] += 60.0
    model = gru(epochs=1, hidden=8, layers=1)
    model.fit(X, y, groups=groups, order=with_gap)
    pieces = model._nights(
        X, y, groups, with_gap, {c: i for i, c in enumerate(model.classes_)}
    )
    assert len(pieces) == len(SUBJECTS) + 1, "S1 should be split in two"
