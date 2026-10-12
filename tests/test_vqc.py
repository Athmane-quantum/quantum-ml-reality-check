import numpy as np
from sklearn.metrics import average_precision_score

from qmlrc.vqc import VariationalQuantumClassifier


def test_vqc_learns_an_easy_problem(rng):
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] > 0.5).astype(int)
    model = VariationalQuantumClassifier(n_layers=2, scale=1.0, steps=120, batch_size=32, random_state=0).fit(X, y)
    scores = model.decision_function(X)
    assert scores.shape == (200,)
    assert average_precision_score(y, scores) > 0.9
    assert model.loss_curve_[-1] < model.loss_curve_[0]
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba.sum(1), 1.0)
    assert set(np.unique(model.predict(X))) <= {0, 1}


def test_vqc_is_deterministic_given_seed(rng):
    X = rng.normal(size=(60, 3))
    y = (X[:, 1] > 0).astype(int)
    a = VariationalQuantumClassifier(steps=5, random_state=3).fit(X, y).decision_function(X)
    b = VariationalQuantumClassifier(steps=5, random_state=3).fit(X, y).decision_function(X)
    np.testing.assert_allclose(a, b)


def test_separable_vqc_is_additive_across_features(rng):
    # with no entangling gates and a mean-of-<Z_i> readout, logit(x) = sum_i f_i(x_i) + b
    X = rng.normal(size=(40, 2))
    y = (X[:, 0] > 0).astype(int)
    model = VariationalQuantumClassifier(n_layers=2, scale=0.7, steps=5, entangling=False, readout="mean",
                                         random_state=1).fit(X, y)
    f = lambda u, v: model.decision_function(np.array([[u, v]]))[0]  # noqa: E731
    assert abs(f(0.3, -1.2) + f(1.1, 0.4) - f(0.3, 0.4) - f(1.1, -1.2)) < 1e-10
    # the entangled model is not additive in general
    ent = VariationalQuantumClassifier(n_layers=2, scale=0.7, steps=5, random_state=1).fit(X, y)
    g = lambda u, v: ent.decision_function(np.array([[u, v]]))[0]  # noqa: E731
    assert abs(g(0.3, -1.2) + g(1.1, 0.4) - g(0.3, 0.4) - g(1.1, -1.2)) > 1e-6


def test_separable_vqc_requires_mean_readout(rng):
    import pytest

    X = rng.normal(size=(20, 2))
    with pytest.raises(ValueError):
        VariationalQuantumClassifier(steps=1, entangling=False).fit(X, (X[:, 0] > 0).astype(int))


def test_standardised_readout_gives_usable_logits_at_tiny_bandwidth(rng):
    # without standardisation a scale of 0.01 makes the readout almost constant (session-2 failure)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] > 0.8).astype(int)
    kw = dict(n_layers=1, scale=0.01, steps=60, readout="mean", log_output_scale=True, random_state=0)
    plain = VariationalQuantumClassifier(**kw).fit(X, y).decision_function(X)
    std = VariationalQuantumClassifier(standardise_readout=True, **kw).fit(X, y).decision_function(X)
    assert plain.std() < 0.05 < 0.5 < std.std()
    assert average_precision_score(y, std) > 0.8


def test_default_vqc_is_unchanged_by_new_options(rng):
    X = rng.normal(size=(50, 3))
    y = (X[:, 0] > 0).astype(int)
    a = VariationalQuantumClassifier(steps=5, random_state=4).fit(X, y).decision_function(X)
    b = VariationalQuantumClassifier(steps=5, random_state=4, readout="z0", log_output_scale=False,
                                     standardise_readout=False).fit(X, y).decision_function(X)
    np.testing.assert_array_equal(a, b)
