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
