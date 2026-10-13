import numpy as np
import pytest

from qmlrc.data import make_split, subsample


def test_subsample_has_requested_size_and_prevalence(pool, rng):
    X, y = pool
    Xs, ys = subsample(X, y, 500, 0.1, rng)
    assert Xs.shape == (500, X.shape[1])
    assert ys.sum() == 50


def test_subsample_refuses_impossible_request(pool, rng):
    X, y = pool
    with pytest.raises(ValueError):
        subsample(X, y, 3000, 0.5, rng)  # pool has only ~400 positives


def test_split_is_deterministic_and_seed_dependent(pool):
    X, y = pool
    kw = dict(n_samples=200, positive_fraction=0.1, test_size=0.3, n_components=4, n_folds=3)
    a, b, c = make_split(X, y, seed=1, **kw), make_split(X, y, seed=1, **kw), make_split(X, y, seed=2, **kw)
    np.testing.assert_array_equal(a.X_train, b.X_train)
    np.testing.assert_array_equal(a.y_test, b.y_test)
    assert not np.array_equal(a.X_train, c.X_train)


def test_split_shapes_folds_and_train_only_preprocessing(small_split):
    s = small_split
    assert s.X_train.shape == (105, 3) and s.X_test.shape == (45, 3)
    assert s.y_test.sum() == 9  # stratified: 20% of 45
    # preprocessing is fitted on the training rows only
    np.testing.assert_allclose(s.X_train.mean(0), 0.0, atol=1e-10)
    np.testing.assert_allclose(s.X_train.std(0), 1.0, atol=1e-10)
    assert not np.allclose(s.X_test.mean(0), 0.0, atol=1e-6)
    seen = np.concatenate([val for _, val in s.cv_splits])
    assert sorted(seen) == list(range(len(s.y_train)))  # folds partition the training set
