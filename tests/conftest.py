import numpy as np
import pytest

from qmlrc.data import load_synthetic, make_split


@pytest.fixture(scope="session")
def pool():
    return load_synthetic(n_samples=4000, n_features=6, positive_fraction=0.1, seed=0)


@pytest.fixture(scope="session")
def small_split(pool):
    X, y = pool
    return make_split(X, y, n_samples=150, positive_fraction=0.2, test_size=0.3, n_components=3, n_folds=3, seed=0)


@pytest.fixture
def rng():
    return np.random.default_rng(123)
