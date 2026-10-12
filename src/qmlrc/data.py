"""Datasets, subsampling and preprocessing.

Every model in a comparison sees exactly the same rows, the same train/test
split, the same cross-validation folds and the same features. Only the model
changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml, make_classification
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

# ULB / Worldline credit-card fraud data, as mirrored on OpenML.
# https://www.openml.org/d/1597 -- 284,807 transactions, 492 frauds.
CREDITCARD_OPENML_ID = 1597
CREDITCARD_N_ROWS = 284_807
CREDITCARD_N_FRAUD = 492
CREDITCARD_N_FEATURES = 29  # V1..V28 (anonymised PCA components) + Amount


def default_data_home() -> Path:
    """Directory where downloaded datasets are cached (git-ignored)."""
    return Path(os.environ.get("QMLRC_DATA_HOME", Path.cwd() / "data"))


def load_creditcard(data_home: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Download (once) and return the full credit-card fraud dataset.

    Returns ``X`` with 29 columns (V1..V28, log1p(Amount)) and binary ``y``
    (1 = fraud). scikit-learn checks the MD5 checksum published by OpenML.
    """
    data_home = Path(data_home) if data_home is not None else default_data_home()
    bunch = fetch_openml(
        data_id=CREDITCARD_OPENML_ID,
        as_frame=False,
        parser="liac-arff",
        data_home=str(data_home),
    )
    X = np.asarray(bunch.data, dtype=float)
    y = (np.asarray(bunch.target).astype(str) == "1").astype(int)
    names = list(bunch.feature_names)
    if X.shape != (CREDITCARD_N_ROWS, CREDITCARD_N_FEATURES) or int(y.sum()) != CREDITCARD_N_FRAUD:
        raise RuntimeError(
            f"Unexpected credit-card data: shape={X.shape}, frauds={int(y.sum())}. "
            "The OpenML mirror may have changed; check the source before trusting any result."
        )
    # Amount is heavy-tailed and non-negative; the log transform is fixed, not fitted.
    X[:, names.index("Amount")] = np.log1p(X[:, names.index("Amount")])
    return X, y


def load_synthetic(
    n_samples: int = 20_000, n_features: int = 12, positive_fraction: float = 0.05, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Small imbalanced synthetic pool, used by tests and ``--quick`` runs (no download)."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(6, n_features),
        n_redundant=min(2, max(0, n_features - 6)),
        weights=[1.0 - positive_fraction],
        class_sep=1.0,
        flip_y=0.01,
        random_state=seed,
    )
    return X.astype(float), y.astype(int)


def load_pool(name: str, data_home: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    if name == "creditcard":
        return load_creditcard(data_home)
    if name == "synthetic":
        return load_synthetic()
    raise ValueError(f"Unknown dataset {name!r}")


def subsample(
    X: np.ndarray, y: np.ndarray, n_samples: int, positive_fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n_samples`` rows with a fixed fraction of positives, without replacement."""
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n_pos = int(round(n_samples * positive_fraction))
    n_neg = n_samples - n_pos
    if n_pos > len(pos) or n_neg > len(neg):
        raise ValueError(f"Cannot draw {n_pos} positives / {n_neg} negatives from the pool")
    idx = np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)])
    idx = rng.permutation(idx)
    return X[idx], y[idx]


def make_preprocessor(n_components: int | None) -> Pipeline:
    """Standardise, then (optionally) PCA to ``n_components`` and re-standardise.

    The preprocessing is unsupervised and is fitted on the training set only.
    """
    if n_components is None:
        return make_pipeline(StandardScaler())
    return make_pipeline(StandardScaler(), PCA(n_components=n_components, svd_solver="full"), StandardScaler())


@dataclass
class Split:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    cv_splits: list[tuple[np.ndarray, np.ndarray]]


def make_split(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    *,
    n_samples: int,
    positive_fraction: float,
    test_size: float,
    n_components: int | None,
    n_folds: int,
    seed: int,
) -> Split:
    """Subsample, split, preprocess and fix the CV folds for one seed."""
    rng = np.random.default_rng(seed)
    X, y = subsample(X_pool, y_pool, n_samples, positive_fraction, rng)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    prep = make_preprocessor(n_components)
    X_tr = prep.fit_transform(X_tr)
    X_te = prep.transform(X_te)
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_splits = list(folds.split(X_tr, y_tr))
    return Split(X_tr, X_te, y_tr, y_te, cv_splits)
