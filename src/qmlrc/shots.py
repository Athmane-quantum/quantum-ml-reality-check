"""Finite-shot estimates of quantum kernels, as a quantum computer would produce them.

Every kernel elsewhere in this package is computed exactly from simulated state vectors. On hardware
each number is an average over R repeated measurements ("shots"):

* Fidelity kernel: k(x, x') is the probability of the all-zeros outcome of the overlap circuit
  U(x')^dagger U(x)|0> ("Loschmidt echo"), so its estimate is Binomial(R, k) / R. The diagonal is 1 by
  construction and is not measured.
* Projected kernel: each single-qubit Pauli expectation <P> is estimated from R shots in the P basis as
  2 * Binomial(R, (1 + <P>) / 2) / R - 1. (On hardware all qubits are measured in the same shot, so
  estimates for different qubits are correlated; drawing them independently keeps the right marginal
  noise and is the usual simplification.)

The estimated fidelity Gram matrix can have small negative eigenvalues; ``clip_to_psd`` sets them to
zero before the SVM sees it.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def sample_fidelity_kernel(K: np.ndarray, shots: int, rng: np.random.Generator, symmetric: bool = False) -> np.ndarray:
    """Shot-noise estimate of a fidelity kernel matrix (entries in [0, 1])."""
    est = rng.binomial(shots, np.clip(K, 0.0, 1.0)) / shots
    if symmetric:  # training Gram matrix: one measurement per pair, unit diagonal
        upper = np.triu(est, 1)
        est = upper + upper.T
        np.fill_diagonal(est, 1.0)
    return est


def sample_bloch_vectors(B: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Shot-noise estimate of Pauli expectations (entries in [-1, 1])."""
    return 2.0 * rng.binomial(shots, np.clip((1.0 + B) / 2.0, 0.0, 1.0)) / shots - 1.0


def clip_to_psd(K: np.ndarray) -> np.ndarray:
    """Nearest positive semi-definite matrix in Frobenius norm (negative eigenvalues set to 0)."""
    w, V = eigh((K + K.T) / 2.0)
    return (V * np.clip(w, 0.0, None)) @ V.T


def fidelity_shot_count(n_train: int, n_test: int, shots: int) -> int:
    """Circuit executions to estimate one fidelity model's training and test kernels."""
    return shots * (n_train * (n_train - 1) // 2 + n_test * n_train)


def projected_shot_count(n_train: int, n_test: int, shots: int) -> int:
    """Circuit executions to estimate one projected model's Bloch vectors (3 Pauli bases per point)."""
    return 3 * shots * (n_train + n_test)
