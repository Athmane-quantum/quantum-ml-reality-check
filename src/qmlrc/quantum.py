"""Quantum feature maps and kernels, simulated exactly with PennyLane.

Feature map: PennyLane's ``IQPEmbedding`` (the ZZ-feature-map family of
Havlicek et al., arXiv:1804.11326): Hadamards, ``RZ(s * x_i)`` on every qubit
and ``MultiRZ(s**2 * x_i * x_j)`` on every pair, repeated ``n_repeats`` times.
``s`` is a bandwidth (input scaling) hyperparameter that is tuned like the
``gamma`` of a classical RBF kernel. With ``entangling=False`` the MultiRZ gates
are dropped (``pattern=[]``): the same map without entanglement, used as the
"separable" control of Bowles et al. (arXiv:2403.07059).

Everything is a noiseless state-vector simulation on a CPU (no shot noise, no
hardware). With at most 10 qubits these models are cheap to simulate
classically, so nothing here can show a computational quantum advantage; the
benchmark only asks whether these model classes are competitive learners.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pennylane as qml

MAX_QUBITS = 10


def _check_inputs(X: np.ndarray) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[1] > MAX_QUBITS:
        raise ValueError(f"{X.shape[1]} features would need {X.shape[1]} qubits; the limit is {MAX_QUBITS}")
    return X


def _pattern(entangling: bool):
    return None if entangling else []  # None = PennyLane's default (all pairs)


@lru_cache(maxsize=None)
def _state_circuit(n_qubits: int, n_repeats: int, entangling: bool = True):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x):
        qml.IQPEmbedding(x, wires=range(n_qubits), n_repeats=n_repeats, pattern=_pattern(entangling))
        return qml.state()

    return circuit


@lru_cache(maxsize=None)
def _bloch_circuit(n_qubits: int, n_repeats: int, entangling: bool = True):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x):
        qml.IQPEmbedding(x, wires=range(n_qubits), n_repeats=n_repeats, pattern=_pattern(entangling))
        return (
            [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
            + [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]
            + [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        )

    return circuit


def iqp_states(X: np.ndarray, scale: float, n_repeats: int = 2, entangling: bool = True) -> np.ndarray:
    """State vectors |phi(s * x)>, shape (n_samples, 2**n_qubits)."""
    X = _check_inputs(X)
    out = np.asarray(_state_circuit(X.shape[1], n_repeats, entangling)(scale * X))
    return out.reshape(X.shape[0], 2 ** X.shape[1])


def bloch_vectors(X: np.ndarray, scale: float, n_repeats: int = 2, entangling: bool = True) -> np.ndarray:
    """Single-qubit Bloch vectors of |phi(s * x)>, shape (n_samples, 3 * n_qubits).

    Columns are <X_0>..<X_{n-1}>, <Y_0>..<Y_{n-1}>, <Z_0>..<Z_{n-1}>. Together they
    determine every one-qubit reduced density matrix rho_k = (I + r_k . sigma) / 2.
    """
    X = _check_inputs(X)
    out = _bloch_circuit(X.shape[1], n_repeats, entangling)(scale * X)
    return np.stack([np.asarray(o, dtype=float).reshape(X.shape[0]) for o in out], axis=1)


def fidelity_kernel(states_a: np.ndarray, states_b: np.ndarray) -> np.ndarray:
    """k(x, x') = |<phi(x)|phi(x')>|^2."""
    return np.abs(states_a.conj() @ states_b.T) ** 2


def projected_kernel(bloch_a: np.ndarray, bloch_b: np.ndarray, gamma: float) -> np.ndarray:
    """Projected quantum kernel of Huang et al. (arXiv:2011.01938).

    k(x, x') = exp(-gamma * sum_k ||rho_k(x) - rho_k(x')||_F^2), where rho_k is the
    reduced state of qubit k. For one qubit ||rho - rho'||_F^2 = |r - r'|^2 / 2.
    """
    sq = (bloch_a**2).sum(1)[:, None] + (bloch_b**2).sum(1)[None, :] - 2.0 * bloch_a @ bloch_b.T
    return np.exp(-gamma * np.maximum(sq, 0.0) / 2.0)
