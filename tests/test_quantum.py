from functools import reduce

import numpy as np
import pytest

from qmlrc.quantum import bloch_vectors, fidelity_kernel, iqp_states, projected_kernel

H1 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
PAULI = {"X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1.0, -1.0])}


def reference_iqp_state(x, n_repeats, entangling=True):
    """Independent NumPy implementation of PennyLane's IQPEmbedding (wire 0 = most significant bit)."""
    n = len(x)
    bits = (np.arange(2**n)[:, None] >> np.arange(n)[::-1]) & 1
    s = 1 - 2 * bits  # Z eigenvalues
    phase = (s * x).sum(1)
    if entangling:
        phase = phase + sum(x[i] * x[j] * s[:, i] * s[:, j] for i in range(n) for j in range(i + 1, n))
    diag = np.exp(-0.5j * phase)
    Hn = reduce(np.kron, [H1] * n)
    psi = np.zeros(2**n, dtype=complex)
    psi[0] = 1.0
    for _ in range(n_repeats):
        psi = diag * (Hn @ psi)
    return psi


def reference_bloch(psi, n):
    out = []
    for name in "XYZ":
        for k in range(n):
            op = reduce(np.kron, [PAULI[name] if q == k else np.eye(2) for q in range(n)])
            out.append(np.real(psi.conj() @ op @ psi))
    return np.array(out)


@pytest.mark.parametrize("n_repeats", [1, 2])
def test_states_match_independent_implementation(rng, n_repeats):
    X = rng.normal(size=(5, 3))
    S = iqp_states(X, scale=0.7, n_repeats=n_repeats)
    for x, psi in zip(X, S):
        ref = reference_iqp_state(0.7 * x, n_repeats)
        assert abs(abs(np.vdot(ref, psi)) - 1.0) < 1e-10  # equal up to a global phase


def test_bloch_vectors_match_independent_implementation(rng):
    X = rng.normal(size=(4, 3))
    B = bloch_vectors(X, scale=0.9)
    for x, b in zip(X, B):
        np.testing.assert_allclose(b, reference_bloch(reference_iqp_state(0.9 * x, 2), 3), atol=1e-10)


def test_fidelity_kernel_is_a_valid_kernel(rng):
    S = iqp_states(rng.normal(size=(30, 4)), scale=0.5)
    K = fidelity_kernel(S, S)
    np.testing.assert_allclose(np.linalg.norm(S, axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    assert K.min() >= 0.0 and K.max() <= 1.0 + 1e-10
    assert np.linalg.eigvalsh(K).min() > -1e-8


def test_projected_kernel_is_a_valid_kernel(rng):
    B = bloch_vectors(rng.normal(size=(30, 4)), scale=0.5)
    assert B.shape == (30, 12)
    K = projected_kernel(B, B, gamma=2.0)
    np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    assert np.linalg.eigvalsh(K).min() > -1e-8


def test_projected_kernel_matches_reduced_density_matrix_definition(rng):
    X = rng.normal(size=(2, 2))
    B = bloch_vectors(X, scale=1.0)
    rho = lambda b, k: 0.5 * (np.eye(2) + sum(b[i * 2 + k] * PAULI[p] for i, p in enumerate("XYZ")))  # noqa: E731
    dist = sum(np.linalg.norm(rho(B[0], k) - rho(B[1], k), "fro") ** 2 for k in range(2))
    assert np.isclose(projected_kernel(B[:1], B[1:], gamma=0.8)[0, 0], np.exp(-0.8 * dist))


def test_tiny_bandwidth_makes_kernel_constant(rng):
    S = iqp_states(rng.normal(size=(10, 4)), scale=1e-4)
    assert fidelity_kernel(S, S).min() > 0.999


def test_qubit_limit_is_enforced(rng):
    with pytest.raises(ValueError):
        iqp_states(rng.normal(size=(2, 11)), scale=1.0)


def test_separable_states_match_reference_and_are_product_states(rng):
    X = rng.normal(size=(4, 3))
    S = iqp_states(X, scale=0.8, entangling=False)
    for x, psi in zip(X, S):
        assert abs(abs(np.vdot(reference_iqp_state(0.8 * x, 2, entangling=False), psi)) - 1.0) < 1e-10
    B = bloch_vectors(X, scale=0.8, entangling=False).reshape(4, 3, 3)  # (sample, XYZ, qubit)
    np.testing.assert_allclose(np.linalg.norm(B, axis=1), 1.0, atol=1e-10)  # every qubit stays pure


def test_separable_fidelity_kernel_is_product_of_one_qubit_kernels(rng):
    X = rng.normal(size=(5, 3))
    S = iqp_states(X, scale=0.6, entangling=False)
    K = fidelity_kernel(S, S)
    one = [fidelity_kernel(iqp_states(X[:, [q]], 0.6, entangling=False), iqp_states(X[:, [q]], 0.6, entangling=False))
           for q in range(3)]
    np.testing.assert_allclose(K, one[0] * one[1] * one[2], atol=1e-10)


def test_entangling_gates_change_the_state(rng):
    X = rng.normal(size=(1, 3))
    a, b = iqp_states(X, 1.0)[0], iqp_states(X, 1.0, entangling=False)[0]
    assert abs(abs(np.vdot(a, b)) - 1.0) > 1e-3
