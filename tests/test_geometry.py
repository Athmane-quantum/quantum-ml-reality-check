import numpy as np

from qmlrc.geometry import (
    MAX_TRAINING_ERROR, classical_kernels, decompose, geometric_difference, minimum_geometric_difference, normalise,
)


def rbf(X, gamma):
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * d2)


def test_geometric_difference_of_a_kernel_with_itself_is_one(rng):
    K = rbf(rng.normal(size=(40, 3)), 0.5)
    g_gen, g_tra = geometric_difference(K, K, lam=1e-9)
    assert abs(g_gen - 1.0) < 1e-4 and g_tra < 1e-3


def test_geometric_difference_matches_hand_computation_for_diagonal_kernels():
    # K_C = diag(c), K_Q = diag(q), both already trace-normalised (trace = N = 3):
    # g_gen^2 = max_i q_i c_i / (c_i + lam)^2 and g_tra^2 = lam^2 max_i q_i / (c_i + lam)^2
    c, q, lam = np.array([2.0, 0.5, 0.5]), np.array([0.5, 0.5, 2.0]), 0.1
    g_gen, g_tra = geometric_difference(np.diag(c), np.diag(q), lam)
    assert np.isclose(g_gen, np.sqrt(np.max(q * c / (c + lam) ** 2)))
    assert np.isclose(g_tra, lam * np.sqrt(np.max(q / (c + lam) ** 2)))


def test_normalise_sets_trace_to_n(rng):
    K = rbf(rng.normal(size=(7, 2)), 1.0) * 3.0
    assert np.isclose(np.trace(normalise(K)), 7.0)


def test_minimum_respects_training_error_and_beats_every_single_pair(rng):
    X = rng.normal(size=(50, 3))
    K_q = rbf(X, 3.0)
    kernels = classical_kernels(X)
    best = minimum_geometric_difference(decompose(kernels), K_q)
    assert best["g_tra"] < MAX_TRAINING_ERROR
    assert np.isclose(best["sqrt_n"], np.sqrt(50))
    g_one, g_tra_one = geometric_difference(kernels["gaussian x1"], K_q, 0.01)
    if g_tra_one < MAX_TRAINING_ERROR:
        assert best["g"] <= g_one + 1e-9
