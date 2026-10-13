import numpy as np
import pytest

from qmlrc.shots import (
    clip_to_psd, fidelity_shot_count, projected_shot_count, sample_bloch_vectors, sample_fidelity_kernel,
)


def test_fidelity_estimator_is_unbiased_with_binomial_variance():
    rng = np.random.default_rng(0)
    K = np.full((200, 200), 0.3)
    draws = np.stack([sample_fidelity_kernel(K, 100, rng) for _ in range(50)])
    assert abs(draws.mean() - 0.3) < 0.003
    assert abs(draws.var() - 0.3 * 0.7 / 100) < 2e-4


def test_training_kernel_estimate_is_symmetric_with_unit_diagonal():
    rng = np.random.default_rng(1)
    A = rng.uniform(size=(30, 30))
    K = (A + A.T) / 2
    est = sample_fidelity_kernel(K, 50, rng, symmetric=True)
    np.testing.assert_array_equal(est, est.T)
    np.testing.assert_array_equal(np.diag(est), 1.0)


def test_bloch_estimator_is_unbiased_and_bounded():
    rng = np.random.default_rng(2)
    B = np.full((500, 6), -0.4)
    est = np.stack([sample_bloch_vectors(B, 200, rng) for _ in range(20)])
    assert abs(est.mean() + 0.4) < 0.005
    assert est.min() >= -1.0 and est.max() <= 1.0
    # variance of 2*Binomial(R, p)/R - 1 is 4 p (1 - p) / R with p = (1 + <P>) / 2
    p = 0.3
    assert abs(est.var() - 4 * p * (1 - p) / 200) < 3e-4


def test_estimates_are_deterministic_given_a_seed():
    K = np.full((5, 5), 0.5)
    a = sample_fidelity_kernel(K, 10, np.random.default_rng(7))
    b = sample_fidelity_kernel(K, 10, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_clip_to_psd_removes_negative_eigenvalues_and_keeps_psd_matrices():
    M = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])  # indefinite
    assert np.linalg.eigvalsh(M).min() < 0
    assert np.linalg.eigvalsh(clip_to_psd(M)).min() > -1e-10
    P = np.array([[2.0, 1.0], [1.0, 2.0]])
    np.testing.assert_allclose(clip_to_psd(P), P, atol=1e-12)


def test_shot_counts():
    assert fidelity_shot_count(3, 2, 10) == 10 * (3 + 6)
    assert projected_shot_count(3, 2, 10) == 3 * 10 * 5


@pytest.mark.parametrize("shots", [None, 100])
def test_shot_models_run(small_split, shots):
    from qmlrc.models import make_fidelity_qsvm, make_projected_qsvm

    for factory in (make_fidelity_qsvm, make_projected_qsvm):
        result = factory(shots=shots)(small_split, 0)
        assert np.all(np.isfinite(result.scores)) and result.scores.shape == (len(small_split.y_test),)
