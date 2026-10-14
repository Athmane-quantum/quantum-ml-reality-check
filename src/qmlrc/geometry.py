"""Geometric difference between classical and quantum kernels (Huang et al., arXiv:2011.01938).

For kernels normalised to trace N, the regularised geometric difference is (their eq. F19 / L8)

    g_gen = sqrt(|| sqrt(K_Q) K_C (K_C + lam I)^-2 sqrt(K_Q) ||_inf)

with training-error term (eq. F20 / L9)

    g_tra = lam * sqrt(|| sqrt(K_Q) (K_C + lam I)^-2 sqrt(K_Q) ||_inf).

Following their Appendix L, we report the minimum g_gen over a list of classical kernels (linear and
Gaussian) and regularisation strengths, keeping only pairs with g_tra < 0.045. The model complexity
satisfies s_C <= g^2 s_Q, so the prediction-error bound of a classical kernel model is at most g
times that of the quantum one, *for any labels*. When g is much smaller than sqrt(N) this rules out
a large quantum prediction advantage, though not a modest one. Only a g of order sqrt(N) leaves room
for a large advantage.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

# Appendix L of Huang et al.: lambda list (L10) and Gaussian gamma list (L11), in units of 1 / (n Var[x])
LAMBDAS = (1e-5, 1e-4, 1e-3, 0.01, 0.025, 0.05, 0.1)
GAUSSIAN_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
MAX_TRAINING_ERROR = 0.045


def normalise(K: np.ndarray) -> np.ndarray:
    """Rescale so that trace(K) = N, as assumed by the bounds."""
    return K * (len(K) / np.trace(K))


def _top_eigenvalue(P: np.ndarray, d: np.ndarray) -> float:
    """Largest eigenvalue of diag(sqrt d) P diag(sqrt d).

    With K_C = V diag(w) V^T and P = V^T K_Q V, this equals the largest eigenvalue of
    sqrt(K_Q) V diag(d) V^T sqrt(K_Q), because A^(1/2) K_Q A^(1/2) and K_Q^(1/2) A K_Q^(1/2)
    have the same non-zero spectrum. No square root of K_Q is needed.
    """
    r = np.sqrt(d)
    M = r[:, None] * P * r[None, :]
    try:
        v0 = np.full(len(d), 1.0 / np.sqrt(len(d)))  # fixed start vector: deterministic output
        return float(eigsh(M, k=1, which="LA", return_eigenvectors=False, tol=1e-7, ncv=40, maxiter=20000, v0=v0)[0])
    except ArpackNoConvergence:  # clustered top of the spectrum: fall back to a dense solver
        return float(eigh(M, eigvals_only=True, subset_by_index=[len(d) - 1, len(d) - 1])[0])


def _project(V: np.ndarray, K_q: np.ndarray) -> np.ndarray:
    P = V.T @ normalise(K_q) @ V
    return (P + P.T) / 2.0


def _g(w: np.ndarray, P: np.ndarray, lam: float) -> tuple[float, float]:
    g_gen = np.sqrt(max(_top_eigenvalue(P, w / (w + lam) ** 2), 0.0))
    g_tra = lam * np.sqrt(max(_top_eigenvalue(P, 1.0 / (w + lam) ** 2), 0.0))
    return float(g_gen), float(g_tra)


def geometric_difference(K_c: np.ndarray, K_q: np.ndarray, lam: float) -> tuple[float, float]:
    """(g_gen, g_tra) for one classical kernel, one quantum kernel and one regularisation."""
    w, V = eigh(normalise(K_c))
    return _g(np.clip(w, 0.0, None), _project(V, K_q), lam)


def classical_kernels(X: np.ndarray) -> dict[str, np.ndarray]:
    """Linear kernel and Gaussian kernels with gamma = f / (n Var[x]) (Huang et al., eq. L11)."""
    sq = (X**2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0.0)
    base = 1.0 / (X.shape[1] * X.var())
    kernels = {"linear": X @ X.T}
    for f in GAUSSIAN_FACTORS:
        kernels[f"gaussian x{f:g}"] = np.exp(-f * base * d2)
    return kernels


def decompose(kernels: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Eigen-decompose each normalised classical kernel once (reused for every quantum kernel)."""
    out = {}
    for name, K in kernels.items():
        w, V = eigh(normalise(K))
        out[name] = (np.clip(w, 0.0, None), V)
    return out


def minimum_geometric_difference(classical: dict[str, tuple[np.ndarray, np.ndarray]], K_q: np.ndarray) -> dict:
    """Smallest g_gen over classical kernels and lambdas with g_tra < MAX_TRAINING_ERROR."""
    best = None
    for name, (w, V) in classical.items():
        P = _project(V, K_q)
        for lam in LAMBDAS:
            g_gen, g_tra = _g(w, P, lam)
            if g_tra < MAX_TRAINING_ERROR and (best is None or g_gen < best["g"]):
                best = {"g": g_gen, "g_tra": g_tra, "lambda": lam, "classical_kernel": name}
    if best is None:  # no pair satisfies the training-error condition
        best = {"g": float("nan"), "g_tra": float("nan"), "lambda": float("nan"), "classical_kernel": "none"}
    best["sqrt_n"] = float(np.sqrt(len(K_q)))
    return best
