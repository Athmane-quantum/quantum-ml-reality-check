"""Scratch helpers for T6 triage checks (numpy-only IQP simulation; not part of the repository)."""
import importlib.util
import sys
from itertools import combinations

import numpy as np

sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location(
    "qmlrc_geometry", "/home/user/quantum-ml-reality-check/src/qmlrc/geometry.py"
)
geometry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(geometry)


def z_signs(n):
    b = (np.arange(2**n)[:, None] >> np.arange(n)[::-1][None, :]) & 1  # wire 0 = most significant bit
    return 1 - 2 * b  # +1 for |0>, -1 for |1>


def hadamard_all(psi, n):
    shp = psi.shape
    psi = psi.reshape(shp[0], *([2] * n))
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    for q in range(n):
        psi = np.moveaxis(np.tensordot(psi, h, axes=([q + 1], [1])), -1, q + 1)
    return psi.reshape(shp)


def iqp_states(X, s, n_repeats=2, pairs="all"):
    """Same circuit as PennyLane IQPEmbedding(s*x): per repeat H on all, RZ(s x_i), MultiRZ(s^2 x_i x_j)."""
    X = np.asarray(X, float) * s
    N, n = X.shape
    Z = z_signs(n)
    if pairs == "all":
        P = list(combinations(range(n), 2))
    elif pairs == "ring":
        P = [(i, (i + 1) % n) for i in range(n)]
    else:
        P = []
    phase = X @ Z.T  # sum_i x_i z_i
    for i, j in P:
        phase = phase + np.outer(X[:, i] * X[:, j], Z[:, i] * Z[:, j])
    D = np.exp(-0.5j * phase)
    psi = np.zeros((N, 2**n), complex)
    psi[:, 0] = 1.0
    for _ in range(n_repeats):
        psi = hadamard_all(psi, n)
        psi = psi * D
    return psi


def bloch(psi, n):
    N = psi.shape[0]
    t = psi.reshape(N, *([2] * n))
    out = np.zeros((N, 3 * n))
    for q in range(n):
        a = np.moveaxis(t, q + 1, 1).reshape(N, 2, -1)
        rho = np.einsum("nik,njk->nij", a, a.conj())
        out[:, q] = 2 * rho[:, 0, 1].real
        out[:, n + q] = -2 * rho[:, 0, 1].imag  # <Y> = 2 Im(rho_10) = -2 Im(rho_01)
        out[:, 2 * n + q] = (rho[:, 0, 0] - rho[:, 1, 1]).real
    return out


def fidelity(A, B):
    return np.abs(A.conj() @ B.T) ** 2


def sqdist(A, B):
    return np.maximum((A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def projected(BA, BB, gamma):
    return np.exp(-gamma * sqdist(BA, BB) / 2.0)


def min_g(Kq, X):
    return geometry.minimum_geometric_difference(geometry.decompose(geometry.classical_kernels(X)), Kq)


def psd_sqrt(K):
    w, V = np.linalg.eigh((K + K.T) / 2)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T
