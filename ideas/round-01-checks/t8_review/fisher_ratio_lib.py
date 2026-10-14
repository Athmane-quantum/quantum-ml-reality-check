# Asymptotic (Cramer-Rao / MLE) sample ratio R(n) = N*_EF / N*_EA for the proposal's
# S family (1D-ring 2-local sparse Pauli-Lindblad channel, rates U(0, 0.02)), using the
# proposal's own metric: l_inf error over all 4^n - 1 Pauli eigenvalues.
# EA = Bell sampling (one Pauli-error sample per use).
# EF = 9-setting product design (van den Berg et al.), equal allocation, noiseless SPAM.
# Under MLE asymptotics delta r ~ N(0, I^{-1}/N); delta lambda_b ~ g_b . delta r;
# N* scales as (median max_b |g_b . z|)^2, so R = (m_EF / m_EA)^2, independent of eps.
import numpy as np, sys, time
rng = np.random.default_rng(1)

def setup(n):
    D = 4 ** n
    codes = np.array([[(a // 4 ** i) % 4 for i in range(n)] for a in range(D)], dtype=np.int8)
    X = np.isin(codes, [1, 2]).astype(np.int8); Z = np.isin(codes, [2, 3]).astype(np.int8)
    anti = ((X.astype(np.int32) @ Z.T.astype(np.int32) + Z.astype(np.int32) @ X.T.astype(np.int32)) % 2).astype(np.int8)
    return D, codes, anti

def pauli_index(sites_codes, n):
    a = 0
    for i, c in sites_codes: a += c * 4 ** i
    return a

def ring_generators(n):
    gens = set()
    for i in range(n):
        for c in (1, 2, 3): gens.add(pauli_index([(i, c)], n))
    bonds = [(0, 1)] if n == 2 else [(i, (i + 1) % n) for i in range(n)]
    for i, j in bonds:
        for c1 in (1, 2, 3):
            for c2 in (1, 2, 3): gens.add(pauli_index([(i, c1), (j, c2)], n))
    return sorted(gens)

def nine_settings(n):
    if n % 2 == 0: col = [i % 2 for i in range(n)]
    else: col = [i % 2 for i in range(n - 1)] + [2]
    if n == 3: col = [0, 1, 2]
    runs = [(i, j, (i + j) % 3) for i in range(3) for j in range(3)]
    return [np.array([run[col[s]] + 1 for s in range(n)]) for run in runs]  # 1=X 2=Y 3=Z

def lam_and_grad(anti, gens, r):
    M = anti[:, gens].astype(float)                  # D x K
    loglam = M @ np.log(1 - 2 * r)
    lam = np.exp(loglam)
    G = -2 * M * lam[:, None] / (1 - 2 * r)[None, :]  # d lambda_b / d r_k
    return lam, G

def fisher_EA(anti, lam, G):
    S = 1 - 2 * anti.astype(float)                   # (-1)^{<c,b>}
    D = len(lam)
    p = S @ lam / D
    J = S @ G / D
    return (J / p[:, None]).T @ J, p

def fisher_setting(codes, n, sigma, lam, G):
    # stabilizer group of product basis sigma: b with b_i in {0, sigma_i}
    T = np.array([[(t >> i) & 1 for i in range(n)] for t in range(2 ** n)])
    b_idx = np.array([sum(int(sigma[i]) * 4 ** i for i in range(n) if T[t, i]) for t in range(2 ** n)])
    H = (-1.0) ** (T @ T.T)                          # (-1)^{e.T}, e indexed like T
    P = H @ lam[b_idx] / 2 ** n
    J = H @ G[b_idx] / 2 ** n
    return (J / P[:, None]).T @ J

def median_linf(G, Sigma, draws=4000):
    Lc = np.linalg.cholesky(Sigma + 1e-18 * np.eye(len(Sigma)))
    z = rng.standard_normal((draws, len(Sigma))) @ Lc.T
    return np.median(np.abs(z @ G[1:].T).max(axis=1))

def median_linf_modelfree_EA(lam, anti, draws=4000):
    # empirical WHT eigenvalues: Cov(l_b, l_b') = lam_{b+b'} - lam_b lam_b' per sample
    D = len(lam)
    # build covariance only over b != 0 (dense, D<=1024)
    idx = np.arange(D)
    # b+b' symplectic sum = XOR of index digits in base 4 with X/Z bits: compute via codes
    return None

