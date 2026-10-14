"""Scratch check for T2 triage (not part of the repository).

Memory vs memoryless single-qubit sensing of an m-step stochastic phase signal theta_1..theta_m.
Target: c = E[cos(sum_j theta_j)].
 - Memory learner: one qubit accumulates sum theta_j coherently, X-basis readout: E[W] = c, Var = 1 - c^2.
   Queries to RMSE eps: N_Q = (1 - c^2)/eps^2.
 - Memoryless learner: same qubit measured after every step (Ramsey, basis phase phi_j), all m bits
   post-processed with the TRUE model family (model-aware). Cost from the Fisher information of the
   2^m-outcome distribution w.r.t. the family parameters: N_C = g^T I^{-1} g / eps^2, g = grad c.
   Model-free memoryless estimator (product of per-step outcomes, rescaled) is also reported.
Families:
 - 'ar1'  : zero-mean Gaussian AR(1), stationary variance s^2, correlation rho; params (s, rho).
 - 'hard' : Prabhu-type constrained sum: theta_1..theta_{m-1} iid U(0,2pi), theta_m = C - sum; param C.
"""
import itertools
import numpy as np

EPS = 0.05


def char_ar1(tvecs, s, rho, m):
    idx = np.arange(m)
    Sigma = s**2 * rho ** np.abs(idx[:, None] - idx[None, :])
    q = np.einsum("ki,ij,kj->k", tvecs, Sigma, tvecs)
    return np.exp(-0.5 * q).astype(complex)


def char_hard(tvecs, C, m):
    out = np.zeros(len(tvecs), dtype=complex)
    allsame = np.all(tvecs == tvecs[:, :1], axis=1)
    out[allsame] = np.exp(1j * tvecs[allsame, 0] * C)
    return out


def outcome_probs(char_vals, tvecs, phis, m):
    # p(b|theta) = prod_j [1/2 + b_j (e^{i(theta_j-phi_j)} + e^{-i(theta_j-phi_j)})/4]
    bits = np.array(list(itertools.product([1, -1], repeat=m)))
    # coefficient for t_j in {-1,0,1}: t=0 -> 1/2 ; t=+1 -> b e^{-i phi}/4 ; t=-1 -> b e^{+i phi}/4
    P = np.zeros(len(bits))
    for k, t in enumerate(tvecs):
        coef = np.ones(len(bits), dtype=complex)
        for j in range(m):
            if t[j] == 0:
                coef *= 0.5
            else:
                coef *= bits[:, j] * np.exp(-1j * t[j] * phis[j]) / 4
        P += np.real(coef * char_vals[k])
    return P


def fisher(fun, params, h=1e-5):
    p0 = fun(params)
    grads = []
    for i in range(len(params)):
        dp = np.array(params, float); dp[i] += h
        dm = np.array(params, float); dm[i] -= h
        grads.append((fun(dp) - fun(dm)) / (2 * h))
    G = np.array(grads)
    mask = p0 > 1e-15
    return (G[:, mask] / p0[mask]) @ G[:, mask].T


def run_ar1(m, s, rho):
    tvecs = np.array(list(itertools.product([-1, 0, 1], repeat=m)))
    best = np.inf
    for phis in (np.zeros(m), np.tile([0, np.pi / 2], m)[:m]):
        f = lambda p: outcome_probs(char_ar1(tvecs, p[0], p[1], m), tvecs, phis, m)
        I = fisher(f, [s, rho])
        cfun = lambda p: np.exp(-0.5 * p[0] ** 2 * (m + 2 * sum((m - d) * p[1] ** d for d in range(1, m))))
        h = 1e-6
        g = np.array([(cfun([s + h, rho]) - cfun([s - h, rho])) / (2 * h),
                      (cfun([s, rho + h]) - cfun([s, rho - h])) / (2 * h)])
        try:
            nc = g @ np.linalg.solve(I, g) / EPS**2
        except np.linalg.LinAlgError:
            nc = np.inf
        best = min(best, nc)
    c = cfun([s, rho])
    nq = (1 - c**2) / EPS**2
    # model-free memoryless: product of per-step X outcomes has mean E[prod cos theta_j]; it does not even
    # target c (it targets a 2^-m-weighted sum over sign patterns), so report its signal-to-target ratio.
    return c, nq, best


def run_hard(m, C=0.3):
    tvecs = np.array([[0] * m, [1] * m, [-1] * m])
    best = np.inf
    for phis in (np.zeros(m), np.tile([0, np.pi / 2], m)[:m], np.r_[np.zeros(m - 1), np.pi / 2]):
        f = lambda p: outcome_probs(char_hard(tvecs, p[0], m), tvecs, phis, m)
        I = fisher(f, [C])[0, 0]
        g = -np.sin(C)
        best = min(best, g**2 / I / EPS**2 if I > 0 else np.inf)
    c = np.cos(C)
    nq = (1 - c**2) / EPS**2  # memory qubit reads cos(C) directly (could also read sin C)
    return c, nq, best


if __name__ == "__main__":
    print("family m  c  N_memory  N_memoryless(model-aware, Fisher)  ratio")
    for m in (2, 3, 4, 5, 6, 7):
        c, nq, nc = run_hard(m)
        print(f"hard   {m}  {c:.3f}  {nq:9.1f}  {nc:12.1f}  {nc/nq:9.2f}")
    for s, rho in ((0.3, 0.5), (0.3, 0.9), (0.5, 0.5)):
        for m in (2, 3, 4, 5, 6, 7):
            c, nq, nc = run_ar1(m, s, rho)
            print(f"ar1 s={s} rho={rho} m={m}  c={c:.3f}  Nq={nq:9.1f}  Nc={nc:12.1f}  ratio={nc/nq:8.2f}")
