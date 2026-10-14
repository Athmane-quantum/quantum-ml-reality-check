"""Continuous N*(0.8) for noisy two-copy Bell sampling vs MUB-cycling single-copy stabilizer bases.
Interface local depolarizing lam on every copy: tr(P rho) -> (1-lam)^|P| tr(P rho).  Exact weight law."""
import numpy as np
from math import comb
from scipy.stats import binom

A0 = 0.9


def wdist(n):
    w = np.arange(1, n + 1)
    p = np.array([comb(n, k) * 3**k for k in w], float)
    return w, p / p.sum()


def s_known(k, b):  # H0 fair signs vs H1 bias b toward +1 (known direction; Bell eta)
    if k == 0:
        return 0.5
    j = np.arange(k + 1)
    return 0.5 * np.maximum(binom.pmf(j, k, 0.5), binom.pmf(j, k, (1 + b) / 2)).sum()


def s_unknown(k, b):  # H0 fair vs H1 bias b toward an unknown sign s
    if k == 0:
        return 0.5
    j = np.arange(k + 1)
    p1 = 0.5 * (binom.pmf(j, k, (1 + b) / 2) + binom.pmf(j, k, (1 - b) / 2))
    return 0.5 * np.maximum(binom.pmf(j, k, 0.5), p1).sum()


def bell(n, N, lam):
    w, pw = wdist(n)
    return sum(p * s_known(N // 2, A0**2 * (1 - lam) ** (2 * wi)) for wi, p in zip(w, pw))


def mub(n, N, lam):
    G = 2**n + 1
    q, r = divmod(N, G)
    w, pw = wdist(n)
    return sum(p * (r * s_unknown(q + 1, A0 * (1 - lam) ** wi) + (G - r) * s_unknown(q, A0 * (1 - lam) ** wi)) / G
               for wi, p in zip(w, pw))


def ncont(f, n, lam, step):
    hi = step
    while f(n, hi, lam) < 0.8:
        hi *= 2
        if hi > 2**20:
            return float("inf")
    lo = hi // 2
    while hi - lo > step:
        mid = (lo + hi) // 2
        mid -= mid % step
        if mid <= lo:
            break
        if f(n, mid, lam) >= 0.8:
            hi = mid
        else:
            lo = mid
    return hi


if __name__ == "__main__":
    for n in range(1, 6):
        out = []
        for lam in [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
            b = ncont(bell, n, lam, 2)
            m = ncont(mub, n, lam, 1)
            out.append(f"{lam}: {b}/{m} (R={m / b:.2f})")
        print("n", n, " | ".join(out), flush=True)
