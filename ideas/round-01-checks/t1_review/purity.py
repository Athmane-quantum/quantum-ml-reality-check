"""Purity arm (P5) exact check. H0: I/d. H1: Haar-random pure state. Equal priors.
Quantum: SWAP outcome from each Bell pair; pure -> +1 always; mixed -> +1 w.p. (1+1/d)/2. Decide pure iff all +1.
Classical: all copies measured in one fixed basis; Bayes LR depends on the multiset of counts:
P1(seq) = (d-1)! prod m_i! / (N+d-1)!, P0(seq) = d^-N. Monte Carlo over H0 and H1 sequences."""
import numpy as np
from math import lgamma, log
rng = np.random.default_rng(1)
def quantum(n, N):
    d = 2**n; T = N//2; q = ((1+1/d)/2)**T
    return 0.5*(1 + (1 - q))
def llr(counts, d, N):
    return (lgamma(d) + sum(lgamma(m+1) for m in counts) - lgamma(N+d)) + N*log(d)
def classical(n, N, inst=20000):
    d = 2**n; ok = 0
    for _ in range(inst):
        x0 = rng.integers(0, d, N); c0 = np.bincount(x0, minlength=d)
        p = rng.exponential(size=d); p /= p.sum(); x1 = rng.choice(d, N, p=p); c1 = np.bincount(x1, minlength=d)
        ok += (llr(c0, d, N) < 0) + (llr(c1, d, N) > 0) + 0.5*(llr(c0, d, N) == 0) + 0.5*(llr(c1, d, N) == 0)
    return ok/(2*inst)
GRID = [2,4,6,8,12,16,24,32,48,64]
for n in range(1, 6):
    nq = next(N for N in GRID if quantum(n, N) >= 0.8)
    nc = next((N for N in GRID if classical(n, N, 4000) >= 0.8), '>64')
    print(n, 'N*_Q', nq, 'N*_C(fixed-basis Bayes)', nc)
