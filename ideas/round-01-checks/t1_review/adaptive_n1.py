"""Exact optimal ADAPTIVE single-copy learner at n=1 restricted to Pauli-axis (stabilizer) measurements, by DP.
Task: Def. 2 of 2112.00778 at n=1 (H0: I/2; H1: (I + 0.9 s P)/2, P in {X,Y,Z}); success = sum_{D,O} (1/6) max(P(D|H0), P(D|H1,O)).
Compare with the best non-adaptive axis allocation (balanced = MUB cycling)."""
from functools import lru_cache
from itertools import product
a = 0.9
def leaf(c):  # c = ((p,m) for X,Y,Z)
    N = sum(p+m for p, m in c); p0 = 0.5**N; tot = 0.0
    for o in range(3):
        p, m = c[o]; rest = N - p - m
        p1 = 0.5**rest * 0.5*(((1+a)/2)**p*((1-a)/2)**m + ((1-a)/2)**p*((1+a)/2)**m)
        tot += max(p0, p1)/6
    return tot
def nonadaptive(alloc):  # alloc = (kX,kY,kZ)
    tot = 0.0
    def rec(i, c):
        nonlocal tot
        if i == 3: tot += leaf(tuple(c)) * 1.0; return
        from math import comb
        k = alloc[i]
        for p in range(k+1):
            # leaf() already includes per-sequence probabilities; multiply by number of sequences
            rec_c = c + [(p, k-p)]
            if i == 2:
                tot += comb(k, p)*leaf_mult(rec_c, alloc)
            else:
                rec_mult(i+1, rec_c, comb(k, p))
    return None
from math import comb
def nonadaptive_succ(alloc):
    s = 0.0
    for pX in range(alloc[0]+1):
        for pY in range(alloc[1]+1):
            for pZ in range(alloc[2]+1):
                mult = comb(alloc[0], pX)*comb(alloc[1], pY)*comb(alloc[2], pZ)
                s += mult*leaf(((pX, alloc[0]-pX), (pY, alloc[1]-pY), (pZ, alloc[2]-pZ)))
    return s
@lru_cache(maxsize=None)
def V(c, rem):
    # value of the subtree: sum over continuations of leaf values (leaf() already holds sequence probabilities)
    if rem == 0: return leaf(c)
    best = 0.0
    for ax in range(3):
        tot = 0.0
        for out in (0, 1):
            cc = list(c); p, m = cc[ax]; cc[ax] = (p+1, m) if out == 0 else (p, m+1)
            tot += V(tuple(cc), rem-1)
        best = max(best, tot)
    return best
for N in range(2, 13):
    ad = V(((0,0),(0,0),(0,0)), N)
    na = max(nonadaptive_succ(al) for al in product(range(N+1), repeat=3) if sum(al) == N)
    print(f'N={N}: optimal adaptive (axis meas.) {ad:.4f}   best non-adaptive allocation {na:.4f}')
