"""Arm B: (1) ideal HNN = measurement in the joint eigenbasis of all 8 stabilisers + Z_L (what the inverse
preparation circuit does), with per-shot Bayes acceptance and the paper's majority/mean rule;
(2) the proposal's non-learned collision-rate test on Pauli-6 data (threshold 7/12), with the real magnetised states;
(3) the proposal's 'independent pairs' back-of-envelope.
"""
import numpy as np
from functools import reduce
from scipy.stats import binom

rng = np.random.default_rng(0)
n = 9
X_STABS = [(0, 1, 3, 4), (4, 5, 7, 8), (1, 2), (6, 7)]
Z_STABS = [(1, 2, 4, 5), (3, 4, 6, 7), (0, 3), (5, 8)]
I2 = np.eye(2); Xm = np.array([[0, 1], [1, 0]], complex); Zm = np.diag([1., -1.]).astype(complex)
Ym = np.array([[0, -1j], [1j, 0]])


def op(p, sites):
    m = [I2] * n
    for s in sites:
        m[s] = p
    return reduce(np.kron, m)


gens = [op(Xm, s) for s in X_STABS] + [op(Zm, s) for s in Z_STABS] + [op(Zm, (0, 1, 2))]
for g in gens[:4]:
    assert np.allclose(g @ gens[8], gens[8] @ g)
c = rng.normal(size=9)
w, V = np.linalg.eigh(sum(ci * g for ci, g in zip(c, gens)))
assert np.min(np.diff(np.sort(w))) > 1e-6
syn = np.array([[np.real(V[:, k].conj() @ g @ V[:, k]) for g in gens] for k in range(512)])
assert np.allclose(np.abs(syn), 1)

hs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
st = np.load("sc_states.npy")
P1 = {h: np.abs(V.conj().T @ st[i]) ** 2 for i, h in enumerate(hs)}
train = [0.0, 0.05, 0.15]  # paper App. B training fields for N=9
P1tr = np.mean([P1[h] for h in train], axis=0)
accept = P1tr > 2.0 ** -9
print("ideal-HNN accept set size:", accept.sum(), " single-shot accept prob per topological state:",
      {h: round(float(P1[h][accept].sum()), 4) for h in hs})

# product states (1-design): Monte Carlo over states, exact per-state acceptance probability
one = {('Z', 0): np.array([1, 0]), ('Z', 1): np.array([0, 1]), ('X', 0): np.array([1, 1]) / np.sqrt(2),
       ('X', 1): np.array([1, -1]) / np.sqrt(2), ('Y', 0): np.array([1, 1j]) / np.sqrt(2), ('Y', 1): np.array([1, -1j]) / np.sqrt(2)}
keys = list(one)
M = 100000
idx = rng.integers(0, 6, size=(M, n))
Pacc = np.empty(M)
VA = V[:, accept]
for b0 in range(0, M, 5000):
    blk = idx[b0:b0 + 5000]
    phis = np.array([reduce(np.kron, [one[keys[k]] for k in row]) for row in blk])
    amp = phis @ VA.conj()
    Pacc[b0:b0 + 5000] = (np.abs(amp) ** 2).sum(1)
print("product states: mean single-shot accept prob", Pacc.mean(), " max", Pacc.max())


def maj_fp(S):  # mean-of-y > 1/2 with binary y; ties count as reject
    return float(np.mean(1 - binom.cdf(np.floor(S / 2), S, Pacc)))


def maj_fn(S, q):
    return float(binom.cdf(np.floor(S / 2), S, q))


print("ideal HNN (majority rule):  S  FP  FN(max over h<=0.15)  FN(h=0.25)  balanced acc (h<=0.15)")
for S in [1, 2, 3, 4, 5, 7]:
    fns = [maj_fn(S, P1[h][accept].sum()) for h in hs]
    fp = maj_fp(S)
    print(S, f"{fp:.2e}", f"{max(fns[:4]):.2e}", f"{fns[5]:.2e}", f"{1 - (fp + np.mean(fns[:4])) / 2:.6f}")

# ---------------- collision test on Pauli-6 data ----------------
U = {'Z': np.eye(2), 'X': np.array([[1, 1], [1, -1]]) / np.sqrt(2), 'Y': np.array([[1, -1j], [1, 1j]]) / np.sqrt(2)}


def sample_topo(psi, S):
    bases = rng.integers(0, 3, size=(S, n))
    outs = np.empty((S, n), int)
    for t in range(S):
        Ut = reduce(np.kron, [U['ZXY'[b]] for b in bases[t]])
        p = np.abs(Ut @ psi) ** 2; p /= p.sum()
        x = rng.choice(512, p=p)
        outs[t] = [(x >> (n - 1 - i)) & 1 for i in range(n)]
    return bases, outs


def sample_prod(S):
    eb = rng.integers(0, 3, size=n); ev = rng.integers(0, 2, size=n)
    bases = rng.integers(0, 3, size=(S, n))
    outs = np.where(bases == eb, ev, rng.integers(0, 2, size=(S, n)))
    return bases, outs


def coll_rate(bases, outs):
    c = p = 0
    S = len(bases)
    for i in range(n):
        for t in range(S):
            for u in range(t + 1, S):
                if bases[t, i] == bases[u, i]:
                    p += 1; c += outs[t, i] == outs[u, i]
    return c / p if p else 0.5


print("\ncollision test (threshold 7/12), Pauli-6 random bases, test states h in {0..0.25} (as proposed) and h<=0.15")
for S in [7, 13, 20]:
    T = 600
    rp = np.array([coll_rate(*sample_prod(S)) for _ in range(T)])
    rt = {h: np.array([coll_rate(*sample_topo(st[i], S)) for _ in range(T // 3)]) for i, h in enumerate(hs)}
    tpr_all = np.mean([np.mean(rt[h] < 7 / 12) for h in hs])
    tpr_low = np.mean([np.mean(rt[h] < 7 / 12) for h in hs[:4]])
    tnr = np.mean(rp >= 7 / 12)
    print(f"S={S}: TNR={tnr:.3f}  TPR(h<=0.25)={tpr_all:.3f}  TPR(h<=0.15)={tpr_low:.3f}  "
          f"bal.acc(all)={(tnr + tpr_all) / 2:.3f}  bal.acc(h<=0.15)={(tnr + tpr_low) / 2:.3f}  "
          f"TPR per h={[round(float(np.mean(rt[h] < 7 / 12)), 2) for h in hs]}")

# proposal's 'independent pairs' estimate: all C(S,2)/3 same-basis pairs per qubit treated as iid
from scipy.stats import norm
for S in [7, 13, 20]:
    m = int(round(n * S * (S - 1) / 2 / 3))
    thr = 7 / 12 * m
    tnr = 1 - binom.cdf(np.floor(thr), m, 2 / 3)
    tpr = binom.cdf(np.floor(thr), m, 1 / 2)
    print(f"iid-pairs estimate S={S}: m={m} bal.acc={(tnr + tpr) / 2:.5f}")
