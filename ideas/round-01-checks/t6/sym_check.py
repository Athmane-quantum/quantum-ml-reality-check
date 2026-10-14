"""T6 sanity check: is a permutation-symmetrised quantum kernel 'far' from classical kernels only because
the classical comparison kernel is not symmetrised?  6 qubits, S_6, exact simulation, one seed."""
import time
from itertools import permutations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from sim_common import geometry, iqp_states, fidelity, sqdist, min_g

t0 = time.time()
rng = np.random.default_rng(1)
n, N, ntr = 6, 300, 200
X = rng.normal(size=(N, n))
perms = [np.array(p) for p in permutations(range(n))]
base = 1.0 / (n * X.var())


def gauss_avg(X, f):
    K = np.zeros((len(X), len(X)))
    for p in perms:
        K += np.exp(-f * base * sqdist(X, X[:, p]))
    return K / len(perms)


def q_avg(X, s):
    S = iqp_states(X, s)
    K = np.zeros((len(X), len(X)))
    for p in perms:
        K += fidelity(S, iqp_states(X[:, p], s))
    return K / len(perms)


def min_g_over(Kq, kernels):
    return geometry.minimum_geometric_difference(geometry.decompose(kernels), Kq)


FS = geometry.GAUSSIAN_FACTORS
plain = geometry.classical_kernels(X)
avg = {f"gaussG x{f:g}": gauss_avg(X, f) for f in FS}
print(f"classical kernels built {time.time()-t0:.1f}s")

rows = []
KQ, KQG = {}, {}
for s in (0.3, 0.7, 1.0):
    KQ[s] = fidelity(iqp_states(X, s), iqp_states(X, s))
    KQG[s] = q_avg(X, s)
    g1 = min_g_over(KQ[s], plain)["g"]
    g2 = min_g_over(KQG[s], plain)["g"]
    g3 = min_g_over(KQG[s], avg)["g"]
    g4 = min_g_over(KQG[s], {**plain, **avg})["g"]
    print(f"s={s}: g(RBF||Q)={g1:.2f}  g(RBF||Q^G)={g2:.2f}  g(RBF^G||Q^G)={g3:.2f}  g(all||Q^G)={g4:.2f}  sqrtN={np.sqrt(N):.1f}")
print(f"kernels+g {time.time()-t0:.1f}s")

# invariant target: label = sign(elementary symmetric e2(x) + 0.5*sum cos(x_i) - median), 10% flips
e2 = ((X.sum(1)) ** 2 - (X**2).sum(1)) / 2
score = e2 + 2.0 * np.cos(X).sum(1)
y = (score > np.median(score)).astype(int)
flip = rng.random(N) < 0.1
y[flip] = 1 - y[flip]
tr, te = np.arange(ntr), np.arange(ntr, N)
cv = StratifiedKFold(3, shuffle=True, random_state=0)
Cs = [0.01, 0.1, 1, 10, 100, 1000]


def fit_pre(Ks: dict):
    best = None
    for name, K in Ks.items():
        gs = GridSearchCV(SVC(kernel="precomputed"), {"C": Cs}, cv=cv).fit(K[np.ix_(tr, tr)], y[tr])
        if best is None or gs.best_score_ > best[0]:
            best = (gs.best_score_, name, gs.best_estimator_, K)
    acc = (best[2].predict(best[3][np.ix_(te, tr)]) == y[te]).mean()
    return best[1], best[0], acc


res = {
    "RBF": fit_pre({f"x{f:g}": plain[f"gaussian x{f:g}"] for f in FS}),
    "RBF^G (perm-averaged)": fit_pre({k: v for k, v in avg.items()}),
    "Q fidelity": fit_pre({f"s={s}": KQ[s] for s in KQ}),
    "Q^G fidelity (perm-averaged)": fit_pre({f"s={s}": KQG[s] for s in KQG}),
}
Xs = np.sort(X, 1)
plain_sorted = {f"x{f:g}": np.exp(-f * base * sqdist(Xs, Xs)) for f in FS}
res["RBF on sorted x"] = fit_pre(plain_sorted)
gb = GridSearchCV(HistGradientBoostingClassifier(), {"learning_rate": [0.03, 0.1, 0.3], "max_leaf_nodes": [7, 31]}, cv=cv)
gb.fit(Xs[tr], y[tr])
res["GBT on sorted x"] = ("", gb.best_score_, (gb.predict(Xs[te]) == y[te]).mean())
for k, (sel, cvs, acc) in res.items():
    print(f"{k:32s} sel={sel:12s} cv={cvs:.3f} test_acc={acc:.3f}")
print(f"total {time.time()-t0:.1f}s")
