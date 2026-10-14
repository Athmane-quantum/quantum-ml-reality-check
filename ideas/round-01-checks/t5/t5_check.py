"""T5 sanity check (scratch, not part of the repository).

f_alpha(x) = <x| U^dag O(alpha) U |x>,  U = exp(-i H tau), H = mixed-field Ising chain (bounded local norm).
1) Walsh-spectrum locality of f vs tau (light cone).
2) Learning curves at n=8: Molteni-style LASSO on exact Pauli features (knows U) vs black-box
   learners that see only x: local-Walsh LASSO, tuned RBF kernel ridge, tuned gradient boosting.
"""
import itertools, sys, time
import numpy as np
from scipy.linalg import expm
from sklearn.linear_model import LassoCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold

n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
rng = np.random.default_rng(0)
I2 = np.eye(2); X = np.array([[0, 1], [1, 0]]); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1., -1.])
P1 = {"X": X, "Y": Y, "Z": Z}

def op(sites_ops):
    mats = [I2] * n
    for s, m in sites_ops:
        mats[s] = m
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

H = sum(op([(i, Z), (i + 1, Z)]) for i in range(n - 1)) \
    + 0.9 * sum(op([(i, X)]) for i in range(n)) + 0.5 * sum(op([(i, Z)]) for i in range(n))

# candidate Paulis: all 1-local and nearest-neighbour 2-local
cands = [((i, a),) for i in range(n) for a in "XYZ"] + \
        [((i, a), (i + 1, b)) for i in range(n - 1) for a in "XYZ" for b in "XYZ"]
Pm = [op([(s, P1[a]) for s, a in c]) for c in cands]
m = len(cands)
alpha = np.zeros(m)
nz = rng.choice(m, 6, replace=False)
alpha[nz] = rng.uniform(-1, 1, 6)
O = sum(a * P for a, P in zip(alpha, Pm) if a != 0)

xs = np.array(list(itertools.product([0, 1], repeat=n)))  # index = binary number
pm = 1 - 2 * xs  # +-1 encoding
# Walsh basis: chi_S(x) = prod_{i in S} (1-2x_i)
subsets = [S for k in range(n + 1) for S in itertools.combinations(range(n), k)]
W = np.array([[np.prod(pm[j, list(S)]) if S else 1.0 for S in subsets] for j in range(2 ** n)])
span = np.array([(max(S) - min(S) + 1) if S else 0 for S in subsets])

def diag_heis(U, A):
    return np.real(np.einsum("ji,jk,ki->i", U.conj(), A, U))  # <x|U^dag A U|x> for all x

print(f"n={n}, m={m} candidate Paulis, true support {[cands[i] for i in nz]}")
for tau in [float(t) for t in sys.argv[2].split(",")]:
    t0 = time.time()
    U = expm(-1j * tau * H)
    f = diag_heis(U, O)
    F = np.stack([diag_heis(U, P) for P in Pm], axis=1)  # exact quantum features
    c = W.T @ f / 2 ** n
    var = np.sum(c[1:] ** 2)
    loc = {w: np.sum(c[(span <= w) & (span > 0)] ** 2) / var for w in range(1, n + 1)}
    print(f"\ntau={tau}: Var f={var:.3f}; Walsh variance with span<=w: " +
          " ".join(f"w{w}:{loc[w]:.3f}" for w in range(1, n + 1)))
    for N in [int(v) for v in sys.argv[3].split(",")]:
        res = {}
        reps = 3
        for r in range(reps):
            tr = rng.choice(2 ** n, N, replace=True)
            te = np.setdiff1d(np.arange(2 ** n), tr)
            y = f + rng.normal(0, 0.02, 2 ** n)  # small label noise (~2500 shots)
            def rel(pred):
                return np.mean((pred - f[te]) ** 2) / var
            # (a) knows U: LASSO on exact Pauli features (Molteni Alg. 1)
            mdl = LassoCV(cv=min(3, N // 4), alphas=30, max_iter=3000, tol=1e-3).fit(F[tr], y[tr])
            res.setdefault("Q-feat LASSO", []).append(rel(mdl.predict(F[te])))
            # (b) black box: local-Walsh LASSO, window chosen by CV
            best = None
            for w in [2, 3, n]:
                cols = np.where(span <= w)[0]
                l = LassoCV(cv=min(3, N // 4), alphas=20, max_iter=3000, tol=1e-3).fit(W[tr][:, cols], y[tr])
                sc = np.min(np.mean(l.mse_path_, axis=1))
                if best is None or sc < best[0]:
                    best = (sc, l, cols)
            res.setdefault("local-Walsh LASSO", []).append(rel(best[1].predict(W[te][:, best[2]])))
            # (c) black box: tuned RBF kernel ridge
            g = GridSearchCV(KernelRidge(kernel="rbf"), {"alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1],
                             "gamma": [0.01, 0.03, 0.1, 0.3, 1]}, cv=KFold(3, shuffle=True, random_state=r),
                             scoring="neg_mean_squared_error").fit(pm[tr], y[tr])
            res.setdefault("RBF-KRR", []).append(rel(g.predict(pm[te])))
            # (d) black box: tuned gradient boosting (small grid)
            gb = GridSearchCV(HistGradientBoostingRegressor(max_iter=200),
                              {"learning_rate": [0.05, 0.2], "max_leaf_nodes": [3, 15],
                               "min_samples_leaf": [2, 5]}, cv=KFold(3, shuffle=True, random_state=r),
                              scoring="neg_mean_squared_error").fit(xs[tr], y[tr])
            res.setdefault("GradBoost", []).append(rel(gb.predict(xs[te])))
        print(f"  N={N:4d}  " + "  ".join(f"{k}: {np.mean(v):.3f}" for k, v in res.items()))
    print(f"  ({time.time() - t0:.0f}s)")
