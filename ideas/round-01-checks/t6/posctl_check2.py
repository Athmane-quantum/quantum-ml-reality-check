"""T6 feasibility check for the positive control (Huang et al. 2011.01938 App. G / L14-L15) at 8 qubits.
One seed, synthetic Gaussian inputs, N = 800. Not a result: it only checks that g is large enough at 8 qubits
for the construction to bite, and what it costs."""
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sim_common import geometry, iqp_states, bloch, fidelity, projected, sqdist, min_g, psd_sqrt

t0 = time.time()
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
rng = np.random.default_rng(seed)
n, N, ntr = 8, 800, 560
X = rng.normal(size=(N, n))
tr, te = np.arange(ntr), np.arange(ntr, N)
cv = StratifiedKFold(3, shuffle=True, random_state=seed)
Cs = [0.01, 0.1, 1, 10, 100, 1000, 1e4]
cl_kernels = geometry.classical_kernels(X)
cl_dec = geometry.decompose(cl_kernels)


def make_labels(Kq, info):
    Kc = geometry.normalise(cl_kernels[info["classical_kernel"]])
    Kqn = geometry.normalise(Kq)
    lam = info["lambda"]
    sKq, sKc = psd_sqrt(Kqn), psd_sqrt(Kc)
    M = np.linalg.inv(Kc + lam * np.eye(N))
    A = sKq @ sKc @ M @ M @ sKc @ sKq
    w, V = np.linalg.eigh((A + A.T) / 2)
    yr = sKq @ V[:, -1]
    y = (yr > np.median(yr)).astype(int)
    flip = rng.random(N) < 0.1
    y[flip] = 1 - y[flip]
    return y


def svm_pre(Ks, y):
    best = None
    for name, K in Ks.items():
        gs = GridSearchCV(SVC(kernel="precomputed"), {"C": Cs}, cv=cv).fit(K[np.ix_(tr, tr)], y[tr])
        if best is None or gs.best_score_ > best[0]:
            best = (gs.best_score_, name, gs.best_estimator_, K)
    return best[1], (best[2].predict(best[3][np.ix_(te, tr)]) == y[te]).mean()


states = {s: iqp_states(X, s) for s in (0.1, 0.3, 1.0)}
blochs = {s: bloch(states[s], n) for s in states}
qk = {}
for s in states:
    qk[("fid", s)] = fidelity(states[s], states[s])
    b = blochs[s]
    g0 = 1.0 / max(np.median(sqdist(b, b)) / 2.0, 1e-12)  # median heuristic for gamma
    qk[("proj", s)] = projected(b, b, g0)
print(f"kernels {time.time()-t0:.1f}s")

rbf = rbf_pre = None
rbf = {f"x{f:g}": np.exp(-f * (1.0 / (n * X[tr].var())) * sqdist(X, X)) for f in (0.03, 0.1, 0.3, 1, 3, 10)}
for key, Kq in qk.items():
    info = geometry.minimum_geometric_difference(cl_dec, Kq)
    y = make_labels(Kq, info)
    ypm = 2.0 * y - 1.0
    lam_b = 1e-3
    def cplx(K):
        Kn = geometry.normalise(K)
        return float(ypm @ np.linalg.solve(Kn + lam_b * np.eye(N), ypm))
    sQ = cplx(Kq)
    sC = min(cplx(K) for K in list(cl_kernels.values()) + list(rbf.values()))
    print(f"   binary-label complexity: s_Q={sQ:.0f} min s_C={sC:.0f} ratio={sC/sQ:.2f} (N={N})")
    q_name, q_acc = svm_pre({f"{key[0]} s={s}": qk[(key[0], s)] for s in (0.1, 0.3, 1.0)}, y)
    c_name, c_acc = svm_pre(rbf, y)
    gb = GridSearchCV(HistGradientBoostingClassifier(random_state=0),
                      {"learning_rate": [0.03, 0.1, 0.3], "max_leaf_nodes": [7, 31], "l2_regularization": [0, 1]}, cv=cv)
    gb.fit(X[tr], y[tr]); gb_acc = (gb.predict(X[te]) == y[te]).mean()
    mlp = GridSearchCV(MLPClassifier(max_iter=2000, random_state=0),
                       {"hidden_layer_sizes": [(64,), (32, 32)], "alpha": [1e-4, 1e-2, 1]}, cv=cv)
    mlp.fit(X[tr], y[tr]); mlp_acc = (mlp.predict(X[te]) == y[te]).mean()
    print(f"{key}: g={info['g']:.2f} (vs {info['classical_kernel']}, lam={info['lambda']:g}) sqrtN={np.sqrt(N):.1f} | "
          f"Q[{q_name}]={q_acc:.3f} RBF[{c_name}]={c_acc:.3f} GBT={gb_acc:.3f} MLP={mlp_acc:.3f} | {time.time()-t0:.0f}s")
