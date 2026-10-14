"""Scratch review check (not repo): out-of-sample engineered labels on the creditcard_q8 inputs, seed 0."""
import sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, "/tmp/claude-0/-home-user-quantum-ml-reality-check/3242e8f8-b193-546c-b381-3fe888e7979d/scratchpad/round01/t6")
sys.path.insert(0, "/home/user/quantum-ml-reality-check/src")
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score
from qmlrc.data import load_creditcard, make_split
from sim_common import geometry, iqp_states, fidelity, sqdist, psd_sqrt, bloch, projected

t0 = time.time()
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
DH = "/tmp/claude-0/-home-user-quantum-ml-reality-check/3242e8f8-b193-546c-b381-3fe888e7979d/scratchpad/round01/t6_review/data"
Xp, yp = load_creditcard(DH)
sp = make_split(Xp, yp, n_samples=2000, positive_fraction=0.10, test_size=0.3, n_components=8, n_folds=3, seed=seed)
Xtr, Xte = sp.X_train, sp.X_test
N = len(Xtr); n = 8
print(f"seed {seed}: N_train={N}, max|x| train={np.abs(Xtr).max():.1f}, kurtosis-ish frac |x|>3: {(np.abs(Xtr)>3).mean():.3f}  [{time.time()-t0:.0f}s]")
Xall = np.vstack([Xtr, Xte])
SC = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
st = {s: iqp_states(Xall, s) for s in SC}
Kfid = {s: fidelity(st[s], st[s]) for s in SC}  # (2000,2000) train rows first
Kgen = dict(Kfid); Kgen[0.3] = fidelity(iqp_states(Xall, 0.3), iqp_states(Xall, 0.3))
base = 1.0 / (n * Xtr.var())
D2 = sqdist(Xall, Xall)
RBFF = [0.1, 0.3, 1.0, 3.0, 10.0]
Krbf = {f: np.exp(-f * base * D2) for f in RBFF}
tr = np.arange(N); te = np.arange(N, len(Xall))
fam = {"linear": Xall @ Xall.T}
for f in sorted(set(geometry.GAUSSIAN_FACTORS) | set(RBFF)):
    fam[f"g{f:g}"] = np.exp(-f * base * D2)
famdec = geometry.decompose({k: v[np.ix_(tr, tr)] for k, v in fam.items()})
print(f"kernels done [{time.time()-t0:.0f}s]; mean offdiag fid s=1: {Kfid[1.0][np.ix_(tr,tr)][~np.eye(N,dtype=bool)].mean():.3f}, s=0.3: {Kgen[0.3][np.ix_(tr,tr)][~np.eye(N,dtype=bool)].mean():.3f}")

def oos_labels(KQ_full, KC_tr, lam, rng, flip=0.1):
    KQ = geometry.normalise(KQ_full[np.ix_(tr, tr)]); KC = geometry.normalise(KC_tr)
    sQ, sC = psd_sqrt(KQ), psd_sqrt(KC)
    M = np.linalg.inv(KC + lam * np.eye(N))
    A = sQ @ sC @ M @ M @ sC @ sQ
    w, V = np.linalg.eigh((A + A.T) / 2)
    yprime = sQ @ V[:, -1]
    wq, Vq = np.linalg.eigh(KQ)
    keep = wq > 1e-8 * wq.max()
    alpha = Vq[:, keep] @ ((Vq[:, keep].T @ yprime) / wq[keep])
    scale = N / np.trace(KQ_full[np.ix_(tr, tr)])
    f_all = scale * KQ_full[:, tr] @ alpha
    med = np.median(f_all[tr])
    clean = (f_all > med).astype(int)
    y = clean.copy(); fl = rng.random(len(y)) < flip; y[fl] = 1 - y[fl]
    return y, clean, f_all, med, w[-1]

def cv(): return StratifiedKFold(3, shuffle=True, random_state=seed)
C8 = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 1e4]
def pre_svm(Kd, y, folds):
    best = None
    for name, K in Kd.items():
        gs = GridSearchCV(SVC(kernel="precomputed"), {"C": C8}, cv=folds, scoring="accuracy").fit(K[np.ix_(tr, tr)], y[tr])
        if best is None or gs.best_score_ > best[0]: best = (gs.best_score_, name, gs.best_estimator_, K)
    m = best[2]; Kt = best[3][np.ix_(te, tr)]
    return best[1], (m.predict(Kt) == y[te]).mean(), average_precision_score(y[te], m.decision_function(Kt)), best[0]
def feat_model(est, grid, F, y, folds):
    gs = GridSearchCV(est, grid, cv=folds, scoring="accuracy", n_jobs=4).fit(F[tr], y[tr])
    p = gs.predict(F[te]); sc = gs.predict_proba(F[te])[:, 1] if hasattr(gs, "predict_proba") else gs.decision_function(F[te])
    return gs.best_params_, (p == y[te]).mean(), average_precision_score(y[te], sc)
def angle_feats(X, s):
    i, j = np.triu_indices(n, 1)
    ang = np.hstack([s * X, s**2 * X[:, i] * X[:, j]])
    return np.hstack([np.cos(ang), np.sin(ang)])

def evaluate(label, y, clean, extra_q=None):
    folds = list(cv().split(Xtr, y[tr]))
    out = {}; tm = {}
    t = time.time(); out["QSVM-fid"] = pre_svm({f"s={s}": Kfid[s] for s in SC}, y, folds); tm["QSVM-fid"] = time.time() - t
    t = time.time(); out["RBF-SVM"] = pre_svm({f"f={f}": Krbf[f] for f in RBFF}, y, folds); tm["RBF-SVM"] = time.time() - t
    t = time.time(); out["GBT"] = feat_model(HistGradientBoostingClassifier(random_state=0),
        {"learning_rate": [0.03, 0.1, 0.3], "max_leaf_nodes": [7, 31], "min_samples_leaf": [5, 20], "l2_regularization": [0, 10]}, Xall, y, folds); tm["GBT"] = time.time() - t
    t = time.time(); out["MLP"] = feat_model(MLPClassifier(max_iter=2000, random_state=0),
        {"hidden_layer_sizes": [(16,), (64,), (32, 32)], "alpha": [1e-4, 1e-2, 1, 10]}, Xall, y, folds); tm["MLP"] = time.time() - t
    t = time.time()
    # IQP-angle surrogate: RBF on [cos, sin] of IQP angles, s over the quantum grid; select by CV accuracy
    best = None
    for s in SC:
        F = angle_feats(Xall, s); F = (F - F[tr].mean(0)) / (F[tr].std(0) + 1e-12)
        b0 = 1.0 / (F.shape[1] * F[tr].var()); D = sqdist(F, F)
        r = pre_svm({f"s={s},f={f}": np.exp(-f * b0 * D) for f in (0.3, 1, 3)}, y, folds)
        if best is None or r[3] > best[3]: best = r
    out["IQP-angle"] = best; tm["IQP-angle"] = time.time() - t
    print(f"--- {label}: Bayes acc (clean labels vs noisy test) = {(clean[te]==y[te]).mean():.3f}")
    for k, v in out.items(): print(f"   {k:9s} sel={v[0]}  acc={v[1]:.3f}  AUPRC={v[2]:.3f}  wall={tm[k]:.0f}s", flush=True)
    return out

rng = np.random.default_rng(1000 + seed)
import os
for s0 in tuple(float(v) for v in os.environ.get("DOSES", "1.0,0.3").split(",")):
    info = geometry.minimum_geometric_difference(famdec, Kgen[s0][np.ix_(tr, tr)])
    name = info["classical_kernel"]
    KC = fam[name][np.ix_(tr, tr)]
    y, clean, f_all, med, g2 = oos_labels(Kgen[s0], KC, info["lambda"], rng)
    ftr, fte = f_all[tr] - med, f_all[te] - med
    print(flush=True); print(f"\nDOSE fidelity s0={s0}: min g={info['g']:.2f} vs {name}, lam={info['lambda']:g} (sqrtN={np.sqrt(N):.1f}); top eig={g2:.2f}")
    print(f"   f spread: sd train={ftr.std():.3g}, sd test={fte.std():.3g}, ratio={fte.std()/ftr.std():.2f}; test balance={clean[te].mean():.2f}; "
          f"corr(test f, nearest-train f)=n/a [{time.time()-t0:.0f}s]")
    evaluate(f"fid s0={s0} OOS", y, clean)
    print(f"   [{time.time()-t0:.0f}s]")
