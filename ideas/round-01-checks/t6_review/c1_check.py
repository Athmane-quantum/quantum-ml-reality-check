"""Scratch review check (not repo): reverse construction C1 on creditcard_q8 inputs, seed 0."""
import sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, "/tmp/claude-0/-home-user-quantum-ml-reality-check/3242e8f8-b193-546c-b381-3fe888e7979d/scratchpad/round01/t6")
sys.path.insert(0, "/home/user/quantum-ml-reality-check/src")
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from qmlrc.data import load_creditcard, make_split
from sim_common import geometry, iqp_states, fidelity, sqdist, psd_sqrt
t0 = time.time(); seed = 0
DH = "/tmp/claude-0/-home-user-quantum-ml-reality-check/3242e8f8-b193-546c-b381-3fe888e7979d/scratchpad/round01/t6_review/data"
Xp, yp = load_creditcard(DH)
sp = make_split(Xp, yp, n_samples=2000, positive_fraction=0.10, test_size=0.3, n_components=8, n_folds=3, seed=seed)
Xtr, Xte = sp.X_train, sp.X_test; N = len(Xtr); n = 8
Xall = np.vstack([Xtr, Xte]); tr = np.arange(N); te = np.arange(N, len(Xall))
SC = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]; RBFF = [0.1, 0.3, 1.0, 3.0, 10.0]
Kfid = {s: fidelity(iqp_states(Xall, s), iqp_states(Xall, s)) for s in SC}
base = 1.0 / (n * Xtr.var()); D2 = sqdist(Xall, Xall)
Krbf = {f: np.exp(-f * base * D2) for f in RBFF}
T = lambda K: K[np.ix_(tr, tr)]
print(f"kernels [{time.time()-t0:.0f}s]")
# role-swapped geometric difference: how far is each RBF kernel from the closest quantum-grid kernel?
for f in RBFF:
    best = None
    for s in SC:
        for lam in geometry.LAMBDAS:
            gg, gt = geometry.geometric_difference(T(Kfid[s]), T(Krbf[f]), lam)
            if gt < geometry.MAX_TRAINING_ERROR and (best is None or gg < best[0]): best = (gg, s, lam)
    gq1 = None
    for lam in geometry.LAMBDAS:
        gg, gt = geometry.geometric_difference(T(Kfid[1.0]), T(Krbf[f]), lam)
        if gt < geometry.MAX_TRAINING_ERROR and (gq1 is None or gg < gq1[0]): gq1 = (gg, lam)
    print(f"RBF f={f}: min_s g(K_Q(s)||K_RBF) = {best[0]:.2f} at s={best[1]}, lam={best[2]:g};  g(K_Q(s=1)||K_RBF) = {gq1[0]:.2f} (lam={gq1[1]:g})")
C8 = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 1e4]
def pre_svm(Kd, y, folds):
    best = None
    for name, K in Kd.items():
        gs = GridSearchCV(SVC(kernel="precomputed"), {"C": C8}, cv=folds, scoring="accuracy").fit(T(K), y[tr])
        if best is None or gs.best_score_ > best[0]: best = (gs.best_score_, name, gs.best_estimator_, K)
    return best[1], (best[2].predict(best[3][np.ix_(te, tr)]) == y[te]).mean()
rng = np.random.default_rng(7)
for fgen in (1.0, 0.3):
    KG = geometry.normalise(T(Krbf[fgen])); KA = geometry.normalise(T(Kfid[1.0]))  # generator = RBF, adversary = K_Q(s=1)
    # lambda by the geometry.py rule (largest g with g_tra<0.045 equivalently min over admissible?) -> use admissible lam with max g
    adm = [(geometry.geometric_difference(T(Kfid[1.0]), T(Krbf[fgen]), lam), lam) for lam in geometry.LAMBDAS]
    adm = [(g[0], lam) for g, lam in adm if g[1] < geometry.MAX_TRAINING_ERROR]
    g_used, lam = min(adm)
    sG, sA = psd_sqrt(KG), psd_sqrt(KA); M = np.linalg.inv(KA + lam * np.eye(N))
    A = sG @ sA @ M @ M @ sA @ sG; w, V = np.linalg.eigh((A + A.T) / 2)
    yp_ = sG @ V[:, -1]; wq, Vq = np.linalg.eigh(KG); keep = wq > 1e-8 * wq.max()
    alpha = Vq[:, keep] @ ((Vq[:, keep].T @ yp_) / wq[keep])
    fa = (N / np.trace(T(Krbf[fgen]))) * Krbf[fgen][:, tr] @ alpha; med = np.median(fa[tr])
    clean = (fa > med).astype(int); y = clean.copy(); fl = rng.random(len(y)) < 0.1; y[fl] = 1 - y[fl]
    folds = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(Xtr, y[tr]))
    q = pre_svm({f"s={s}": Kfid[s] for s in SC}, y, folds)
    q1 = pre_svm({"s=1": Kfid[1.0]}, y, folds)
    c = pre_svm({f"f={f}": Krbf[f] for f in RBFF}, y, folds)
    print(f"C1 generator RBF f={fgen}, adversary K_Q(s=1), g={g_used:.2f}, lam={lam:g}: QSVM tuned over s -> {q[0]} acc={q[1]:.3f}; "
          f"QSVM fixed s=1 acc={q1[1]:.3f}; RBF-SVM -> {c[0]} acc={c[1]:.3f}; Bayes={(clean[te]==y[te]).mean():.3f} [{time.time()-t0:.0f}s]", flush=True)
