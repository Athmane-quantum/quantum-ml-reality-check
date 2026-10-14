"""Scratch check: direct Monte Carlo of the chain estimator (degree 2, raw TF-IDF, lam=1) on one
20NG pair, to confirm the analytic sample-cost numbers of pilot_variance.py.
Each chain for term r draws r+1 uniform rows; budget split by Neyman allocation."""
import sys
import numpy as np
from numpy.polynomial import chebyshev as C
from numpy.polynomial import polynomial as P
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from scipy.stats import norm

DATA = sys.argv[1]; LAM = 1.0; DEG = 2
d = fetch_20newsgroups(subset="all", data_home=DATA)
texts, target = d.data, d.target
ca, cb = 0, 1
idx = np.where((target == ca) | (target == cb))[0]
y_all = np.where(target[idx] == ca, 1.0, -1.0)
docs = [texts[i] for i in idx]
tr, te = next(StratifiedKFold(5, shuffle=True, random_state=0).split(docs, y_all))
vec = TfidfVectorizer(stop_words="english")
Xtr = vec.fit_transform([docs[i] for i in tr]); Xte = vec.transform([docs[i] for i in te])
y = y_all[tr]; N = len(y)
K = (Xtr @ Xtr.T).toarray(); lmax = np.linalg.eigvalsh(K)[-1]
a, b = LAM, LAM + lmax
tt = np.cos(np.pi * (np.arange(DEG + 1) + 0.5) / (DEG + 1)); xx = 0.5 * (b - a) * tt + 0.5 * (b + a)
pt = C.cheb2poly(C.chebfit(tt, 1.0 / xx, DEG))
alpha, beta = 2.0 / (b - a), -(b + a) / (b - a); lin = np.array([alpha * LAM + beta, alpha])
poly_mu = np.zeros(1); powt = np.array([1.0])
for cj in pt:
    poly_mu = P.polyadd(poly_mu, cj * powt); powt = P.polymul(powt, lin)
Kte = (Xte @ Xtr.T).toarray(); yte = y_all[te]
alpha_exact = np.linalg.solve(K + LAM * np.eye(N), y); s_exact = Kte @ alpha_exact
rng = np.random.default_rng(5)
tests = rng.choice(len(te), 60, replace=False)
# analytic per-term moments
means, var = [], []
Kr_y = y.copy(); K2r_1 = np.ones(N); K2 = K * K
for r in range(DEG + 1):
    m = Kte @ Kr_y; s2 = N ** (r + 1) * ((Kte ** 2) @ K2r_1)
    means.append(m); var.append(np.maximum(s2 - m ** 2, 0)); Kr_y = K @ Kr_y; K2r_1 = K2 @ K2r_1
means = np.array(means); var = np.array(var)
for M in [1e4, 1e5, 1e6]:
    pred, emp = [], []
    for t in tests:
        w = np.abs(poly_mu) * np.sqrt(var[:, t] / (np.arange(DEG + 1) + 1))
        w = w / w.sum() if w.sum() > 0 else np.ones(DEG + 1) / (DEG + 1)
        rows_r = M * w  # rows allotted to term r
        m_r = np.maximum((rows_r / (np.arange(DEG + 1) + 1)).astype(int), 1)  # chains per term
        score_true = poly_mu @ means[:, t]
        sd = np.sqrt(np.sum(poly_mu ** 2 * var[:, t] / m_r))
        pred.append(norm.cdf(abs(score_true) / sd) if sd > 0 else 1.0)
        # simulate 20 repetitions
        ok = 0
        for rep in range(20):
            est = 0.0
            for r in range(DEG + 1):
                ii = rng.integers(0, N, size=(m_r[r], r + 1))
                prod = Kte[t, ii[:, 0]]
                for j in range(r):
                    prod = prod * K[ii[:, j], ii[:, j + 1]]
                prod = prod * y[ii[:, r]] * N ** (r + 1)
                est += poly_mu[r] * prod.mean()
            ok += np.sign(est) == np.sign(score_true)
        emp.append(ok / 20)
    print(f"M={M:.0e} rows: predicted P(sign correct) mean={np.mean(pred):.3f}  simulated={np.mean(emp):.3f}", flush=True)
print("agreement deg2 vs exact on these tests:", np.mean(np.sign(poly_mu @ means[:, tests]) == np.sign(s_exact[tests])))
