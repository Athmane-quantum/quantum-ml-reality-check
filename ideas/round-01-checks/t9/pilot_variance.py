"""Pilot (scratch only): sample cost of a streaming chain estimator for the degree-d polynomial
LS-SVM score k'^T p_d(K + lam I) y, with memory = test vector + one stored sample + O(d) scalars.

Monomial form: p_d(A) = sum_r a_r K^r (expand Chebyshev series in powers of K).
Term r estimator from a chain of r+1 i.i.d. uniform rows i0..ir:
    E_r = N^{r+1} <x', x_i0> <x_i0, x_i1> ... <x_i(r-1), x_ir> y_ir
    mean   = k'^T K^r y
    2nd mo = N^{r+1} (k'.^2)^T (K.^2)^r 1
Total rows needed so that |score| >= z * std with optimal allocation over terms
(cost r+1 rows per chain):  M = (z / score)^2 * (sum_r |a_r| sqrt((r+1) Var_r))^2.
"""
import sys
import numpy as np
from numpy.polynomial import chebyshev as C
from numpy.polynomial import polynomial as P
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from scipy.sparse.linalg import eigsh

DATA = sys.argv[1]
LAM = 1.0
Z = 2.0
d = fetch_20newsgroups(subset="all", data_home=DATA)
texts, target = d.data, d.target
rng = np.random.default_rng(1)
pairs = [(0, 1), (12, 16), (5, 6)]
for norm in ["raw", "specnorm1"]:
    for deg in [0, 2, 4]:
        allM, acc, nnz_rows = [], [], []
        for (ca, cb) in pairs:
            idx = np.where((target == ca) | (target == cb))[0]
            y_all = np.where(target[idx] == ca, 1.0, -1.0)
            docs = [texts[i] for i in idx]
            tr, te = next(StratifiedKFold(5, shuffle=True, random_state=0).split(docs, y_all))
            vec = TfidfVectorizer(stop_words="english")
            Xtr = vec.fit_transform([docs[i] for i in tr])
            Xte = vec.transform([docs[i] for i in te])
            if norm == "specnorm1":
                smax = np.sqrt(eigsh((Xtr @ Xtr.T).astype(float), k=1, return_eigenvectors=False)[0])
                Xtr, Xte = Xtr / smax, Xte / smax
            nnz_rows.append(np.diff(Xtr.indptr))
            y = y_all[tr]; N = len(y)
            K = (Xtr @ Xtr.T).toarray(); K2 = K * K
            lmax = np.linalg.eigvalsh(K)[-1]
            a, b = LAM, LAM + lmax
            # Chebyshev interpolant of 1/x on [a,b] in variable t=(2x-(a+b))/(b-a), then to powers of x
            tt = np.cos(np.pi * (np.arange(deg + 1) + 0.5) / (deg + 1))
            xx = 0.5 * (b - a) * tt + 0.5 * (b + a)
            cc = C.chebfit(tt, 1.0 / xx, deg) if deg > 0 else np.array([np.mean(1.0 / xx)])
            pt = C.cheb2poly(cc)  # coefficients in t
            # t = alpha*x + beta, x = mu + lam (mu eigenvalue of K) -> polynomial in mu
            alpha, beta = 2.0 / (b - a), -(b + a) / (b - a)
            lin = np.array([alpha * LAM + beta, alpha])  # t as polynomial in mu
            poly_mu = np.zeros(1)
            powt = np.array([1.0])
            for j, cj in enumerate(pt):
                poly_mu = P.polyadd(poly_mu, cj * powt)
                powt = P.polymul(powt, lin)
            Kte = (Xte @ Xtr.T).toarray()
            yte = y_all[te]
            # moments per term
            means, var = [], []
            vk = np.tile(y, (Kte.shape[0], 1)).T  # columns per test point
            Kr_y = y.copy(); K2r_1 = np.ones(N)
            for r in range(deg + 1):
                m = Kte @ Kr_y
                s2 = N ** (r + 1) * ((Kte ** 2) @ K2r_1)
                means.append(m); var.append(np.maximum(s2 - m ** 2, 0))
                Kr_y = K @ Kr_y; K2r_1 = K2 @ K2r_1
            means = np.array(means); var = np.array(var)
            score = poly_mu[:, None].T @ means if False else np.sum(poly_mu[:, None] * means, axis=0)
            cost = np.sum(np.abs(poly_mu)[:, None] * np.sqrt((np.arange(deg + 1)[:, None] + 1) * var), axis=0) ** 2
            M = (Z / np.abs(score)) ** 2 * cost
            allM.append(M); acc.append(np.mean(np.sign(score) == yte))
        allM = np.concatenate(allM); nr = np.concatenate(nnz_rows)
        print(f"{norm:9s} deg={deg} acc={np.mean(acc):.4f}  rows needed per test point: median={np.median(allM):.3g} "
              f"p90={np.percentile(allM, 90):.3g}  N~{N}  row nnz mean={nr.mean():.0f} max={nr.max()}", flush=True)
