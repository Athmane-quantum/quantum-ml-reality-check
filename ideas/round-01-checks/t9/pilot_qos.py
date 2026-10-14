"""Pilot (scratch only): can a low-memory, per-test-point classical estimator reproduce the
LS-SVM decision rule used by Zhao et al. (arXiv:2604.07639) on 20 Newsgroups?

Task (theirs): TF-IDF (English stop words), LS-SVM w = (X^T X + lam I)^-1 X^T y, lam = 1 for
20NG, 5-fold CV accuracy averaged over random category pairs, predict sign(x'^T w) for a
single test point.

Classical per-test-point estimators considered here:
  P_d : score_d(x') = k'^T p_d(K) y  with K = X X^T, k' = X x', p_d a degree-d polynomial
        approximation of (K + lam I)^-1 (Chebyshev on [lam, lam + ||K||]).
        A streaming machine can estimate each monomial k'^T K^r y by a chain of r+1 random
        samples, holding only the test vector and the previous sample (memory ~ 2 s + O(1)
        floats, s = row sparsity), at a sample cost set by the estimator variance.
  C0  : degree-0 term only, sign(sum_i y_i <x', x_i>) (a centroid classifier).
We report sign agreement with the exact LS-SVM and the accuracy of each.
"""
import sys
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from scipy.sparse.linalg import eigsh

DATA = sys.argv[1]
rng = np.random.default_rng(0)
d = fetch_20newsgroups(subset="all", data_home=DATA, remove=())
texts, target = d.data, d.target
NPAIRS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
LAM = 1.0
DEGS = [0, 1, 2, 4, 8, 16]


def cheb_coeffs(a, b, deg):
    """Chebyshev interpolation coefficients of 1/x on [a, b]."""
    n = deg + 1
    k = np.arange(n)
    t = np.cos(np.pi * (k + 0.5) / n)
    x = 0.5 * (b - a) * t + 0.5 * (b + a)
    f = 1.0 / x
    c = np.array([2.0 / n * np.sum(f * np.cos(j * np.pi * (k + 0.5) / n)) for j in range(n)])
    c[0] /= 2
    return c


def cheb_apply(K, v, a, b, c):
    """p(K) v with p the Chebyshev series c on [a, b] (K symmetric, spectrum in [a,b])."""
    alpha, beta = 2.0 / (b - a), -(b + a) / (b - a)
    T0 = v
    out = c[0] * T0
    if len(c) == 1:
        return out
    T1 = alpha * (K @ v) + beta * v
    out = out + c[1] * T1
    for j in range(2, len(c)):
        T2 = 2 * (alpha * (K @ T1) + beta * T1) - T0
        out = out + c[j] * T2
        T0, T1 = T1, T2
    return out


rows = []
pairs = set()
while len(pairs) < NPAIRS:
    a, b = rng.choice(20, 2, replace=False)
    pairs.add((min(a, b), max(a, b)))
for norm in ["raw", "specnorm1"]:
    for (ca, cb) in sorted(pairs):
        idx = np.where((target == ca) | (target == cb))[0]
        y_all = np.where(target[idx] == ca, 1.0, -1.0)
        docs = [texts[i] for i in idx]
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        acc = {"exact": [], **{f"P{dd}": [] for dd in DEGS}}
        agree = {f"P{dd}": [] for dd in DEGS}
        kap = []
        for tr, te in skf.split(docs, y_all):
            vec = TfidfVectorizer(stop_words="english")
            Xtr = vec.fit_transform([docs[i] for i in tr])
            Xte = vec.transform([docs[i] for i in te])
            if norm == "specnorm1":
                smax = np.sqrt(eigsh((Xtr @ Xtr.T).astype(float), k=1, return_eigenvectors=False)[0])
                Xtr = Xtr / smax
                Xte = Xte / smax
            y = y_all[tr]
            K = (Xtr @ Xtr.T).toarray()
            lmax = np.linalg.eigvalsh(K)[-1]
            kap.append((lmax + LAM) / LAM)
            alpha_exact = np.linalg.solve(K + LAM * np.eye(len(y)), y)
            Kte = (Xte @ Xtr.T).toarray()
            s_exact = Kte @ alpha_exact
            yte = y_all[te]
            acc["exact"].append(np.mean(np.sign(s_exact) == yte))
            for dd in DEGS:
                c = cheb_coeffs(LAM, LAM + lmax, dd)
                al = cheb_apply(K + LAM * np.eye(len(y)), y, LAM, LAM + lmax, c)
                s = Kte @ al
                acc[f"P{dd}"].append(np.mean(np.sign(s) == yte))
                agree[f"P{dd}"].append(np.mean(np.sign(s) == np.sign(s_exact)))
        row = dict(norm=norm, pair=(ca, cb), N=len(tr), kappa=np.mean(kap),
                   **{k: np.mean(v) for k, v in acc.items()},
                   **{"agree_" + k: np.mean(v) for k, v in agree.items()})
        rows.append(row)
        print(norm, (ca, cb), "N=%d kappa=%.1f exact=%.3f" % (row["N"], row["kappa"], row["exact"]),
              " ".join("P%d=%.3f(ag %.3f)" % (dd, row[f"P{dd}"], row[f"agree_P{dd}"]) for dd in DEGS),
              flush=True)

for norm in ["raw", "specnorm1"]:
    R = [r for r in rows if r["norm"] == norm]
    print("MEAN", norm, "kappa=%.1f exact=%.4f" % (np.mean([r["kappa"] for r in R]), np.mean([r["exact"] for r in R])),
          " ".join("P%d=%.4f(ag %.4f)" % (dd, np.mean([r[f"P{dd}"] for r in R]), np.mean([r[f"agree_P{dd}"] for r in R])) for dd in DEGS))
