"""Adversarial scratch check of the T9 best version (chain estimator vs QOS). Not a result."""
import sys, math
import numpy as np
from numpy.polynomial import chebyshev as C
from numpy.polynomial import polynomial as P
from scipy.stats import norm
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

DATA = sys.argv[1]; NP = int(sys.argv[2]); LAM = 1.0; EPS = 0.1
d = fetch_20newsgroups(subset="all", data_home=DATA)
texts, target = d.data, d.target
rng = np.random.default_rng(7)
pairs = set()
while len(pairs) < NP:
    a, b = rng.choice(20, 2, replace=False); pairs.add((min(a, b), max(a, b)))

def mono_coeffs(a, b, deg):
    tt = np.cos(np.pi * (np.arange(deg + 1) + 0.5) / (deg + 1)); xx = 0.5 * (b - a) * tt + 0.5 * (b + a)
    pt = C.cheb2poly(C.chebfit(tt, 1.0 / xx, deg)) if deg > 0 else np.array([np.mean(1.0 / xx)])
    al, be = 2.0 / (b - a), -(b + a) / (b - a); lin = np.array([al * LAM + be, al])
    pm = np.zeros(1); pw = np.array([1.0])
    for cj in pt:
        pm = P.polyadd(pm, cj * pw); pw = P.polymul(pw, lin)
    return pm

for norm_ in ["raw", "specnorm1"]:
    print("=====", norm_)
    for (ca, cb) in sorted(pairs):
        idx = np.where((target == ca) | (target == cb))[0]
        y_all = np.where(target[idx] == ca, 1.0, -1.0); docs = [texts[i] for i in idx]
        tr, te = next(StratifiedKFold(5, shuffle=True, random_state=0).split(docs, y_all))
        vec = TfidfVectorizer(stop_words="english")
        Xtr = vec.fit_transform([docs[i] for i in tr]).tocsr(); Xte = vec.transform([docs[i] for i in te]).tocsr()
        N, D = Xtr.shape
        rnnz = np.diff(Xtr.indptr); cnnz = np.diff(Xtr.tocsc().indptr); tnnz = np.diff(Xte.indptr)
        s = max(rnnz.max(), cnnz.max())
        SQ = 2 * math.ceil(math.log2(N + 2 * D)) + math.ceil(math.log2(s + 1 + 1)) + 4
        K = (Xtr @ Xtr.T).toarray()
        lmax = np.linalg.eigvalsh(K)[-1]
        if norm_ == "specnorm1":
            Xtr = Xtr / np.sqrt(lmax); Xte = Xte / np.sqrt(lmax); K = K / lmax; lmax = 1.0
        y = y_all[tr]; yte = y_all[te]
        kappa = (lmax + LAM) / LAM; MQ = N * kappa ** 2 / EPS; MQ_reg = N * kappa / EPS
        fro = np.linalg.norm(K)
        alpha = np.linalg.solve(K + LAM * np.eye(N), y); Kte = (Xte @ Xtr.T).toarray(); sx = Kte @ alpha
        acc_ex = np.mean(np.sign(sx) == yte)
        K2 = K * K
        out = []
        for interval in ["lmax", "fro"]:
            b = LAM + (lmax if interval == "lmax" else fro)
            for deg in [0, 1, 2, 3, 4]:
                pm = mono_coeffs(LAM, b, deg)
                means, var = [], []
                Kr_y = y.copy(); K2r = np.ones(N)
                for r in range(deg + 1):
                    m = Kte @ Kr_y; s2 = float(N) ** (r + 1) * ((Kte ** 2) @ K2r)
                    means.append(m); var.append(np.maximum(s2 - m ** 2, 0)); Kr_y = K @ Kr_y; K2r = K2 @ K2r
                means = np.array(means); var = np.array(var)
                score = pm @ means
                cost = (np.abs(pm)[:, None] * np.sqrt((np.arange(deg + 1)[:, None] + 1) * var)).sum(0) ** 2
                rows_z2 = (2.0 / np.abs(score)) ** 2 * cost
                acc_d = np.mean(np.sign(score) == yte); agr = np.mean(np.sign(score) == np.sign(sx))
                # expected accuracy at budgets M (Neyman allocation, normal approx): sd = sqrt(cost/M)
                res = {}
                for M in [1e5, 1e6, MQ_reg, MQ]:
                    pc = norm.cdf(np.abs(score) / np.sqrt(cost / M))
                    res[M] = np.mean(np.where(np.sign(score) == yte, pc, 1 - pc))
                out.append((interval, deg, acc_d, agr, np.median(rows_z2), np.percentile(rows_z2, 90), np.percentile(rows_z2, 99),
                            res[1e6], res[MQ_reg], res[MQ]))
        best = [o for o in out if o[0] == "lmax" and o[1] == 2][0]
        bestf = [o for o in out if o[0] == "fro" and o[1] == 2][0]
        print(f"pair {ca:2d},{cb:2d} N={N} D={D} s_row={rnnz.max()} s_col={cnnz.max()} S_Q={SQ} | nnz train mean={rnnz.mean():.0f} max={rnnz.max()} "
              f"test mean={tnnz.mean():.0f} p90={np.percentile(tnnz,90):.0f} max={tnnz.max()} | peakReg(median x')={2*np.median(tnnz)+2*rnnz.max():.0f} "
              f"meanReg={2*tnnz.mean()+2*rnnz.mean():.0f} | kappa={kappa:.1f} M_Q={MQ:.2e} M_Q(kreg^2)={MQ_reg:.2e} fro/lmax={fro/lmax:.2f} exact={acc_ex:.4f}")
        for o in out:
            print(f"   {o[0]:4s} d={o[1]} acc={o[2]:.4f} agree={o[3]:.4f} rows50={o[4]:.2e} rows90={o[5]:.2e} rows99={o[6]:.2e} "
                  f"Eacc@1e6={o[7]:.4f} Eacc@N*kap/eps={o[8]:.4f} Eacc@M_Q={o[9]:.4f}")
        sys.stdout.flush()
