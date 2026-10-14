"""Scratch: per-test-point low-memory baselines (centroid, class-balanced Rocchio, kNN) vs exact LS-SVM, 5 folds."""
import sys
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
DATA = sys.argv[1]; NP = int(sys.argv[2]); LAM = 1.0
d = fetch_20newsgroups(subset="all", data_home=DATA); texts, target = d.data, d.target
rng = np.random.default_rng(11); pairs = set()
while len(pairs) < NP:
    a, b = rng.choice(20, 2, replace=False); pairs.add((min(a, b), max(a, b)))
R = {k: np.zeros((NP, 5)) for k in ["exact", "cent", "rocchio", "knn5", "knn15", "knn45"]}
for p, (ca, cb) in enumerate(sorted(pairs)):
    idx = np.where((target == ca) | (target == cb))[0]
    y_all = np.where(target[idx] == ca, 1.0, -1.0); docs = [texts[i] for i in idx]
    for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(docs, y_all)):
        vec = TfidfVectorizer(stop_words="english")
        Xtr = vec.fit_transform([docs[i] for i in tr]); Xte = vec.transform([docs[i] for i in te])
        y = y_all[tr]; yte = y_all[te]; N = len(y)
        K = (Xtr @ Xtr.T).toarray(); Kte = (Xte @ Xtr.T).toarray()
        R["exact"][p, f] = np.mean(np.sign(Kte @ np.linalg.solve(K + LAM * np.eye(N), y)) == yte)
        R["cent"][p, f] = np.mean(np.sign(Kte @ y) == yte)
        w = np.where(y > 0, 1.0 / (y > 0).sum(), -1.0 / (y < 0).sum())
        R["rocchio"][p, f] = np.mean(np.sign(Kte @ w) == yte)
        order = np.argsort(-Kte, axis=1)
        for k in [5, 15, 45]:
            R[f"knn{k}"][p, f] = np.mean(np.sign(np.sum(y[order[:, :k]] * np.take_along_axis(Kte, order[:, :k], 1), 1)) == yte)
for k, v in R.items():
    fold = v.mean(0)
    diff = R["exact"].mean(0) - fold
    print(f"{k:8s} mean acc={v.mean():.4f} per-fold={np.round(fold,4)}  exact-minus-this: mean={diff.mean():+.4f} sd={diff.std(ddof=1):.4f} folds exact higher={np.sum(diff>0)}/5")
