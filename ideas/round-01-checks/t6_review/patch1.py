p = "posctl_fraud.py"; s = open(p).read()
a = """    folds = list(cv().split(Xtr, y[tr]))
    out = {}
    out["QSVM-fid"] = pre_svm({f"s={s}": Kfid[s] for s in SC}, y, folds)
    out["RBF-SVM"] = pre_svm({f"f={f}": Krbf[f] for f in RBFF}, y, folds)
    out["GBT"] = feat_model("""
b = """    folds = list(cv().split(Xtr, y[tr]))
    out = {}; tm = {}
    t = time.time(); out["QSVM-fid"] = pre_svm({f"s={s}": Kfid[s] for s in SC}, y, folds); tm["QSVM-fid"] = time.time() - t
    t = time.time(); out["RBF-SVM"] = pre_svm({f"f={f}": Krbf[f] for f in RBFF}, y, folds); tm["RBF-SVM"] = time.time() - t
    t = time.time(); out["GBT"] = feat_model("""
assert a in s; s = s.replace(a, b)
a = """{"learning_rate": [0.03, 0.1, 0.3], "max_leaf_nodes": [7, 31], "min_samples_leaf": [5, 20], "l2_regularization": [0, 10]}, Xall, y, folds)
    out["MLP"] = feat_model(MLPClassifier(max_iter=1000, random_state=0),
        {"hidden_layer_sizes": [(16,), (64,), (32, 32)], "alpha": [1e-4, 1e-2, 1, 10]}, Xall, y, folds)"""
b = """{"learning_rate": [0.03, 0.1, 0.3], "max_leaf_nodes": [7, 31], "min_samples_leaf": [5, 20], "l2_regularization": [0, 10]}, Xall, y, folds); tm["GBT"] = time.time() - t
    t = time.time(); out["MLP"] = feat_model(MLPClassifier(max_iter=2000, random_state=0),
        {"hidden_layer_sizes": [(16,), (64,), (32, 32)], "alpha": [1e-4, 1e-2, 1, 10]}, Xall, y, folds); tm["MLP"] = time.time() - t
    t = time.time()"""
assert a in s; s = s.replace(a, b)
a = """    out["IQP-angle"] = best
"""
assert a in s; s = s.replace(a, """    out["IQP-angle"] = best; tm["IQP-angle"] = time.time() - t
""")
a = """    for k, v in out.items(): print(f"   {k:9s} sel={v[0]}  acc={v[1]:.3f}  AUPRC={v[2]:.3f}")"""
assert a in s; s = s.replace(a, """    for k, v in out.items(): print(f"   {k:9s} sel={v[0]}  acc={v[1]:.3f}  AUPRC={v[2]:.3f}  wall={tm[k]:.0f}s", flush=True)""")
s = s.replace("print(f\"\\nDOSE", "print(flush=True); print(f\"\\nDOSE")
open(p, "w").write(s)
print("ok")
