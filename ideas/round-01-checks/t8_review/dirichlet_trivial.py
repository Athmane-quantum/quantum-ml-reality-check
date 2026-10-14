import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("fr", "fisher_ratio_lib.py"); fr = importlib.util.module_from_spec(spec); spec.loader.exec_module(fr)
rng = np.random.default_rng(3)
for n in [1,2,3,4,5]:
    D, codes, anti = fr.setup(n); S = 1 - 2*anti.astype(float)
    for alpha in [1.0, 0.1]:
        m = []
        for _ in range(50):
            p = rng.dirichlet(np.full(D, alpha)); lam = S @ p
            m.append(np.abs(lam[1:]).max())
        print(f"n={n} alpha={alpha}: median max_b|lambda_b| (= l_inf error of the all-zero estimator) = {np.median(m):.3f}")
