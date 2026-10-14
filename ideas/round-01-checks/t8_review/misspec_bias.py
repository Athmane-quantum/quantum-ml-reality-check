# M arm: true channel = ring 2-local SPL + Z^{(x)n} generator with rate rho.
# Fit the (misspecified) local model by exact MLE in the infinite-data limit (KL projection),
# separately for EA data (Bell sampling) and EF data (9-setting product design).
# Report the l_inf bias over all 4^n - 1 eigenvalues (the "bias floor") vs the 0.05 threshold,
# and how many samples a model-free / correctly-specified learner needs to beat that floor.
import numpy as np, time
from scipy.optimize import minimize
import importlib.util, sys
spec = importlib.util.spec_from_file_location("fr", __file__.replace("misspec_bias.py", "fisher_ratio_lib.py"))
fr = importlib.util.module_from_spec(spec); spec.loader.exec_module(fr)
rng = np.random.default_rng(7)

for n in [3, 4, 5]:
    D, codes, anti = fr.setup(n)
    gens = fr.ring_generators(n); K = len(gens)
    zall = fr.pauli_index([(i, 3) for i in range(n)], n)
    sets = fr.nine_settings(n)
    S = 1 - 2 * anti.astype(float)
    T = np.array([[(t >> i) & 1 for i in range(n)] for t in range(2 ** n)])
    H = (-1.0) ** (T @ T.T)
    bidx = [np.array([sum(int(s[i]) * 4 ** i for i in range(n) if T[t, i]) for t in range(2 ** n)]) for s in sets]
    for rho in [0.003, 0.01]:
        res = []
        for inst in range(4):
            r = rng.uniform(0.0, 0.02, K)
            lam_true, _ = fr.lam_and_grad(anti, gens + [zall], np.r_[r, rho])
            p_true = S @ lam_true / D
            P_true = [H @ lam_true[b] / 2 ** n for b in bidx]
            def nll_EA(th):
                lam, G = fr.lam_and_grad(anti, gens, th)
                p = S @ lam / D; J = S @ G / D
                return -(p_true @ np.log(p)), -(J.T @ (p_true / p))
            def nll_EF(th):
                lam, G = fr.lam_and_grad(anti, gens, th)
                f, g = 0.0, np.zeros(K)
                for b, Pt in zip(bidx, P_true):
                    P = H @ lam[b] / 2 ** n; J = H @ G[b] / 2 ** n
                    f -= Pt @ np.log(P) / len(sets); g -= J.T @ (Pt / P) / len(sets)
                return f, g
            out = {}
            for name, fun in [("EA-model", nll_EA), ("EF-model", nll_EF)]:
                o = minimize(fun, np.full(K, 0.01), jac=True, method="L-BFGS-B", bounds=[(1e-9, 0.3)] * K, options={"maxiter": 5000})
                if not o.success: print("  warn", name, o.message)
                lam_fit, _ = fr.lam_and_grad(anti, gens, o.x)
                out[name] = np.abs(lam_fit - lam_true)[1:].max()
            res.append(out)
        ea = np.mean([o["EA-model"] for o in res]); ef = np.mean([o["EF-model"] for o in res])
        # model-free EA l_inf at N: ~ sqrt(2 log(2*4^n)) * max sd / sqrt(N); find N where it equals floor
        print(f"n={n} rho={rho}: l_inf bias floor of best local fit  EA-model={ea:.4f}  EF-model={ef:.4f}  (threshold 0.05)", flush=True)
