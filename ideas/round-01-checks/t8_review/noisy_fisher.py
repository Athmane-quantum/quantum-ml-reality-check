import numpy as np, importlib.util
spec = importlib.util.spec_from_file_location("fr", "fisher_ratio_lib.py"); fr = importlib.util.module_from_spec(spec); spec.loader.exec_module(fr)
rng = np.random.default_rng(5)
def fisher_EA_att(anti, lam, G, att):
    S = 1 - 2*anti.astype(float); D = len(lam)
    p = S @ (lam*att) / D; J = S @ (G*att[:,None]) / D
    return (J/p[:,None]).T @ J
def fisher_EF_att(codes, n, sets, lam, G, att):
    return sum(fr.fisher_setting(codes, n, s, lam*att, G*att[:,None]) for s in sets)/len(sets)
for n in [3,4,5]:
    D, codes, anti = fr.setup(n); gens = fr.ring_generators(n); sets = fr.nine_settings(n)
    w = (codes > 0).sum(1)
    r = rng.uniform(0.001, 0.02, len(gens)); lam, G = fr.lam_and_grad(anti, gens, r)
    for F in [1.0, 0.99, 0.97, 0.95, 0.90]:
        q = 1 - np.sqrt((4*F-1)/3)             # per-qubit depolarizing p with F_Bell=(1+3(1-p)^2)/4
        attEA = (1-q)**(2*w)                     # 2309.13461 convention: noise once on each of 2n qubits
        SEA = np.linalg.inv(fisher_EA_att(anti, lam, G, attEA))
        SEF0 = np.linalg.inv(fisher_EF_att(codes, n, sets, lam, G, np.ones(D)))   # EF noiseless
        SEFm = np.linalg.inv(fisher_EF_att(codes, n, sets, lam, G, attEA))        # EF with same per-qubit prep+meas depolarizing
        a = fr.median_linf(G, SEA); b0 = fr.median_linf(G, SEF0); bm = fr.median_linf(G, SEFm)
        print(f"n={n} F_Bell={F:.2f} (p={q:.4f}): R vs noiseless EF = {(b0/a)**2:6.2f} | R vs EF with matched (1-p)^(2|b|) SPAM = {(bm/a)**2:6.2f}", flush=True)
