import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "/tmp/claude-0/-home-user-quantum-ml-reality-check/3242e8f8-b193-546c-b381-3fe888e7979d/scratchpad/round01/t6")
import numpy as np
from sim_common import iqp_states, fidelity, sqdist
rng = np.random.default_rng(0); X = rng.normal(size=(120, 8))
D2 = sqdist(X, X); off = ~np.eye(len(X), dtype=bool)
for s in (0.005, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
    K = fidelity(iqp_states(X, s), iqp_states(X, s))
    c = ((1 - K[off]) / (s**2 * D2[off])).mean()
    G = np.exp(-0.25 * s**2 * D2)
    print(f"s={s:5.3f}: (1-K)/(s^2|u|^2) mean={c:.4f}; max|K - exp(-s^2|u|^2/4)|={np.abs(K-G)[off].max():.3f}; mean offdiag K={K[off].mean():.3f}")
