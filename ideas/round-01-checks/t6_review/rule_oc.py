"""Operating characteristics of the DESIGN.md section-6 rule with 5 seeds (scratch, not repo)."""
import numpy as np
rng = np.random.default_rng(0)
R = 400_000
def wins(d):  # d: (R,5) paired differences A-B; "A beats B"
    return ((d > 0).sum(1) >= 4) & (d.mean(1) > d.std(1, ddof=1))
d = rng.normal(size=(R, 5))
p1 = wins(d).mean(); p2 = wins(-d).mean()
print(f"null: P(A beats B)={p1:.4f}  P(B beats A)={p2:.4f}  two-sided={p1+p2:.4f}")
for m in (6, 8, 12, 16, 24):
    # independent pairs (upper bound on failure); and correlated pairs via a shared quantum-model seed effect
    ind = 1 - (1 - (p1 + p2)) ** m
    # correlated: pairs share the quantum model's per-seed noise (rho = 0.5)
    q = rng.normal(size=(R // 20, 5)); ok = np.ones(R // 20, bool)
    for _ in range(m):
        dd = np.sqrt(0.5) * q + np.sqrt(0.5) * rng.normal(size=q.shape)
        ok &= ~(wins(dd) | wins(-dd))
    print(f"m={m:2d} pairs: P(at least one false 'beats') indep={ind:.2f}, shared-noise rho=0.5: {1-ok.mean():.2f}")
print("power vs true mean paired difference delta (in units of the paired SD sigma):")
for delta in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
    print(f"  delta={delta:4.2f} sigma: P(beats)={wins(rng.normal(delta, 1, size=(R, 5))).mean():.3f}")
