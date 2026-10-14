import numpy as np
rng = np.random.default_rng(0)
M = 200000
# F_hard: c = cos C, C ~ U[-0.5,0.5]
C = rng.uniform(-0.5, 0.5, M); c = np.cos(C)
print("F_hard (any m): prior mean %.4f prior SD %.4f  (eps=0.05)" % (c.mean(), c.std()))
# F_ar1
s = rng.uniform(0.2, 0.6, M); rho = rng.uniform(0.3, 0.95, M)
def c_ar1(s, rho, m):
    v = m + 2*sum((m-d)*rho**d for d in range(1, m))
    return np.exp(-0.5*s**2*v)
for m in range(2, 9):
    c = c_ar1(s, rho, m)
    print(f"F_ar1 m={m}: prior mean {c.mean():.4f} SD {c.std():.4f} frac(c<0.05)={np.mean(c<0.05):.3f} frac(c<0.1)={np.mean(c<0.1):.3f}")
# F_tele
a = rng.uniform(0.3, 1.5, M); q = rng.uniform(0.05, 0.4, M)
def c_tele(a, q, m):
    # E[cos(a * sum s_j)], s Markov +-1 flip q, uniform start. Transfer matrix on (s, phase)
    # v[s] = E[e^{i a sum} 1{s_last=s}]
    vp = 0.5*np.exp(1j*a); vm = 0.5*np.exp(-1j*a)
    for j in range(1, m):
        vp, vm = ((1-q)*vp + q*vm)*np.exp(1j*a), (q*vp + (1-q)*vm)*np.exp(-1j*a)
    return np.real(vp+vm)
for m in range(2, 9):
    c = c_tele(a, q, m)
    print(f"F_tele m={m}: prior mean {c.mean():.4f} SD {c.std():.4f} frac(|c|<0.05)={np.mean(abs(c)<0.05):.3f}")
