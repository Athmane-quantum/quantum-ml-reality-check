"""Quick sanity check of predicted copy counts for the T1 best version (Theorem-6 task of arXiv:2112.00778).
H0: rho = I/d, O uniform non-identity Pauli.  H1: rho = (I + a s O)/d, O announced after measurement.
Bayes-optimal LR test for the announced O, on both sides.  Optional depolarizing interface noise lam
(shrinks a -> a (1-lam)^w on every copy, both sides) and extra memory noise mu on the stored copy (two-copy side only).
"""
import numpy as np
rng = np.random.default_rng(0)
a0 = 0.9

def weights(n, size):
    return rng.binomial(n, 0.75, size=size)   # weight of a uniform random non-identity Pauli ~ Bin(n,3/4) (ignoring w=0)

def succ_bell(n, N, lam=0.0, mu=0.0, inst=20000):
    T = N // 2
    w = weights(n, inst); w = np.maximum(w, 1)
    aeff2 = (a0**2) * (1 - lam)**(2*w) * (1 - mu)**w
    p1 = (1 + aeff2) / 2
    k1 = rng.binomial(T, p1)            # H1 counts of eta=+1
    k0 = rng.binomial(T, 0.5, size=inst)
    def llr(k, a2):
        return k*np.log1p(a2) + (T-k)*np.log1p(-a2)   # vs H0 (1/2 each): log((1+a2)^k (1-a2)^(T-k))
    l1 = llr(k1, aeff2); l0 = llr(k0, aeff2)
    s1 = np.where(l1 > 0, 1.0, np.where(l1 == 0, 0.5, 0.0))
    s0 = np.where(l0 < 0, 1.0, np.where(l0 == 0, 0.5, 0.0))
    return 0.5*(s1.mean() + s0.mean())

def lr_from_m(m, a):
    # m: (inst, N) array of per-copy "signals"; LR = 0.5[prod(1+a m) + prod(1-a m)]
    lp = np.log1p(a[:, None]*m).sum(1); lm = np.log1p(-a[:, None]*m).sum(1)
    mx = np.maximum(lp, lm)
    return mx + np.log(0.5*(np.exp(lp-mx) + np.exp(lm-mx)))

def succ_pauli(n, N, lam=0.0, inst=4000):
    w = np.maximum(weights(n, inst), 1)
    a = a0 * (1 - lam)**w
    match = rng.random((inst, N)) < (1/3.0)**w[:, None]
    s = rng.choice([-1, 1], size=inst)
    par1 = np.where(rng.random((inst, N)) < (1 + a[:, None]*s[:, None])/2, 1.0, -1.0)
    par0 = np.where(rng.random((inst, N)) < 0.5, 1.0, -1.0)
    l1 = lr_from_m(match*par1, a); l0 = lr_from_m(match*par0, a)
    s1 = np.where(l1 > 1e-12, 1.0, np.where(np.abs(l1) <= 1e-12, 0.5, 0.0))
    s0 = np.where(l0 < -1e-12, 1.0, np.where(np.abs(l0) <= 1e-12, 0.5, 0.0))
    return 0.5*(s1.mean() + s0.mean())

def succ_haar(n, N, lam=0.0, inst=4000):
    d = 2**n
    w = np.maximum(weights(n, inst), 1)
    a = a0 * (1 - lam)**w
    m0 = 2*rng.beta(d/2, d/2, size=(inst, N)) - 1
    # H1: rejection sampling to tilt by (1 + a s m)
    s = rng.choice([-1, 1], size=inst)
    m1 = np.empty((inst, N)); todo = np.ones((inst, N), bool)
    while todo.any():
        c = 2*rng.beta(d/2, d/2, size=todo.sum()) - 1
        aa = np.broadcast_to(a[:, None], (inst, N))[todo]; ss = np.broadcast_to(s[:, None], (inst, N))[todo]
        acc = rng.random(c.size) < (1 + aa*ss*c)/(1 + aa)
        idx = np.argwhere(todo)
        sel = idx[acc]
        m1[sel[:, 0], sel[:, 1]] = c[acc]
        todo[sel[:, 0], sel[:, 1]] = False
    l1 = lr_from_m(m1, a); l0 = lr_from_m(m0, a)
    return 0.5*((l1 > 0).mean() + (l0 <= 0).mean())

def ncopies(f, n, target=0.8, **kw):
    for N in [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]:
        if f(n, N, **kw) >= target:
            return N
    return '>4096'

if __name__ == '__main__':
    for lam in [0.0, 0.1, 0.2]:
        print(f'interface depolarizing lam={lam}')
        for n in range(1, 6):
            print(f'  n={n}: two-copy Bell N={ncopies(succ_bell, n, lam=lam)}, '
                  f'random-Pauli N={ncopies(succ_pauli, n, lam=lam)}, Haar-basis N={ncopies(succ_haar, n, lam=lam)}')
