"""Pilot for T3 best version: well-specified vs misspecified generative models on IQP targets.

Scratch only (not part of the repository). Exact probabilities at n <= 10.
Models: add-alpha histogram (alpha tuned on validation), logistic autoregressive chain (C tuned),
MPS Born machine (chi grid), well-specified IQP Born machine (exact NLL), generic hardware-efficient
QCBM (exact NLL, depth grid). Metric: forward KL(p || q).
"""
import itertools, sys, time
import numpy as np
import jax, jax.numpy as jnp
from sklearn.linear_model import LogisticRegression

jax.config.update("jax_enable_x64", True)

def bits(n):
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int64)  # index order: x0 is MSB

def wht(a, n):
    a = a.reshape((2,) * n)
    for ax in range(n):
        a0 = jnp.take(a, 0, axis=ax); a1 = jnp.take(a, 1, axis=ax)
        a = jnp.stack([(a0 + a1), (a0 - a1)], axis=ax) / jnp.sqrt(2.0)
    return a.reshape(-1)

def iqp_probs(params, n, F):
    # F: precomputed float feature matrix [Z_i, Z_i Z_j] (avoids int gathers under jit)
    phase = F @ params
    amp = jnp.exp(1j * phase) / 2 ** (n / 2)
    p = jnp.abs(wht(amp, n)) ** 2
    return p / p.sum()

def hea_probs(params, n, L):
    # params shape (L, n, 2): RY then RZ on each qubit, then CZ ring
    psi = jnp.zeros((2,) * n, dtype=jnp.complex128).at[(0,) * n].set(1.0)
    B = bits(n); czsign = np.ones(2 ** n)
    for i in range(n):
        j = (i + 1) % n
        if n > 2 or i == 0:
            czsign *= np.where((B[:, i] == 1) & (B[:, j] == 1), -1.0, 1.0)
    czsign = jnp.asarray(czsign.reshape((2,) * n))
    for l in range(L):
        for q in range(n):
            a, b = params[l, q, 0], params[l, q, 1]
            ry = jnp.array([[jnp.cos(a / 2), -jnp.sin(a / 2)], [jnp.sin(a / 2), jnp.cos(a / 2)]], dtype=jnp.complex128)
            rz = jnp.array([[jnp.exp(-1j * b / 2), 0], [0, jnp.exp(1j * b / 2)]], dtype=jnp.complex128)
            U = rz @ ry
            psi = jnp.moveaxis(jnp.tensordot(U, psi, axes=([1], [q])), 0, q)
        psi = psi * czsign
    p = jnp.abs(psi.reshape(-1)) ** 2
    return p / p.sum()

def mps_probs(params, n, chi, B):
    # params: list-free packing: tensors A_k[x] of shape (chi,chi), boundary vectors
    A = params.reshape(n, 2, chi, chi)
    left = jnp.ones((chi,)) / jnp.sqrt(chi)
    # amplitude for all strings via sequential contraction over basis (2^n small)
    vecs = jnp.tile(left, (2 ** n, 1))
    Bf = jnp.asarray(B, dtype=jnp.float64)
    for k in range(n):
        Ak = (1 - Bf[:, k])[:, None, None] * A[k, 0][None] + Bf[:, k][:, None, None] * A[k, 1][None]
        vecs = jnp.einsum('bi,bij->bj', vecs, Ak)
    amp = vecs.sum(axis=1)
    p = amp ** 2
    return p / p.sum()

def kl(p, q):
    q = np.clip(q, 1e-300, None)
    m = p > 0
    return float(np.sum(p[m] * (np.log(p[m]) - np.log(q[m]))))

def adam_fit(loss, x0, steps, lr, val_loss=None):
    m = jnp.zeros_like(x0); v = jnp.zeros_like(x0); x = x0
    g = jax.jit(jax.value_and_grad(loss))
    vl = jax.jit(val_loss) if val_loss is not None else None
    best = (np.inf, x)
    for t in range(1, steps + 1):
        _, gr = g(x)
        m = 0.9 * m + 0.1 * gr; v = 0.999 * v + 0.001 * gr ** 2
        x = x - lr * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        if vl is not None and (t % 25 == 0 or t == steps):
            val = float(vl(x))
            if val < best[0]:
                best = (val, x)
    return best

def counts(idx, n):
    return np.bincount(idx, minlength=2 ** n).astype(float)

def run(n, N, seed):
    rng = np.random.default_rng(seed)
    B = bits(n); Z = 1 - 2 * B
    pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
    true = rng.uniform(0, 2 * np.pi, n + len(pairs))
    F = jnp.asarray(np.column_stack([Z, Z[:, pairs[:, 0]] * Z[:, pairs[:, 1]]]).astype(np.float64))
    p = np.asarray(iqp_probs(jnp.asarray(true), n, F))
    idx = rng.choice(2 ** n, size=N, p=p)
    ntr = int(0.8 * N)
    tr, va = idx[:ntr], idx[ntr:]
    ctr, cva = counts(tr, n), counts(va, n)
    call = counts(idx, n)
    out = {}
    # 1. histogram, alpha tuned on validation, refit on all
    best = min((-(cva * np.log((ctr + a) / (ctr.sum() + a * 2 ** n))).sum(), a) for a in [0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    a = best[1]; out['hist'] = kl(p, (call + a) / (call.sum() + a * 2 ** n))
    # 2. logistic autoregressive chain with pairwise products of previous bits (C tuned on val)
    def chain_probs(train_idx, C):
        X = B[train_idx]; logq = np.zeros(2 ** n)
        for k in range(n):
            def feats(M):
                prev = M[:, :k]
                if k == 0:
                    return np.zeros((len(M), 1))
                pr = [prev[:, i] * prev[:, j] for i in range(k) for j in range(i + 1, k)]
                return np.column_stack([prev] + pr) if pr else prev
            y = X[:, k]
            if y.min() == y.max():
                pk1 = np.full(2 ** n, (y.mean() * len(y) + 0.5) / (len(y) + 1))
            else:
                clf = LogisticRegression(C=C, max_iter=2000).fit(feats(X), y)
                pk1 = clf.predict_proba(feats(B))[:, 1]
            logq += np.where(B[:, k] == 1, np.log(pk1), np.log(1 - pk1))
        q = np.exp(logq); return q / q.sum()
    Cs = [0.01, 0.1, 1, 10]
    bestC = min((-(cva * np.log(chain_probs(tr, C) + 1e-300)).sum(), C) for C in Cs)[1]
    out['ar_logistic'] = kl(p, chain_probs(idx, bestC))
    # helper for exact-NLL models: tune on val, report best-val params
    def nll_of(probf, c):
        return lambda th: -(jnp.asarray(c) * jnp.log(probf(th) + 1e-300)).sum() / c.sum()
    # 3. well-specified IQP Born machine (several restarts, pick best val)
    f = lambda th: iqp_probs(th, n, F)
    res = []
    for r in range(12):
        x0 = jnp.asarray(rng.uniform(0, 2 * np.pi, n + len(pairs)))
        res.append(adam_fit(nll_of(f, ctr), x0, 600, 0.05, nll_of(f, cva)))
    out['iqp_wellspec'] = kl(p, np.asarray(f(min(res, key=lambda t: t[0])[1])))
    # 4. generic HEA QCBM, depth grid
    res = []
    for L in [2, 4, 6]:
        g = lambda th, L=L: hea_probs(th.reshape(L, n, 2), n, L)
        x0 = jnp.asarray(rng.normal(0, 1.0, L * n * 2))
        v, th = adam_fit(nll_of(g, ctr), x0, 600, 0.05, nll_of(g, cva))
        res.append((v, L, th))
    v, L, th = min(res, key=lambda t: t[0])
    out['hea_qcbm'] = kl(p, np.asarray(hea_probs(th.reshape(L, n, 2), n, L))); out['hea_L'] = L
    # 5. MPS Born machine, chi grid
    res = []
    for chi in [2, 4, 8, 16]:
        h = lambda th, chi=chi: mps_probs(th, n, chi, B)
        x0 = jnp.asarray(rng.normal(0, 1.0 / np.sqrt(chi), n * 2 * chi * chi))
        v, th = adam_fit(nll_of(h, ctr), x0, 600, 0.02, nll_of(h, cva))
        res.append((v, chi, th))
    v, chi, th = min(res, key=lambda t: t[0])
    out['mps'] = kl(p, np.asarray(mps_probs(th, n, chi, B))); out['mps_chi'] = chi
    out['uniform'] = kl(p, np.ones(2 ** n) / 2 ** n)
    return out

if __name__ == '__main__':
    n = int(sys.argv[1]); N = int(sys.argv[2]); seeds = int(sys.argv[3])
    for s in range(seeds):
        t0 = time.time(); o = run(n, N, s)
        print(f"n={n} N={N} seed={s} ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()}, f"{time.time()-t0:.0f}s", flush=True)
