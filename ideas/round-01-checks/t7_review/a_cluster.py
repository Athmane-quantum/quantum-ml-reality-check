"""Arm A: cluster-Ising chain, OBC, n=9, sector P_odd=P_even=+1, line h1=0.5J.
(1) finite-size crossover location vs the proposal's exclusion window |h2-0.423|<0.1;
(2) Cong M_min (eq. 4) for SOPs of length 5, 7 and 9;
(3) per-copy distinguishability of two states straddling the crossover:
    - classical Chernoff information of the best fixed product Pauli basis (all 3^9 bases),
    - of the two 'sublattice' bases ZXZXZXZXZ / XZXZXZXZX (the basis in which Cong's QCNN operator is diagonal),
    - of the CZ-disentangler + X-basis POVM (all ZXZ stabilisers of both sublattices in one shot),
    - quantum bound -log|<psi|phi>|^2 (achieved by a single-copy projective measurement onto one hypothesis).
"""
import itertools, numpy as np
from functools import reduce

n = 9
I2 = np.eye(2); X = np.array([[0, 1], [1, 0]], complex); Z = np.diag([1, -1]).astype(complex)
Y = np.array([[0, -1j], [1j, 0]])


def op(ps):  # dict site->pauli
    return reduce(np.kron, [ps.get(i, I2) for i in range(n)])


ZXZ = sum(op({i: Z, i + 1: X, i + 2: Z}) for i in range(n - 2))
Xs = sum(op({i: X}) for i in range(n))
XX = sum(op({i: X, i + 1: X}) for i in range(n - 1))
Pe = op({i: X for i in range(1, n, 2)}); Po = op({i: X for i in range(0, n, 2)})


def gs(h1, h2, J=1.0):
    H = -J * ZXZ - h1 * Xs - h2 * XX - 20 * (Pe + Po)
    w, v = np.linalg.eigh(H)
    return w[0] + 40, v[:, 0]


h2s = np.round(np.arange(-0.6, 1.0001, 0.02), 3)
E = []; S = {}
sop = lambda L, a: op({a: Z, **{a + k: X for k in range(1, L - 1, 2)}, a + L - 1: Z})
SOP = {L: sop(L, (n - L) // 2) for L in (5, 7, 9)}
for h2 in h2s:
    e, v = gs(0.5, h2); E.append(e); S[h2] = v
E = np.array(E)
d2 = np.gradient(np.gradient(E, h2s), h2s)
print("n=9 crossover from d2E/dh2 extremum on h1=0.5 line: h2 =", h2s[np.argmin(d2)], "(iDMRG value 0.423)")
print(" h2    <S5>    <S7>    <S9>   Mmin(S5) Mmin(S7) Mmin(S9)")
mmin = lambda e: 1.96 ** 2 / (np.arcsin(np.sqrt((1 + e) / 2)) - np.arcsin(np.sqrt(0.5))) ** 2
for h2 in [-0.4, -0.2, 0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
    v = S[h2]
    ev = [float(np.real(v.conj() @ SOP[L] @ v)) for L in (5, 7, 9)]
    print(f"{h2:5.2f} " + " ".join(f"{e:7.3f}" for e in ev) + "  " + " ".join(f"{mmin(e):8.1f}" for e in ev))

# ---- per-copy exponents ----
Hd = np.array([[1, 1], [1, -1]]) / np.sqrt(2); Sd = np.array([[1, 0], [0, -1j]])
R = {'Z': I2, 'X': Hd, 'Y': Hd @ Sd}


def apply1(psi, U, q):
    t = psi.reshape([2] * n)
    t = np.moveaxis(np.tensordot(U, t, axes=([1], [q])), 0, q)
    return t.reshape(-1)


def probs(psi, basis):
    for q, b in enumerate(basis):
        if b != 'Z':
            psi = apply1(psi, R[b], q)
    return np.abs(psi) ** 2


sgrid = np.linspace(0.01, 0.99, 99)


def chernoff(p, q):
    return -min(np.log(np.sum(p ** s * q ** (1 - s))) for s in sgrid)


CZ = np.ones(2 ** n)
bits = (np.arange(2 ** n)[:, None] >> (n - 1 - np.arange(n))) & 1
for i in range(n - 1):
    CZ *= (-1) ** (bits[:, i] * bits[:, i + 1])


def czx(psi):
    psi = CZ * psi
    for q in range(n):
        psi = apply1(psi, Hd, q)
    return np.abs(psi) ** 2


def exponents(a, b, full=True):
    psi, phi = S[a], S[b]
    out = {}
    out['quantum(-logF)'] = -np.log(abs(psi.conj() @ phi) ** 2)
    out['CZ+X (both sublattices)'] = chernoff(czx(psi), czx(phi))
    out['ZXZXZXZXZ'] = chernoff(probs(psi, 'ZXZXZXZXZ'), probs(phi, 'ZXZXZXZXZ'))
    out['XZXZXZXZX'] = chernoff(probs(psi, 'XZXZXZXZX'), probs(phi, 'XZXZXZXZX'))
    out['all-X'] = chernoff(probs(psi, 'X' * n), probs(phi, 'X' * n))
    if full:
        best = (0, None)
        for basis in itertools.product('XYZ', repeat=n):
            p, q = probs(psi, basis), probs(phi, basis)
            # quick screen with s=1/2 then refine
            c = -np.log(np.sum(np.sqrt(p * q)))
            if c > best[0] * 0.9:
                c = chernoff(p, q)
                if c > best[0]:
                    best = (c, ''.join(basis))
        out['best product basis'] = best
    return out


hc = 0.42
for (a, b) in [(0.2, 0.6), (0.3, 0.5), (0.36, 0.48), (0.0, 0.8)]:
    a, b = round(a, 2), round(b, 2)
    ex = exponents(a, b)
    bp = ex.pop('best product basis')
    print(f"\npair h2=({a},{b}):  best product basis {bp[1]} C={bp[0]:.4f}")
    for k, v in ex.items():
        print(f"   {k:26s} {v:.4f}   ratio to best product basis {v / bp[0]:.2f}")
