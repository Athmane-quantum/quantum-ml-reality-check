"""Adversarial check of arm B (3x3 rotated surface code in a Z field vs 1-design product states).

H = -sum A_s - sum B_p - h sum Z_i  (2606.28199 eqs. 8-9: A_s = prod X, B_p = prod Z, field along Z).
Layout: qubits 0..8 row-major on a 3x3 grid (standard rotated d=3 code).
"""
import itertools, numpy as np
from functools import reduce

n = 9
X_STABS = [(0, 1, 3, 4), (4, 5, 7, 8), (1, 2), (6, 7)]
Z_STABS = [(1, 2, 4, 5), (3, 4, 6, 7), (0, 3), (5, 8)]
I2 = np.eye(2); Xm = np.array([[0, 1], [1, 0]]); Zm = np.diag([1., -1.])


def op(pauli, sites):
    mats = [I2] * n
    for s in sites:
        mats[s] = pauli
    return reduce(np.kron, mats)


A = [op(Xm, s) for s in X_STABS]
B = [op(Zm, s) for s in Z_STABS]
Zs = [op(Zm, (i,)) for i in range(n)]
Xs = [op(Xm, (i,)) for i in range(n)]
# commutation check
for a in A:
    for b in B:
        assert np.allclose(a @ b, b @ a)
HSC = -sum(A) - sum(B)
ZL = op(Zm, (0, 1, 2))  # logical Z: a Z string commuting with all stabs?
for a in A:
    if not np.allclose(a @ ZL, ZL @ a):
        ZL = None; break
if ZL is None:
    for cand in [(0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 1, 2), (3, 4, 5), (6, 7, 8)]:
        M = op(Zm, cand)
        if all(np.allclose(a @ M, M @ a) for a in A):
            ZL = M; zl_sites = cand; break
XL = None
for cand in [(0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 1, 2), (3, 4, 5), (6, 7, 8)]:
    M = op(Xm, cand)
    if all(np.allclose(b @ M, M @ b) for b in B) and not np.allclose(M @ ZL, ZL @ M):
        XL = M; break

hs = np.round(np.arange(0, 0.401, 0.01), 3)
E0 = []; states = {}
for h in hs:
    H = HSC - h * sum(Zs)
    w, v = np.linalg.eigh(H)
    g = v[:, 0]
    if h == 0:  # degenerate code space: take the h->0+ limit (Z_L=+1 state favoured by the Z field)
        Hs = HSC - 1e-4 * sum(Zs); w2, v2 = np.linalg.eigh(Hs); g = v2[:, 0]
    E0.append(w[0]); states[h] = g
E0 = np.array(E0)
d2 = np.gradient(np.gradient(E0, hs), hs)
print("gap at h=0 (degeneracy check): first eigenvalues", np.round(np.linalg.eigvalsh(HSC)[:3], 4))
print("finite-size h_c estimate (min of d2E/dh2):", hs[np.argmin(d2[2:-2]) + 2])
print(" h    <B_p>min  <A_s>min  <Z_i>mean  <Z_i>max  <X_i>max  <Z_L>")
for h in [0.0, 0.05, 0.1, 0.15, 0.17, 0.2, 0.25]:
    g = states[h]
    ev = lambda M: float(np.real(g.conj() @ M @ g))
    print(f"{h:4.2f}  {min(ev(b) for b in B):8.5f}  {min(ev(a) for a in A):8.5f}  {np.mean([ev(z) for z in Zs]):8.4f}  {max(ev(z) for z in Zs):8.4f}  {max(abs(ev(x)) for x in Xs):8.4f}  {ev(ZL):7.4f}")

np.save("sc_states.npy", np.array([states[h] for h in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]]))

# ---------- exact false-positive rate of the Z-basis stabiliser-parity test on 1-design product states ----------
# qubit types: 0 = Z-eigenstate (bit fixed, value uniform), 1 = X or Y eigenstate (fresh uniform bit each Z-basis shot)
def gf2_rank(rows):
    rows = [r for r in rows if r]
    rank = 0
    rows = list(rows)
    while rows:
        piv = rows.pop()
        if piv == 0:
            continue
        rank += 1
        low = piv & -piv
        rows = [r ^ piv if r & low else r for r in rows]
    return rank


def fp_Ztest(S):
    tot = 0.0
    for types in itertools.product([0, 1], repeat=n):
        p_t = np.prod([1 / 3 if t == 0 else 2 / 3 for t in types])
        free = [i for i in range(n) if types[i] == 1]
        fixed = [i for i in range(n) if types[i] == 0]
        # parity of stab = (sum fixed bits) + (sum free bits). Accept iff all parities even on every shot.
        rows = [sum(1 << free.index(q) for q in st if q in free) for st in Z_STABS]
        # average over fixed bit values (uniform): constraints on fixed bits for stabs with no free qubits
        # P(accept) = E_fixed[ prod_shots P(all parity even | fixed) ]; per shot P = 2^-rank(rows) if the affine system
        # is consistent, else 0. Consistency depends on fixed bits: enumerate fixed bits.
        r = gf2_rank(rows)
        acc = 0.0
        for bits in itertools.product([0, 1], repeat=len(fixed)):
            val = dict(zip(fixed, bits))
            # affine system: rows_k . f = c_k where c_k = parity of fixed bits in stab k
            aug = []
            for st, row in zip(Z_STABS, rows):
                c = sum(val[q] for q in st if q in val) % 2
                aug.append((row, c))
            # consistency via augmented rank
            r_aug = gf2_rank([(row << 1) | c for row, c in aug])
            consistent = (r_aug == r)
            acc += (2.0 ** (-r)) ** S if consistent else 0.0
        tot += p_t * acc / 2 ** len(fixed)
    return tot


print("\nZ-basis stabiliser test (accept iff all 4 B_p = +1 on all S shots): FN = 0 exactly for every h")
for S in [1, 2, 3, 4, 5, 6, 7, 10, 13, 20]:
    fp = fp_Ztest(S)
    print(f"S={S:2d}  FP={fp:.3e}  balanced acc={1 - fp / 2:.6f}")
