# Pilot (scratch only): are symmetry-scrambled cluster-Ising ground states (n=10) still
# "locally easy", and does the two-copy Z2xZ2 invariant still separate SPT from trivial?
import numpy as np, itertools, sys
from functools import reduce
rng = np.random.default_rng(1)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
I2 = np.eye(2); X = np.array([[0.,1],[1,0]]); Z = np.diag([1.,-1]); Y = np.array([[0,-1j],[1j,0]])
Hd = np.array([[1,1],[1,-1]])/np.sqrt(2)
def op(single, sites):
    mats = [I2]*n
    for s, m in zip(sites, single): mats[s] = m
    return reduce(np.kron, mats)
def ham(h1, h2, J=1.0):
    H = np.zeros((2**n, 2**n))
    for i in range(n-2): H -= J*op([Z,X,Z],[i,i+1,i+2])
    for i in range(n): H -= h1*op([X],[i])
    for i in range(n-1): H -= h2*op([X,X],[i,i+1])
    return H
Podd = op([X]*len(range(0,n,2)), list(range(0,n,2)))   # sites 0,2,4.. ("odd" in 1-based)
Peven = op([X]*len(range(1,n,2)), list(range(1,n,2)))
Ppp = (np.eye(2**n)+Podd)@(np.eye(2**n)+Peven)/4
def ground(h1, h2):
    H = ham(h1, h2) + 50*(np.eye(2**n)-Ppp)
    w, v = np.linalg.eigh(H); return v[:,0]
def sym_gate4():
    # 4 qubits (a,b,c,d) = (odd,even,odd,even); commute with X_a X_c and X_b X_d.
    # In X basis: block diagonal over parities (xa^xc, xb^xd).
    U = np.zeros((16,16), complex)
    sectors = {}
    for idx in range(16):
        bits = [(idx>>(3-k))&1 for k in range(4)]
        key = (bits[0]^bits[2], bits[1]^bits[3]); sectors.setdefault(key, []).append(idx)
    for key, ids in sectors.items():
        G = rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)); q, r = np.linalg.qr(G)
        q = q*(np.diag(r)/abs(np.diag(r)))
        U[np.ix_(ids, ids)] = q
    H4 = reduce(np.kron, [Hd]*4)
    return H4@U@H4
def apply_gate(psi, U, sites):
    psi = psi.reshape([2]*n); k = len(sites)
    psi = np.moveaxis(psi, sites, list(range(k))).reshape(2**k, -1)
    psi = (U@psi).reshape([2]*k + [2]*(n-k))
    return np.moveaxis(psi, list(range(k)), sites).reshape(-1)
def scramble(psi, layers=1):
    for l in range(layers):
        off = rng.integers(0, 2)*2   # offset 0 or 2 keeps (odd,even,odd,even) pattern
        for s in range(off, n-3, 4):
            psi = apply_gate(psi, sym_gate4(), [s, s+1, s+2, s+3])
    return psi
def rdm(psi, keep):
    psi = psi.reshape([2]*n); rest = [i for i in range(n) if i not in keep]
    m = np.moveaxis(psi, keep+rest, list(range(n))).reshape(2**len(keep), -1)
    return m@m.conj().T
def local_feats(psi):
    f = []
    P = [X, Y, Z]
    for i in range(n):
        r = rdm(psi, [i]); f += [np.trace(r@p).real for p in P]
    for i in range(n-1):
        for j in (i+1, i+2):
            if j >= n: continue
            r = rdm(psi, [i, j])
            f += [np.trace(r@np.kron(p,q)).real for p in P for q in P]
    for i in range(n-2):
        r = rdm(psi, [i,i+1,i+2]); f += [np.trace(r@reduce(np.kron,[p,q,s])).real for p in P for q in P for s in P]
    return np.array(f)
def sop(psi, a, b):
    sites = [a] + list(range(a+1, b, 2)) + [b]
    mats = [Z] + [X]*(len(sites)-2) + [Z]
    v = op(mats, sites)@psi; return (psi.conj()@v).real
def mbti(psi, I1, J2, I3, g_on, h_on):
    # Z = Tr_{I1 u I3}[ Tr_I2[(G_h) (G_g rho G_g)] Tr_I2[(G_h) rho] ]  (Pollmann-Turner / Elben B1 form)
    I = I1+J2+I3; r = rdm(psi, I); k1, k2, k3 = len(I1), len(J2), len(I3)
    def string(sites_sub, parity):  # X on sites with given parity, within the listed sublattice
        return reduce(np.kron, [X if (s % 2)==parity else I2 for s in sites_sub])
    Gg = np.kron(np.kron(string(I1, g_on), np.eye(2**k2)), np.eye(2**k3))
    Gh = np.kron(np.kron(np.eye(2**k1), string(J2, h_on)), np.eye(2**k3))
    def red(M):
        M = (Gh@M).reshape([2]*(2*len(I))); L = len(I)
        # trace over I2 positions k1..k1+k2-1
        for t in range(k2):
            ax = k1  # after each trace the indices shift
            M = np.trace(M, axis1=ax, axis2=ax+M.ndim//2)
        d = 2**(k1+k3); return M.reshape(d, d)
    rb = Gg@r@Gg
    A, B = red(rb), red(r)
    z = np.trace(A@B).real
    pur = np.trace(red(np.eye(2**(k1+k2+k3)) @ r) @ red(r)).real if False else None
    p13 = rdm(psi, I1+I3); norm = np.trace(p13@p13).real
    return z, z/norm
pts_spt = [(h1, h2) for h1 in np.linspace(0.0, 0.5, 4) for h2 in np.linspace(-0.2, 0.2, 3)]
pts_tri = [(h1, h2) for h1 in np.linspace(1.5, 2.5, 4) for h2 in np.linspace(-0.2, 0.2, 3)]
data = []
for lab, pts in ((1, pts_spt), (0, pts_tri)):
    for (h1, h2) in pts:
        psi0 = ground(h1, h2)
        for rep in range(3):
            psi = scramble(psi0.copy(), layers=1) if rep > 0 else psi0
            data.append((lab, rep > 0, psi))
# 1) local-easiness on scrambled states: leave-one-out nearest-centroid / logistic on exact local feats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
for scr in (False, True):
    sub = [(l, p) for (l, s, p) in data if s == scr]
    Xf = np.array([local_feats(p) for l, p in sub]); y = np.array([l for l, p in sub])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    acc = cross_val_score(clf, Xf, y, cv=4).mean()
    # strongest single local feature
    best = max(abs(np.corrcoef(Xf[:,j], y)[0,1]) if Xf[:,j].std()>1e-9 else 0 for j in range(Xf.shape[1]))
    print(f"scrambled={scr}: N={len(y)}  CV acc (exact <=3-local Pauli feats, logistic) = {acc:.3f};  max |corr| single feature = {best:.3f}")
# 2) SOP and MBTI
mid = n//2
for scr in (False, True):
    sub = [(l, p) for (l, s, p) in data if s == scr]
    s_spt = [sop(p, 1, n-3) for l, p in sub if l==1]; s_tri = [sop(p, 1, n-3) for l, p in sub if l==0]
    print(f"scrambled={scr}: SOP(1,{n-3}) SPT mean {np.mean(s_spt):+.3f} [{np.min(s_spt):+.3f},{np.max(s_spt):+.3f}]  trivial mean {np.mean(s_tri):+.3f} [{np.min(s_tri):+.3f},{np.max(s_tri):+.3f}]")
    for (I1, J2, I3) in (([1,2],[3,4,5,6],[7,8]), ([0,1],[2,3,4,5,6,7],[8,9]), ([0,1,2],[3,4,5,6],[7,8,9])):
        if max(I3) >= n: continue
        for g_on, h_on in ((0,1),(1,0)):
            v1 = [mbti(p, I1, J2, I3, g_on, h_on)[1] for l, p in sub if l==1]
            v0 = [mbti(p, I1, J2, I3, g_on, h_on)[1] for l, p in sub if l==0]
            sep = (min(v1) > max(v0)) or (max(v1) < min(v0))
            print(f"   I1={I1} I2={J2} I3={I3} g_par={g_on} h_par={h_on}: SPT [{min(v1):+.3f},{max(v1):+.3f}]  trivial [{min(v0):+.3f},{max(v0):+.3f}]  separable={sep}")
