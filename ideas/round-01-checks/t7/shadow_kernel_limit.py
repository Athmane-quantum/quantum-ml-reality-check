# Check: infinite-T limit of the Huang et al. (2106.12627, eq. 6) shadow kernel inner sum
#   E_{s~rho, s'~rho'} exp((gamma/n) sum_i tr(sigma_i sigma'_i)),  sigma = 3|s><s| - I
# equals sum_A a^{n-|A|} b^{|A|} tr(rho_A rho'_A)  with  F = a I + b SWAP per qubit.
import numpy as np, itertools
rng = np.random.default_rng(0)
I2 = np.eye(2); X = np.array([[0,1],[1,0]]); Y = np.array([[0,-1j],[1j,0]]); Z = np.diag([1.,-1.])
# six Pauli eigenstates
kets = []
for P in (X, Y, Z):
    w, v = np.linalg.eigh(P)
    for k in range(2): kets.append(v[:, k])
proj = [np.outer(k, k.conj()) for k in kets]
def F_local(c):
    F = np.zeros((4,4), complex)
    for p in proj:
        for q in proj:
            sig, sigp = 3*p - I2, 3*q - I2
            f = np.exp(c*np.trace(sig@sigp).real)
            F += (1/9)*f*np.kron(p, q)
    return F
SW = np.zeros((4,4)); 
for i in range(2):
    for j in range(2): SW[2*i+j, 2*j+i] = 1
def ab(c):
    F = F_local(c)
    t1, t2 = np.trace(F).real, np.trace(F@SW).real
    # 4a+2b=t1, 2a+4b=t2
    a, b = np.linalg.solve([[4,2],[2,4]], [t1, t2])
    resid = np.linalg.norm(F - (a*np.eye(4) + b*SW))
    return a, b, resid
n = 3
def rand_rho(n):
    G = rng.normal(size=(2**n, 2**n)) + 1j*rng.normal(size=(2**n, 2**n))
    r = G@G.conj().T; return r/np.trace(r)
rho, rhop = rand_rho(n), rand_rho(n)
def ptrace_keep(r, keep, n):
    r = r.reshape([2]*2*n)
    idx = list(range(n)); 
    # trace out qubits not in keep
    out = r
    cur = n
    for q in sorted(set(range(n)) - set(keep), reverse=True):
        out = np.trace(out, axis1=q, axis2=q+cur); cur -= 1
    d = 2**len(keep); return out.reshape(d, d)
for gamma in (0.5, 1.0, 3.0, 10.0):
    c = gamma/n
    a, b, res = ab(c)
    # exact expectation over snapshot pairs (6^n x 6^n)
    E = 0
    for s in itertools.product(range(6), repeat=n):
        Ps = np.ones(1)
        ket = np.array([1.0+0j])
        for i in s: ket = np.kron(ket, kets[i])
        ps = (1/3**n)*np.real(ket.conj()@rho@ket)
        for sp in itertools.product(range(6), repeat=n):
            ketp = np.array([1.0+0j])
            for i in sp: ketp = np.kron(ketp, kets[i])
            psp = (1/3**n)*np.real(ketp.conj()@rhop@ketp)
            expo = sum(np.trace((3*proj[i]-I2)@(3*proj[j]-I2)).real for i, j in zip(s, sp))
            E += ps*psp*np.exp(c*expo)
    S = 0
    for k in range(n+1):
        for A in itertools.combinations(range(n), k):
            if k == 0: ov = 1.0
            else: ov = np.trace(ptrace_keep(rho, A, n)@ptrace_keep(rhop, A, n)).real
            S += a**(n-k) * b**k * ov
    print(f"gamma={gamma}: a={a:.4f} b={b:.4f} resid={res:.1e}  exact={E:.6f}  formula={S:.6f}  weight of global overlap b^n/a^n={ (b/a)**n:.3e}")
