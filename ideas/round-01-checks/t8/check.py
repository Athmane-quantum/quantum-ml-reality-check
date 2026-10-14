import numpy as np, itertools, time
from math import comb
from scipy.stats import binom
# (1) I-26: identity vs completely depolarizing, Helstrom with/without Bell ancilla
def helstrom(r0,r1): return 0.5+0.25*np.abs(np.linalg.eigvalsh(r0-r1)).sum()
psi=np.array([1,0.]); rho=np.outer(psi,psi)
print("product probe:", helstrom(rho, np.eye(2)/2))
phi=np.array([1,0,0,1])/np.sqrt(2); PHI=np.outer(phi,phi)
print("Bell probe   :", helstrom(PHI, np.eye(4)/4))
# random product-probe search (pure inputs suffice) -> max 0.75
best=0
for _ in range(2000):
    v=np.random.randn(2)+1j*np.random.randn(2); v/=np.linalg.norm(v); r=np.outer(v,v.conj())
    best=max(best,helstrom(r,np.eye(2)/2))
print("best random product:",best)
# N uses: error 2^-N/2 (product) vs 4^-N/2 (Bell)
for N in [1,2,3,5]: print(N, 0.5*2.0**-N, 0.5*4.0**-N)

# (2) Le Cam revealed-a game: Lambda0 (fully depolarizing) vs Lambda_{a,s}, lambda_a = s*eps0
# EA: every Bell sample gives x=(-1)^<a,c> with mean lambda_a. EF-MUB: fraction 1/(2^n+1) of samples informative.
# EF-Pauli (random Pauli bases): fraction 3^-|a| informative.
# Bayes-optimal test with a revealed: H0 x~Rademacher(0); H1: mixture over s of Rademacher(+-eps0).
# sufficient stat: k = #(+1) among m informative samples.
def success(m, eps0):
    k=np.arange(m+1)
    p0=binom.pmf(k,m,0.5)
    p1=0.5*(binom.pmf(k,m,(1+eps0)/2)+binom.pmf(k,m,(1-eps0)/2))
    return 0.5*np.maximum(p0,p1).sum()
eps0=1/3
def Nstar(frac_fn, target=0.9, n=1):
    for N in np.unique(np.logspace(0,7,400).astype(int)):
        s=frac_fn(N)
        if s>=target: return N
    return None
t=time.time()
for n in range(1,6):
    # EA
    fa=lambda N: success(N,eps0)
    # EF-MUB, deterministic cycling: informative m = N/(2^n+1)
    fm=lambda N: success(int(N//(2**n+1)),eps0)
    # EF-Pauli: average over weight distribution of uniformly random a != 0, m ~ Binomial(N,3^-w); use expectation m=N*3^-w
    ws=np.arange(1,n+1); pw=np.array([comb(n,w)*3**w for w in ws],float); pw/=pw.sum()
    fp=lambda N: sum(p*success(int(N*3.0**-w),eps0) for w,p in zip(ws,pw))
    Na,Nm,Np=Nstar(fa),Nstar(fm),Nstar(fp)
    print(f"n={n}: N*(0.9) EA={Na} EF-MUB={Nm} EF-Pauli={Np}  ratios MUB/EA={Nm/Na:.1f} (2^n+1={2**n+1}) Pauli/EA={Np/Na:.1f}")
print("time",time.time()-t)
