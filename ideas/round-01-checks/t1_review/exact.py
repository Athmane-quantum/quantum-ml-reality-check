"""Adversarial check of the T1 best version (Theorem-6 task, Def. 2 of 2112.00778).
Exact (closed-form) Bayes success for: two-copy Bell sampling; MUB-cycling stabilizer bases (each non-identity
Pauli covered exactly floor/ceil(N/(d+1)) times); random global Clifford (coverage ~ Binomial(N, 1/(d+1)));
random product-Pauli bases (coverage ~ Binomial(N, 3^-w)). Interface depolarizing lam: a -> a (1-lam)^w.
Exact weight distribution P(w) = C(n,w) 3^w / (4^n - 1).  Also the explicit lower bound (D38) for ANY adaptive
single-copy learner, and its noisy analogue via Lemma F.4 of 2512.10929 (sum_P (1-lam)^{2|P|} <phi|P|phi>^2 <= (2 f)^n)."""
import numpy as np
from math import comb, log
GRID = [2,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048,3072,4096]
A0 = 0.9

def wdist(n):
    w = np.arange(1, n+1); p = np.array([comb(n,k)*3**k for k in w], float); return w, p/p.sum()

def succ_signs(k, b):
    """Bayes success, prior 1/2: H0 k fair signs vs H1 k signs with bias b toward unknown s=+-1."""
    if k == 0: return 0.5
    j = np.arange(k+1); c = np.array([comb(k,int(i)) for i in j], float)
    p0 = c * 0.5**k
    p1 = c * 0.5*(((1+b)/2)**j*((1-b)/2)**(k-j) + ((1-b)/2)**j*((1+b)/2)**(k-j))
    return 0.5*np.maximum(p0, p1).sum()

_cache = {}
def S(k, b):
    key = (k, round(b, 12))
    if key not in _cache: _cache[key] = succ_signs(k, b)
    return _cache[key]

def binom_pmf(N, p):
    from scipy.stats import binom
    k = np.arange(N+1); return k, binom.pmf(k, N, p)

def succ_known(k, b):
    """H0: k fair signs; H1: k signs with KNOWN bias b toward +1 (Bell eta after the known transpose sign)."""
    if k == 0: return 0.5
    j = np.arange(k+1); c = np.array([comb(k,int(i)) for i in j], float)
    p0 = c*0.5**k; p1 = c*((1+b)/2)**j*((1-b)/2)**(k-j)
    return 0.5*np.maximum(p0, p1).sum()

def bell(n, N, lam=0.0, mu=0.0):
    T = N//2; w, pw = wdist(n)
    return sum(p*succ_known(T, A0**2*(1-lam)**(2*wi)*(1-mu)**wi) for wi, p in zip(w, pw))

def mub(n, N, lam=0.0):
    d = 2**n; G = d+1; q, r = divmod(N, G); w, pw = wdist(n)
    out = 0.0
    for wi, p in zip(w, pw):
        b = A0*(1-lam)**wi
        out += p*((r*S(q+1, b) + (G-r)*S(q, b))/G)
    return out

def clifford(n, N, lam=0.0):
    d = 2**n; k, pk = binom_pmf(N, 1/(d+1)); w, pw = wdist(n)
    out = 0.0
    for wi, p in zip(w, pw):
        b = A0*(1-lam)**wi
        out += p*sum(pk[i]*S(int(k[i]), b) for i in range(len(k)) if pk[i] > 1e-12)
    return out

def pauli(n, N, lam=0.0):
    w, pw = wdist(n); out = 0.0
    for wi, p in zip(w, pw):
        k, pk = binom_pmf(N, 3.0**(-wi)); b = A0*(1-lam)**wi
        out += p*sum(pk[i]*S(int(k[i]), b) for i in range(len(k)) if pk[i] > 1e-12)
    return out

def nstar(f, n, **kw):
    for N in GRID:
        if f(n, N, **kw) >= 0.8: return N
    return 99999

def lb_any_single_copy(n, lam=0.0, delta=0.2):
    """D38 of 2112.00778 (lam=0): T >= (2^n+1)/0.8505 ln(1/(2 delta)).  Noisy analogue: replace
    E_P <phi|P|phi>^2 = 1/(2^n+1) by E_P (1-lam)^{2|P|}<phi|P|phi>^2 <= ((2f)^n - 1)/(4^n - 1), f = 1-lam+lam^2/2,
    and 0.81 by a^2 (log(1-x) >= -2.1x still valid since a^2 (1-lam)^{2w} m^2 <= 0.81)."""
    d = 2**n; f = 1 - lam + lam**2/2
    m2 = ((2*f)**n - 1)/(4**n - 1)
    return log(1/(2*delta))/(0.5*2.1*A0**2*m2)

if __name__ == '__main__':
    print('exact Bell success at N=4,6:', [round(bell(5, N), 4) for N in (4, 6)])
    print('noiseless N*(0.8): n | Bell | MUB-cycling | random Clifford | random Pauli | LB any adaptive single-copy (D38)')
    for n in range(1, 6):
        print(n, nstar(bell, n), nstar(mub, n), nstar(clifford, n), nstar(pauli, n), round(lb_any_single_copy(n), 1))
    for lam in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        print(f'lam={lam}: n | Bell | MUB | Clifford | LB-any-single-copy(noisy)')
        for n in range(1, 6):
            print('  ', n, nstar(bell, n, lam=lam), nstar(mub, n, lam=lam), nstar(clifford, n, lam=lam), round(lb_any_single_copy(n, lam), 1))
