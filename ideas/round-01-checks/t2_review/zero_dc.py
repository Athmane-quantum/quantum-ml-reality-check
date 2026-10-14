import sys, itertools, numpy as np
sys.path.insert(0, '..')
from t2_check import outcome_probs, fisher
EPS = 0.05
def char_gauss(tvecs, Sig):
    q = np.einsum("ki,ij,kj->k", tvecs, Sig, tvecs); return np.exp(-0.5*q).astype(complex)
def sig_tele(V, r, m):   # theta_j = A_j - A_{j-1}, A stationary AR(1) var V corr r  (zero-DC / high-pass signal)
    idx = np.arange(m+1); SA = V * r**np.abs(idx[:,None]-idx[None,:])
    D = np.zeros((m, m+1)); D[np.arange(m), np.arange(1,m+1)] = 1; D[np.arange(m), np.arange(m)] = -1
    return D @ SA @ D.T
def sig_ar1(s, rho, m):
    idx = np.arange(m); return s**2 * rho**np.abs(idx[:,None]-idx[None,:])
def run(sigf, p0, m):
    tvecs = np.array(list(itertools.product([-1,0,1], repeat=m)))
    ones = np.ones((1,m))
    cfun = lambda p: np.real(char_gauss(ones, sigf(p[0],p[1],m))[0])
    h=1e-6; g = np.array([(cfun([p0[0]+h,p0[1]])-cfun([p0[0]-h,p0[1]]))/(2*h), (cfun([p0[0],p0[1]+h])-cfun([p0[0],p0[1]-h]))/(2*h)])
    best = np.inf
    pats = [np.zeros(m), np.tile([0,np.pi/2],m)[:m], np.full(m,np.pi/2)]
    for phis in pats:
        f = lambda p: outcome_probs(char_gauss(tvecs, sigf(p[0],p[1],m)), tvecs, phis, m)
        I = fisher(f, list(p0))
        # also allow splitting trajectories evenly over the patterns (mixture) -> use sum of Fisher / len
        try: best = min(best, g @ np.linalg.solve(I, g)/EPS**2)
        except np.linalg.LinAlgError: pass
    Imix = sum(fisher(lambda p, ph=ph: outcome_probs(char_gauss(tvecs, sigf(p[0],p[1],m)), tvecs, ph, m), list(p0)) for ph in pats)/len(pats)
    try: best = min(best, g @ np.linalg.solve(Imix, g)/EPS**2)
    except np.linalg.LinAlgError: pass
    c = cfun(p0); nq = (1-c**2)/EPS**2
    return c, nq, best
print("zero-DC (increments of AR(1) A with var V, corr r): c_m = exp(-V(1-r^m)) stays O(1)")
for V, r in ((0.5,0.3),(1.0,0.3),(2.0,0.3),(2.0,0.0)):
    for m in (2,3,4,5,6,7):
        c,nq,nc = run(sig_tele,(V,r),m)
        print(f"  V={V} r={r} m={m} c={c:.3f} Nq={nq:7.1f} Nc={nc:12.1f} R={nc/nq:9.2f}")
print("anti-correlated AR(1), s, rho<0")
for s, rho in ((0.5,-0.8),(1.0,-0.8),(1.0,-0.95)):
    for m in (2,3,4,5,6,7):
        c,nq,nc = run(sig_ar1,(s,rho),m)
        print(f"  s={s} rho={rho} m={m} c={c:.3f} Nq={nq:7.1f} Nc={nc:12.1f} R={nc/nq:9.2f}")
