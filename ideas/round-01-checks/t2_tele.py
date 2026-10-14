import itertools, numpy as np
from t2_check import outcome_probs, fisher, EPS

def char_tele(tvecs, a, q, m):
    # s_j Markov on {+1,-1}, flip prob q, stationary uniform start; theta_j = a s_j
    T = np.array([[1-q, q],[q, 1-q]])
    svals = np.array([1.0, -1.0])
    out = np.empty(len(tvecs), dtype=complex)
    for k, t in enumerate(tvecs):
        v = 0.5*np.exp(1j*a*t[0]*svals)
        for j in range(1, m):
            v = (v @ T) * np.exp(1j*a*t[j]*svals)
        out[k] = v.sum()
    return out

def run(m, a, q):
    tvecs = np.array(list(itertools.product([-1,0,1], repeat=m)))
    allp = np.array(list(itertools.product([1,-1], repeat=m)))
    def cfun(p):
        return np.real(char_tele(np.ones((1,m),int), p[0], p[1], m)[0])
    best = np.inf
    for phis in (np.zeros(m), np.tile([0, np.pi/2], m)[:m], np.full(m, np.pi/2)):
        f = lambda p: outcome_probs(char_tele(tvecs, p[0], p[1], m), tvecs, phis, m)
        I = fisher(f, [a, q])
        h = 1e-6
        g = np.array([(cfun([a+h,q])-cfun([a-h,q]))/(2*h), (cfun([a,q+h])-cfun([a,q-h]))/(2*h)])
        try: nc = g @ np.linalg.solve(I, g)/EPS**2
        except np.linalg.LinAlgError: nc = np.inf
        best = min(best, nc)
    c = cfun([a,q]); nq = (1-c**2)/EPS**2
    return c, nq, best

for a, q in ((0.4, 0.2), (1.0, 0.2), (2.0, 0.1)):
    for m in (2,3,4,5,6,7):
        c, nq, nc = run(m, a, q)
        print(f"tele a={a} q={q} m={m} c={c:.3f} Nq={nq:8.1f} Nc={nc:10.1f} ratio={nc/nq:8.2f}")
