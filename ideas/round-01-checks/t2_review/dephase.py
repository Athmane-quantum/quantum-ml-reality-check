import numpy as np
# F_hard Fisher ratio with SYMMETRIC per-window dephasing (contrast e^{-g} per window on both sides)
def I_mem(C, m, g):
    best = 0
    for ph in np.linspace(0, np.pi, 181):  # memory readout basis phase
        E = np.exp(-g*m)*np.cos(C-ph); dE = -np.exp(-g*m)*np.sin(C-ph)
        best = max(best, dE**2/(1-E**2))
    return best
def I_less(C, m, g, asym=False):
    k = 1.0 if asym else np.exp(-g*m)
    best = 0
    for ph in np.linspace(0, np.pi, 181):
        A = 2.0**(1-m)*k
        E = A*np.cos(C-ph); dE = -A*np.sin(C-ph)
        best = max(best, dE**2/(1-E**2))
    return best
for m in (4, 6, 8):
    for C in (0.0, 0.3):
        r0 = I_mem(C,m,0)/I_less(C,m,0)
        r1 = I_mem(C,m,0.05)/I_less(C,m,0.05)
        r2 = I_mem(C,m,0.05)/I_less(C,m,0.05,asym=True)
        print(f"m={m} C={C}: R(g=0)={r0:9.1f}  R(g=.05,symmetric)={r1:9.1f}  R(g=.05, classical noiseless)={r2:9.1f}  4^m e^-2gm={4**m*np.exp(-0.1*m):9.1f}")
