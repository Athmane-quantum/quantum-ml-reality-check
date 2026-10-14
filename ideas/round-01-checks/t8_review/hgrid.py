import numpy as np
from scipy.stats import binom
def success(m, eps0=1/3):
    if m <= 0: return 0.5
    k=np.arange(m+1); p0=binom.pmf(k,m,0.5)
    p1=0.5*(binom.pmf(k,m,(1+eps0)/2)+binom.pmf(k,m,(1-eps0)/2))
    return 0.5*np.maximum(p0,p1).sum()
grid = np.logspace(2, 6, 25)
print("grid step factor", grid[1]/grid[0])
def nstar(frac, grid):
    for N in grid:
        if success(int(N*frac)) >= 0.9: return N
    return np.nan
fine = np.unique(np.logspace(0, 6, 3000).astype(int))
for n in range(1,6):
    ea_g, ef_g = nstar(1.0, grid), nstar(1/(2**n+1), grid)
    ea_f, ef_f = nstar(1.0, fine), nstar(1/(2**n+1), fine)
    Rg = ef_g/ea_g
    print(f"n={n}: true N*_EA={ea_f} N*_EF={ef_f} R_true={ef_f/ea_f:.2f} | on proposal grid: N*_EA={ea_g:.0f} (censored at grid min) N*_EF={ef_g:.0f} R_grid={Rg:.2f} vs 2^n+1={2**n+1}: dev={100*(Rg/(2**n+1)-1):+.0f}%  gate(<=15%) {'PASS' if abs(Rg/(2**n+1)-1)<=0.15 else 'FAIL'}")
