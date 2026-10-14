import numpy as np, exact
from exact import bell, mub, wdist
# continuous N* by bisection on even N (Bell) / integer N (MUB), to locate lambda*(n) vs MUB-cycling
def ncont(f, n, lam, hi=2**18, even=False):
    lo = 1
    if f(n, hi, lam=lam) < 0.8: return np.inf
    while hi - lo > 1:
        mid = (lo+hi)//2
        if even: mid += mid % 2
        if mid >= hi: break
        if f(n, mid, lam=lam) >= 0.8: hi = mid
        else: lo = mid
    return hi
for n in range(1, 6):
    row = []
    for lam in [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        b = ncont(bell, n, lam, even=True); m = ncont(mub, n, lam)
        row.append(f'{lam}:{b}/{m}')
    print('n', n, ' lam: Bell/MUB  ', '  '.join(row))
