import numpy as np, importlib.util, sys
spec = importlib.util.spec_from_file_location("a", "a_cluster.py")
src = open("a_cluster.py").read().split("# ---- per-copy exponents ----")[0]
exec(src)
for n_ in [9]:
    pass
# fidelity susceptibility along the line
hh = h2s
F = [abs(S[hh[i]].conj() @ S[hh[i+1]])**2 for i in range(len(hh)-1)]
chi = [(1-f)/0.02**2 for f in F]
k = int(np.argmax(chi)); print("fidelity-susceptibility peak at h2 =", (hh[k]+hh[k+1])/2, "chi=%.2f"%chi[k])
print("d2E/dh2 around min:", [(h, round(x,3)) for h, x in zip(hh, d2) if 0.0 <= h <= 0.5][::2])
