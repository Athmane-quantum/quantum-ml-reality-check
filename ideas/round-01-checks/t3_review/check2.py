import numpy as np, itertools, sys
sys.path.insert(0,'/tmp/claude-0/-home-user-quantum-ml-reality-check/3242e8f8-b193-546c-b381-3fe888e7979d/scratchpad/round01/t3_review')
from check1 import iqp_p, idqnn_p0, grid, kl, bits
rng=np.random.default_rng(7)
def hist_kl(p,N,rg):
    K=len(p); idx=rg.choice(K,size=N,p=p); ntr=int(0.8*N)
    ctr=np.bincount(idx[:ntr],minlength=K); cva=np.bincount(idx[ntr:],minlength=K); call=ctr+cva
    a=min([0.01,0.03,0.1,0.3,1,3,10],key=lambda a:-(cva*np.log((ctr+a)/(ctr.sum()+a*K))).sum())
    return kl(p,(call+a)/(call.sum()+a*K))
n=10
for fam in ['IQP n=10 (all N used)','IDQNN 2x5 (N/3 at x=0^n)']:
    for N in [1000,3000,10000]:
        v=[];u=[]
        for s in range(5):
            if fam.startswith('IQP'):
                h=rng.uniform(0,2*np.pi,n); J=np.triu(rng.uniform(0,2*np.pi,(n,n)),1); p=iqp_p(h,J,n); Neff=N; d=n+n*(n-1)//2
            else:
                p=idqnn_p0(rng.uniform(0,2*np.pi,n),grid(2,5),n); Neff=N//3; d=n
            v.append(hist_kl(p,Neff,rng)); u.append(kl(p,np.ones(2**n)/2**n))
        print(f"{fam:28s} N={N:6d}: tuned add-alpha KL={np.mean(v):.3f}  uniform KL={np.mean(u):.3f}  (K-1)/(2N_eff)={(2**n-1)/(2*Neff):.3f}  well-specified d/(2N_eff)={d/(2*Neff):.4f}")
