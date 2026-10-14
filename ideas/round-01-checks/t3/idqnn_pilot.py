# Pilot: IDQNN (arXiv:2509.09033, Def. 16) on a small 2D grid, inference at x = 0^n.
# Compares: paper's closed-form estimator (eq. S.2.1, sign-blind), same with oracle signs,
# and an add-alpha lookup table fitted on the x = 0^n training samples. Exact KL. numpy only.
import numpy as np, itertools, sys
sys.path.insert(0, "/tmp/claude-0/-home-user-quantum-ml-reality-check/3242e8f8-b193-546c-b381-3fe888e7979d/scratchpad/round01/t3")
from idqnn_sign import idqnn_p0
def grid(rows, cols):
    idx=lambda r,c: r*cols+c
    return [(idx(r,c),idx(r,c+1)) for r in range(rows) for c in range(cols-1)]+[(idx(r,c),idx(r+1,c)) for r in range(rows-1) for c in range(cols)]
def kl(p,q):
    m=p>0; return float(np.sum(p[m]*np.log(p[m]/np.clip(q[m],1e-300,None))))
def run(rows, cols, N, seed):
    n=rows*cols; E=grid(rows,cols); rng=np.random.default_rng(seed)
    nbr=[[j for (i,j) in E if i==q]+[i for (i,j) in E if j==q] for q in range(n)]
    th=rng.uniform(0,2*np.pi,n); p,B=idqnn_p0(th,E,n)
    comp=rng.integers(0,3,N)                       # paper's D(x): 1/3 zeros, 1/3 iid P0=.6, 1/3 iid P0=.2
    X=np.zeros((N,n),int)
    X[comp==1]=(rng.random(((comp==1).sum(),n))<0.4).astype(int)
    X[comp==2]=(rng.random(((comp==2).sum(),n))<0.8).astype(int)
    # closed-form estimator: samples with x_j=0 and all neighbours 1 -> single-qubit |+>,RZ,X-measure
    th_hat=np.zeros(n); nsp=[]
    for j in range(n):
        sel=(X[:,j]==0)&np.all(X[:,nbr[j]]==1,axis=1); k=sel.sum(); nsp.append(k)
        y=rng.random(k)<np.sin(th[j]/2)**2
        th_hat[j]=np.arccos(np.clip(1-2*y.mean(),-1,1)) if k>0 else np.pi/2
    q_blind,_=idqnn_p0(th_hat,E,n)
    th_or=np.where(np.sin(th)<0,-th_hat,th_hat)   # oracle sign: theta in (pi,2pi) == -theta' with theta' in (0,pi)
    q_or,_=idqnn_p0(th_or,E,n)
    # lookup table on x=0^n samples (they are 1/3 of N), alpha tuned on a held-out 20%
    n0=(comp==0).sum(); ys=rng.choice(2**n,size=n0,p=p); ntr=int(0.8*n0)
    ctr=np.bincount(ys[:ntr],minlength=2**n); cva=np.bincount(ys[ntr:],minlength=2**n); call=ctr+cva
    a=min([0.01,0.03,0.1,0.3,1,3,10],key=lambda a: -(cva*np.log((ctr+a)/(ctr.sum()+a*2**n))).sum())
    q_hist=(call+a)/(call.sum()+a*2**n)
    return dict(n=n,N=N,min_Nsp=int(min(nsp)),kl_estimator_signblind=kl(p,q_blind),kl_estimator_oraclesign=kl(p,q_or),kl_hist=kl(p,q_hist),kl_uniform=kl(p,np.ones(2**n)/2**n))
for (r,c) in [(2,3),(2,4),(2,5)]:
    for N in [1000,3000,30000]:
        res=[run(r,c,N,s) for s in range(5)]
        print({k:(round(float(np.mean([d[k] for d in res])),3) if isinstance(res[0][k],float) else res[0][k]) for k in res[0]}, 'min_Nsp over seeds', min(d['min_Nsp'] for d in res))
