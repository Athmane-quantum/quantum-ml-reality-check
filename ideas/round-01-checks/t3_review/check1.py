# Adversarial checks for T3 best version. numpy only.
import numpy as np, itertools
def bits(n): return np.array(list(itertools.product([0,1],repeat=n)))
def wht(a):
    a=a.copy(); N=len(a); h=1
    while h<N:
        for i in range(0,N,2*h):
            x=a[i:i+h].copy(); y=a[i+h:i+2*h].copy(); a[i:i+h]=x+y; a[i+h:i+2*h]=x-y
        h*=2
    return a/np.sqrt(N)
def idqnn_p0(theta,edges,n):
    B=bits(n); Z=1-2*B
    amp=np.ones(2**n)/2**(n/2)*np.exp(-0.5j*(Z@theta))
    for (i,j) in edges: amp=amp*np.where((B[:,i]==1)&(B[:,j]==1),-1,1)
    return np.abs(wht(amp))**2
def iqp_p(h,J,n):
    # p = |<y| H^n exp(i sum h_i Z_i + i sum J_ij Z_i Z_j) H^n |0>|^2
    B=bits(n); Z=1-2*B
    ph=Z@h+sum(J[i,j]*Z[:,i]*Z[:,j] for i in range(n) for j in range(i+1,n))
    return np.abs(wht(np.exp(1j*ph)/2**(n/2)))**2
def grid(r,c):
    idx=lambda a,b:a*c+b
    return [(idx(a,b),idx(a,b+1)) for a in range(r) for b in range(c-1)]+[(idx(a,b),idx(a+1,b)) for a in range(r-1) for b in range(c)]
rng=np.random.default_rng(1)
for (r,c) in [(2,3),(2,5),(3,3)]:
    n=r*c; E=grid(r,c); th=rng.uniform(0,2*np.pi,n)
    p=idqnn_p0(th,E,n)
    # CZ_ij = exp(i pi/4 (1 - Z_i - Z_j + Z_i Z_j)) ; RZ(t)=exp(-i t Z/2)
    deg=np.zeros(n); J=np.zeros((n,n))
    for (i,j) in E: J[min(i,j),max(i,j)]=np.pi/4; deg[i]+=1; deg[j]+=1
    h=-th/2-np.pi/4*deg
    q=iqp_p(h,J,n)
    print(f"grid {r}x{c}: max|p_IDQNN - p_IQP| = {np.abs(p-q).max():.2e}  (IDQNN(x=0^n) is an all-to-all IQP instance)")
# Data-driven sign resolution from x=0^n samples only: ML over all 2^n sign patterns of |theta_hat|
def kl(p,q): m=p>0; return float(np.sum(p[m]*np.log(p[m]/np.clip(q[m],1e-300,None))))
r,c=2,5; n=10; E=grid(r,c)
nbr=[[j for (i,j) in E if i==q]+[i for (i,j) in E if j==q] for q in range(n)]
res=[]
for seed in range(5):
    rg=np.random.default_rng(100+seed); th=rg.uniform(0,2*np.pi,n); p=idqnn_p0(th,E,n)
    N=1000; comp=rg.integers(0,3,N); X=np.zeros((N,n),int)
    X[comp==1]=(rg.random(((comp==1).sum(),n))<0.4); X[comp==2]=(rg.random(((comp==2).sum(),n))<0.8)
    th_abs=np.zeros(n)
    for j in range(n):
        sel=(X[:,j]==0)&np.all(X[:,nbr[j]]==1,axis=1); k=sel.sum()
        y=rg.random(k)<np.sin(th[j]/2)**2
        th_abs[j]=np.arccos(np.clip(1-2*y.mean(),-1,1)) if k>0 else np.pi/2
    n0=(comp==0).sum(); ys=rg.choice(2**n,size=n0,p=p); cnt=np.bincount(ys,minlength=2**n)
    best=None
    for s in bits(n):
        t=np.where(s==1,-th_abs,th_abs); q=idqnn_p0(t,E,n)
        ll=(cnt*np.log(np.clip(q,1e-300,None))).sum()
        if best is None or ll>best[0]: best=(ll,t)
    q_ml=idqnn_p0(best[1],E,n); q_or=idqnn_p0(np.where(np.sin(th)<0,-th_abs,th_abs),E,n)
    res.append((kl(p,q_ml),kl(p,q_or),kl(p,idqnn_p0(th_abs,E,n))))
print("2x5, N=1000: KL sign-ML(x=0^n samples) / oracle-sign / sign-blind per seed:",[tuple(round(v,3) for v in t) for t in res])
