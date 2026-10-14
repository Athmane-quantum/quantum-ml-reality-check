# Pilot: EA (Bell sampling) vs EF (Pauli-basis product probes) learning of a structured Pauli channel.
import numpy as np, itertools, time
from scipy.optimize import nnls
rng=np.random.default_rng(0)
n=4; D=4**n
# Pauli index a = sum_i code_i 4^i, code: 0 I,1 X,2 Y,3 Z ; x bit = code in {1,2}, z bit = code in {2,3}
codes=np.array([[ (a//4**i)%4 for i in range(n)] for a in range(D)])
X=np.isin(codes,[1,2]).astype(int); Z=np.isin(codes,[2,3]).astype(int)
anti=((X@Z.T + Z@X.T)%2)           # anti[a,b]=1 if P_a,P_b anticommute
wt=(codes>0).sum(1)
def pauli(s):  # 'XIIZ' -> index, site i = s[i]
    m={'I':0,'X':1,'Y':2,'Z':3}; return sum(m[c]*4**i for i,c in enumerate(s))
def mult(a,b): # index of P_a P_b (mod phase): XOR of symplectic bits
    x=X[a]^X[b]; z=Z[a]^Z[b]; c=np.where(x&z,2,np.where(x,1,np.where(z,3,0))); return int(sum(c[i]*4**i for i in range(n)))
def spl(gens,rates):
    p=np.zeros(D); p[0]=1.0
    for g,r in zip(gens,rates):
        q=(1-r)*p; perm=np.array([mult(a,g) for a in range(D)]); q2=np.zeros(D); q2[perm]+=r*p; p=q+q2
    lam=np.prod([1-2*r*anti[g] for g,r in zip(gens,rates)],axis=0)
    return p,lam
# local model: weight-1 on each site, weight-2 on ring neighbours
local=[]
for i in range(n):
    for c in 'XYZ': s=['I']*n; s[i]=c; local.append(pauli(''.join(s)))
for i in range(n):
    j=(i+1)%n
    for c1 in 'XYZ':
        for c2 in 'XYZ': s=['I']*n; s[i]=c1; s[j]=c2; local.append(pauli(''.join(s)))
local=sorted(set(local))
def ea_sample(p,N):
    c=rng.choice(D,size=N,p=p); return c
def ea_lam(c):  # empirical eigenvalues from Bell samples
    cnt=np.bincount(c,minlength=D)/len(c); return (1-2*anti).T@cnt*0+ ( (1-2*anti)@cnt )
def ef_data(p,N):
    sig=rng.integers(1,4,size=(N,n))               # random Pauli basis per shot
    c=rng.choice(D,size=N,p=p)
    # outcome bit i flips iff error's single-site Pauli anticommutes with sigma_i
    ec=codes[c]; e=((ec!=0)&(ec!=sig)).astype(int)
    return sig,e
def ef_lam(sig,e):
    lam=np.full(D,np.nan); lam[0]=1
    for b in range(1,D):
        sb=codes[b]; S=sb>0
        ok=np.all(sig[:,S]==sb[S],axis=1)
        if ok.sum()>0: lam[b]=np.mean((-1)**e[ok][:,S].sum(1))
    return lam
def fit(lamhat,gens,usable):
    b=np.where(usable & (lamhat>1e-3))[0]
    A=anti[np.ix_(b,gens)].astype(float); y=-np.log(lamhat[b])
    th,_=nnls(A,y); return np.exp(-anti[:,gens]@th)
res={}
for trial in range(5):
    rates=rng.uniform(0,0.02,len(local))
    gens=list(local); rr=list(rates)
    for mis in [0.0,0.01]:
        g2=gens+([pauli('ZZZZ')] if mis>0 else []); r2=rr+([mis] if mis>0 else [])
        p,lam=spl(g2,r2)
        for N in [10**3,10**4,10**5]:
            c=ea_sample(p,N); le=ea_lam(c)
            sig,e=ef_data(p,N); lf=ef_lam(sig,e)
            out={}
            out['EA-free']=np.nanmax(np.abs(le-lam)[1:])
            out['EA-model']=np.max(np.abs(fit(le,local,wt<=4)-lam)[1:])
            lf2=np.where(np.isnan(lf),0,lf)
            out['EF-free']=np.max(np.abs(lf2-lam)[1:])
            out['EF-model(w<=2 data)']=np.max(np.abs(fit(lf2,local,(wt<=2)&~np.isnan(lf))-lam)[1:])
            for k,v in out.items(): res.setdefault((mis,N,k),[]).append(v)
for key in sorted(res): print(key, f"{np.mean(res[key]):.4f} +- {np.std(res[key]):.4f}")
