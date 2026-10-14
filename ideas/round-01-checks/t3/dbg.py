import numpy as np, jax, jax.numpy as jnp
import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-quantum-ml-reality-check/3242e8f8-b193-546c-b381-3fe888e7979d/scratchpad/round01/t3")
from pilot import *
n=6; N=1000; rng=np.random.default_rng(0)
B=bits(n); Z=1-2*B; pairs=np.array([(i,j) for i in range(n) for j in range(i+1,n)])
true=rng.uniform(0,2*np.pi,n+len(pairs))
Zj,Pj=jnp.asarray(Z),jnp.asarray(pairs)
f=lambda th: iqp_probs(th,n,Zj,Pj)
p=np.asarray(f(jnp.asarray(true)))
idx=rng.choice(2**n,size=N,p=p); c=counts(idx,n)
nll=lambda th: -(jnp.asarray(c)*jnp.log(f(th)+1e-300)).sum()/c.sum()
print('NLL true', float(nll(jnp.asarray(true))), 'entropy', float(-(p*np.log(p)).sum()))
vals=[]
for r in range(20):
    x0=jnp.asarray(rng.uniform(0,2*np.pi,n+len(pairs)))
    v,th=adam_fit(nll,x0,1500,0.05,nll)
    vals.append((v,kl(p,np.asarray(f(th)))))
vals.sort(); print('best 5 restarts (trainNLL, KL):',[(round(a,3),round(b,3)) for a,b in vals[:5]]); print('worst', vals[-1])
# from near-true init
v,th=adam_fit(nll,jnp.asarray(true)+0.3*rng.normal(size=true.shape),1500,0.02,nll); print('near-true init', v, kl(p,np.asarray(f(th))))
