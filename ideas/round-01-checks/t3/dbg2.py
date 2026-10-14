import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-quantum-ml-reality-check/3242e8f8-b193-546c-b381-3fe888e7979d/scratchpad/round01/t3")
import numpy as np, jax, jax.numpy as jnp
from pilot import *
n=6; N=1000; rng=np.random.default_rng(0)
B=bits(n); Z=1-2*B; pairs=np.array([(i,j) for i in range(n) for j in range(i+1,n)])
true=rng.uniform(0,2*np.pi,n+len(pairs))
Zj,Pj=jnp.asarray(Z),jnp.asarray(pairs)
f=lambda th: iqp_probs(th,n,Zj,Pj)
p=np.asarray(f(jnp.asarray(true)))
idx=rng.choice(2**n,size=N,p=p); c=counts(idx,n)
nll=lambda th: -(jnp.asarray(c)*jnp.log(f(th)+1e-300)).sum()/c.sum()
th=jnp.asarray(true)
print('nll true', float(nll(th)), 'kl true', kl(p,np.asarray(f(th))))
g=jax.grad(nll)(th); print('grad norm at true', float(jnp.linalg.norm(g)))
# finite-difference check
e=np.zeros_like(true); e[3]=1e-5
print('fd', (float(nll(th+e))-float(nll(th-e)))/2e-5, 'ad', float(g[3]))
th2=th-0.02*g/jnp.linalg.norm(g)
print('after small step nll', float(nll(th2)), 'kl', kl(p,np.asarray(f(th2))))
v,thf=adam_fit(nll,jnp.asarray(true)+0.3*rng.normal(size=true.shape),1500,0.02,nll)
q=np.asarray(f(thf))
print('returned val', v, 'recomputed nll', float(nll(thf)), 'emp CE', float(-(c/c.sum()*np.log(q)).sum()), 'H(p)+KL', float(-(p*np.log(p)).sum())+kl(p,q))
contrib=p*np.log(p/np.clip(q,1e-300,None)); o=np.argsort(-contrib)[:5]
for i in o: print(i, 'p',round(p[i],4),'q',q[i],'count',c[i],'contrib',round(contrib[i],3))
print('jit nll', float(jax.jit(nll)(thf)), 'plain nll', float(nll(thf)))
print('jit f sum', float(jax.jit(f)(thf).sum()), 'max abs diff', float(jnp.abs(jax.jit(f)(thf)-f(thf)).max()))
print(thf.dtype, thf[:3])
