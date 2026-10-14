import numpy as np, itertools, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
n=6; B=np.array(list(itertools.product([0,1],repeat=n))); Z=1-2*B
P=np.array([(i,j) for i in range(n) for j in range(i+1,n)])
Zj,Pj=jnp.asarray(Z),jnp.asarray(P); t=jnp.asarray(np.random.default_rng(0).uniform(0,6,len(P)))
g=lambda t: (Zj[:,Pj[:,0]]*Zj[:,Pj[:,1]])@t
ref=(Z[:,P[:,0]]*Z[:,P[:,1]])@np.asarray(t)
print('eager err',float(jnp.abs(g(t)-ref).max()),'jit err',float(jnp.abs(jax.jit(g)(t)-ref).max()))
h=lambda t: Zj@t[:n]
print('matvec int64@float64 jit err', float(jnp.abs(jax.jit(h)(jnp.asarray(np.arange(len(P),dtype=float)))-Z@np.arange(n,dtype=float)).max()))
M=(Z[:,P[:,0]]*Z[:,P[:,1]]); Mj=jnp.asarray(M)
k=lambda t: Mj@t
print('int64 matrix @ float64 vec, jit err', float(jnp.abs(jax.jit(k)(t)-ref).max()), 'eager err', float(jnp.abs(k(t)-ref).max()))
print(jax.__version__, jaxlib_v:=__import__('jaxlib').__version__)
