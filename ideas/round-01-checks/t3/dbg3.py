import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-quantum-ml-reality-check/3242e8f8-b193-546c-b381-3fe888e7979d/scratchpad/round01/t3")
import numpy as np, jax, jax.numpy as jnp
from pilot import *
n=6; rng=np.random.default_rng(0)
B=bits(n); Z=1-2*B; pairs=np.array([(i,j) for i in range(n) for j in range(i+1,n)])
th=rng.uniform(0,2*np.pi,n+len(pairs))
# numpy reference: full matrix
H=np.array([[1,1],[1,-1]])/np.sqrt(2); Hn=np.array([[1.]])
for _ in range(n): Hn=np.kron(Hn,H)
phase=Z@th[:n]+(Z[:,pairs[:,0]]*Z[:,pairs[:,1]])@th[n:]
pref=np.abs(Hn@(np.exp(1j*phase)/2**(n/2)))**2
Zj,Pj=jnp.asarray(Z),jnp.asarray(pairs)
pe=np.asarray(iqp_probs(jnp.asarray(th),n,Zj,Pj)); pj=np.asarray(jax.jit(lambda t: iqp_probs(t,n,Zj,Pj))(jnp.asarray(th)))
print('eager vs ref', np.abs(pe-pref).max(), 'jit vs ref', np.abs(pj-pref).max())
a=jnp.asarray(np.exp(1j*phase)/2**(n/2))
print('wht eager vs ref', float(jnp.abs(wht(a,n)-Hn@np.asarray(a)).max()), 'wht jit vs ref', float(jnp.abs(jax.jit(lambda v: wht(v,n))(a)-Hn@np.asarray(a)).max()))
Zf=jnp.asarray(Z,dtype=jnp.float64)
pj2=np.asarray(jax.jit(lambda t: iqp_probs(t,n,Zf,Pj))(jnp.asarray(th)))
print('jit float Z vs ref', np.abs(pj2-pref).max())
ph_j=jax.jit(lambda t: Zj@t[:n]+(Zj[:,Pj[:,0]]*Zj[:,Pj[:,1]])@t[n:])(jnp.asarray(th))
print('phase jit vs ref', float(jnp.abs(ph_j-phase).max()))
ph_j2=jax.jit(lambda t: Zf@t[:n]+(Zf[:,Pj[:,0]]*Zf[:,Pj[:,1]])@t[n:])(jnp.asarray(th))
print('phase jit float vs ref', float(jnp.abs(ph_j2-phase).max()))
F=np.column_stack([Z, Z[:,pairs[:,0]]*Z[:,pairs[:,1]]]).astype(np.float64)
Fj=jnp.asarray(F)
ph3=jax.jit(lambda t: Fj@t)(jnp.asarray(th)); print('phase jit precomputed F vs ref', float(jnp.abs(ph3-phase).max()))
ph4=jax.jit(lambda t: jnp.dot(Fj,t,precision=jax.lax.Precision.HIGHEST))(jnp.asarray(th)); print('HIGHEST', float(jnp.abs(ph4-phase).max()))
print(jax.devices(), jax.config.read('jax_enable_x64'), ph3.dtype)
