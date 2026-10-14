# Timing: conditional Q-HEA (input-dependent |0>/|+> preparation) at n=10, depth L, exact NLL, batch B.
import time, numpy as np, jax, jax.numpy as jnp, itertools, sys
jax.config.update("jax_enable_x64", True)
n=10; L=int(sys.argv[1]); B=int(sys.argv[2])
Bits=np.array(list(itertools.product([0,1],repeat=n)))
cz=np.ones(2**n)
for i in range(n):
    j=(i+1)%n; cz*=np.where((Bits[:,i]==1)&(Bits[:,j]==1),-1.0,1.0)
cz=jnp.asarray(cz.reshape((2,)*n))
def states(params,X):
    # X: (B,n) inputs; qubit i starts in |+> if x_i=0 else |0>
    s=1/np.sqrt(2)
    q0=jnp.where(X[:,:,None]==0, jnp.array([s,s]), jnp.array([1.0,0.0])).astype(jnp.complex128)  # (B,n,2)
    psi=q0[:,0]
    for i in range(1,n): psi=jnp.einsum('b...,bk->b...k',psi,q0[:,i])
    for l in range(L):
        for q in range(n):
            a,b=params[l,q,0],params[l,q,1]
            ry=jnp.array([[jnp.cos(a/2),-jnp.sin(a/2)],[jnp.sin(a/2),jnp.cos(a/2)]],dtype=jnp.complex128)
            rz=jnp.array([[jnp.exp(-1j*b/2),0],[0,jnp.exp(1j*b/2)]],dtype=jnp.complex128)
            psi=jnp.moveaxis(jnp.tensordot(rz@ry,psi,axes=([1],[q+1])),0,q+1)
        psi=psi*cz[None]
    return psi.reshape(psi.shape[0],-1)
def nll(params,X,Y):
    psi=states(params,X); p=jnp.abs(psi)**2
    return -jnp.mean(jnp.log(p[jnp.arange(X.shape[0]),Y]+1e-300))
g=jax.jit(jax.value_and_grad(nll))
rng=np.random.default_rng(0)
X=jnp.asarray(rng.integers(0,2,(B,n))); Y=jnp.asarray(rng.integers(0,2**n,B)); th=jnp.asarray(rng.normal(0,1,(L,n,2)))
t0=time.time(); v,gr=g(th,X,Y); gr.block_until_ready(); t1=time.time()
for _ in range(10): v,gr=g(th,X,Y)
gr.block_until_ready(); t2=time.time()
print(f"L={L} B={B}: compile {t1-t0:.1f}s, per step {(t2-t1)/10*1000:.0f} ms")
