import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-quantum-ml-reality-check/3242e8f8-b193-546c-b381-3fe888e7979d/scratchpad/round01/t3")
import numpy as np, jax, jax.numpy as jnp
from pilot import *
n=6; rng=np.random.default_rng(1); B=bits(n)
L=3; th=jnp.asarray(rng.normal(size=L*n*2))
a=hea_probs(th.reshape(L,n,2),n,L); b=jax.jit(lambda t: hea_probs(t.reshape(L,n,2),n,L))(th); print('hea jit diff', float(jnp.abs(a-b).max()), float(a.sum()))
chi=4; th=jnp.asarray(rng.normal(size=n*2*chi*chi))
a=mps_probs(th,n,chi,B); b=jax.jit(lambda t: mps_probs(t,n,chi,B))(th); print('mps jit diff', float(jnp.abs(a-b).max()))
# brute-force MPS reference
A=np.asarray(th).reshape(n,2,chi,chi); ref=[]
for x in B:
    v=np.ones(chi)/np.sqrt(chi)
    for k in range(n): v=v@A[k,x[k]]
    ref.append(v.sum()**2)
ref=np.array(ref); ref/=ref.sum(); print('mps vs brute', np.abs(np.asarray(b)-ref).max())
# HEA vs pennylane reference
import pennylane as qml
dev=qml.device('default.qubit',wires=n)
P=np.asarray(rng.normal(size=(L,n,2)))
@qml.qnode(dev)
def circ():
    for l in range(L):
        for q in range(n):
            qml.RY(P[l,q,0],wires=q); qml.RZ(P[l,q,1],wires=q)
        for i in range(n): qml.CZ(wires=[i,(i+1)%n])
    return qml.probs(wires=range(n))
print('hea vs pennylane', float(np.abs(np.asarray(circ())-np.asarray(jax.jit(lambda t: hea_probs(t,n,L))(jnp.asarray(P)))).max()))
