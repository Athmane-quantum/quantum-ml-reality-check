import sys, inspect, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, "/home/user/quantum-ml-reality-check/src")
from qmlrc import vqc
print(inspect.signature(vqc._functions.__wrapped__ if hasattr(vqc._functions,'__wrapped__') else vqc._functions))
rng=np.random.default_rng(0)
for ent, ro in [(True,'z0'),(True,'mean'),(False,'mean')]:
    logits, step, circ_jit = vqc._functions(8, 2, 1, ent, ro, False, False)
    # unjitted reference: rebuild the qnode outputs through the jitted object's python fn
    w = jnp.asarray(rng.normal(0, 0.5, (2, 8, 3))); x = jnp.asarray(rng.normal(0, 1, (4, 8)))
    a = circ_jit(w, x)
    with jax.disable_jit():
        b = circ_jit(w, x)
    print(ent, ro, 'max |jit - nojit| =', float(jnp.abs(jnp.asarray(a) - jnp.asarray(b)).max()))
