"""Variational quantum classifier, simulated with PennyLane and trained with JAX.

Circuit: ``IQPEmbedding(s * x)`` (same feature map as the quantum kernels, one
repetition) followed by ``StronglyEntanglingLayers``; the model output is
``logit = a * <Z_0> + b`` with trainable ``a`` and ``b``. Training minimises a
class-weighted binary cross-entropy with Adam on mini-batches, using exact
(backpropagated) gradients of the noiseless simulation.
"""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
from sklearn.base import BaseEstimator, ClassifierMixin

from .quantum import MAX_QUBITS

jax.config.update("jax_enable_x64", True)

_PREDICT_CHUNK = 256


@lru_cache(maxsize=None)
def _functions(n_qubits: int, n_layers: int, n_repeats: int):
    """Jitted forward pass and Adam step for one circuit shape (compiled once)."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="jax", diff_method="backprop")
    def circuit(weights, x):
        qml.IQPEmbedding(x, wires=range(n_qubits), n_repeats=n_repeats)
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    def logits(params, x):
        return params["a"] * circuit(params["w"], x) + params["b"]

    def loss(params, x, y, sample_weight):
        z = logits(params, x)
        # numerically stable binary cross-entropy with logits
        per_sample = jnp.maximum(z, 0.0) - z * y + jnp.log1p(jnp.exp(-jnp.abs(z)))
        return jnp.sum(sample_weight * per_sample) / jnp.sum(sample_weight)

    value_and_grad = jax.value_and_grad(loss)

    @jax.jit
    def adam_step(params, m, v, t, x, y, sample_weight, lr):
        value, grads = value_and_grad(params, x, y, sample_weight)
        b1, b2, eps = 0.9, 0.999, 1e-8
        m = jax.tree_util.tree_map(lambda m_, g: b1 * m_ + (1 - b1) * g, m, grads)
        v = jax.tree_util.tree_map(lambda v_, g: b2 * v_ + (1 - b2) * g**2, v, grads)
        m_hat = jax.tree_util.tree_map(lambda m_: m_ / (1 - b1**t), m)
        v_hat = jax.tree_util.tree_map(lambda v_: v_ / (1 - b2**t), v)
        params = jax.tree_util.tree_map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, m_hat, v_hat)
        return params, m, v, value

    return jax.jit(logits), adam_step


class VariationalQuantumClassifier(ClassifierMixin, BaseEstimator):
    """Binary classifier; ``decision_function`` returns logits (positive class = 1)."""

    def __init__(
        self,
        n_layers: int = 2,
        scale: float = 0.3,
        n_repeats: int = 1,
        steps: int = 150,
        batch_size: int = 64,
        learning_rate: float = 0.05,
        init_std: float = 0.1,
        random_state: int = 0,
    ):
        self.n_layers = n_layers
        self.scale = scale
        self.n_repeats = n_repeats
        self.steps = steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.init_std = init_std
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        n_qubits = X.shape[1]
        if n_qubits > MAX_QUBITS:
            raise ValueError(f"{n_qubits} features would need {n_qubits} qubits; the limit is {MAX_QUBITS}")
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = n_qubits
        rng = np.random.default_rng(self.random_state)

        # 'balanced' class weights, as for the classical models' class_weight option
        n_pos = max(int(y.sum()), 1)
        n_neg = max(len(y) - int(y.sum()), 1)
        weights = np.where(y == 1, len(y) / (2.0 * n_pos), len(y) / (2.0 * n_neg))

        shape = qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=n_qubits)
        params = {
            "w": jnp.asarray(rng.normal(scale=self.init_std, size=shape)),
            "a": jnp.asarray(1.0),
            "b": jnp.asarray(0.0),
        }
        m = jax.tree_util.tree_map(jnp.zeros_like, params)
        v = jax.tree_util.tree_map(jnp.zeros_like, params)
        _, adam_step = _functions(n_qubits, self.n_layers, self.n_repeats)

        Xs = self.scale * X
        batch = min(self.batch_size, len(y))
        losses = []
        for t in range(1, self.steps + 1):
            idx = rng.choice(len(y), size=batch, replace=False)
            params, m, v, value = adam_step(
                params, m, v, t, jnp.asarray(Xs[idx]), jnp.asarray(y[idx], dtype=float),
                jnp.asarray(weights[idx]), self.learning_rate,
            )
            losses.append(float(value))
        self.params_ = jax.tree_util.tree_map(np.asarray, params)
        self.loss_curve_ = losses
        return self

    def decision_function(self, X):
        X = self.scale * np.asarray(X, dtype=float)
        logits, _ = _functions(self.n_features_in_, self.n_layers, self.n_repeats)
        params = jax.tree_util.tree_map(jnp.asarray, self.params_)
        out = []
        # fixed-size chunks so the jitted forward pass compiles only once per shape
        for start in range(0, len(X), _PREDICT_CHUNK):
            chunk = X[start : start + _PREDICT_CHUNK]
            pad = _PREDICT_CHUNK - len(chunk)
            padded = np.vstack([chunk, np.zeros((pad, X.shape[1]))]) if pad else chunk
            out.append(np.asarray(logits(params, jnp.asarray(padded)))[: len(chunk)])
        return np.concatenate(out)

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)
