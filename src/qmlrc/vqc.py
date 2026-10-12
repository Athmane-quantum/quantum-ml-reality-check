"""Variational quantum classifier, simulated with PennyLane and trained with JAX.

Circuit: ``IQPEmbedding(s * x)`` (same feature map as the quantum kernels, one
repetition) followed by ``StronglyEntanglingLayers``; the model output is
``logit = a * <Z_0> + b`` with trainable ``a`` and ``b``. Training minimises a
class-weighted binary cross-entropy with Adam on mini-batches, using exact
(backpropagated) gradients of the noiseless simulation.

Options used by the entanglement ablation (DESIGN.md §4):

* ``readout="mean"``: the output observable is mean_i <Z_i> (the observable of the
  separable classifier of Bowles et al., arXiv:2403.07059) instead of <Z_0>.
* ``log_output_scale=True``: the trainable output scale is a = exp(c) instead of a.
  Small-bandwidth circuits produce readouts that vary by only a few hundredths, so a
  linear ``a`` (starting at 1, moving ~0.05 per Adam step) cannot reach useful logits;
  on a log scale it can.
* ``standardise_readout=True``: the readout is standardised, (o - mean) / std, before the
  affine map (a one-output batch normalisation): batch statistics during training, statistics
  of the whole training set, frozen, for prediction. Without it a nearly constant readout makes
  the output scale and the bias indistinguishable, and the scale never grows.
* ``entangling=False``: the embedding loses its MultiRZ gates and each layer is a
  ``Rot`` on every qubit with no CNOTs (same parameter count). Requires ``readout="mean"``,
  since a single-qubit readout of a product state sees one feature only.
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
def _functions(
    n_qubits: int, n_layers: int, n_repeats: int, entangling: bool = True,
    readout: str = "z0", log_output_scale: bool = False, standardise_readout: bool = False,
):
    """Jitted forward pass and Adam step for one circuit shape (compiled once)."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="jax", diff_method="backprop")
    def entangled(weights, x):
        qml.IQPEmbedding(x, wires=range(n_qubits), n_repeats=n_repeats)
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    @qml.qnode(dev, interface="jax", diff_method="backprop")
    def entangled_all(weights, x):
        qml.IQPEmbedding(x, wires=range(n_qubits), n_repeats=n_repeats)
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    @qml.qnode(dev, interface="jax", diff_method="backprop")
    def separable(weights, x):
        qml.IQPEmbedding(x, wires=range(n_qubits), n_repeats=n_repeats, pattern=[])
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.Rot(weights[layer, q, 0], weights[layer, q, 1], weights[layer, q, 2], wires=q)
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    def circuit(weights, x):
        if entangling and readout == "z0":
            return entangled(weights, x)
        if entangling:
            return jnp.mean(jnp.stack(entangled_all(weights, x)), axis=0)
        return jnp.mean(jnp.stack(separable(weights, x)), axis=0)

    def logits(params, x, stats=None):
        """``stats`` = (mean, std) of the readout; None means: use this batch's statistics."""
        o = circuit(params["w"], x)
        if standardise_readout:
            mean, std = (jnp.mean(o), jnp.std(o)) if stats is None else stats
            o = (o - mean) / (std + 1e-8)
        a = jnp.exp(params["a"]) if log_output_scale else params["a"]
        return a * o + params["b"]

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

    return jax.jit(logits), adam_step, jax.jit(circuit)


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
        entangling: bool = True,
        readout: str = "z0",
        log_output_scale: bool = False,
        standardise_readout: bool = False,
        random_state: int = 0,
    ):
        self.n_layers = n_layers
        self.scale = scale
        self.n_repeats = n_repeats
        self.steps = steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.init_std = init_std
        self.entangling = entangling
        self.readout = readout
        self.log_output_scale = log_output_scale
        self.standardise_readout = standardise_readout
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        n_qubits = X.shape[1]
        if n_qubits > MAX_QUBITS:
            raise ValueError(f"{n_qubits} features would need {n_qubits} qubits; the limit is {MAX_QUBITS}")
        if self.readout not in ("z0", "mean"):
            raise ValueError(f"unknown readout {self.readout!r}")
        if not self.entangling and self.readout != "mean":
            raise ValueError("a separable circuit needs readout='mean' (a <Z_0> readout would see one feature)")
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
            "a": jnp.asarray(0.0 if self.log_output_scale else 1.0),  # output scale starts at 1 either way
            "b": jnp.asarray(0.0),
        }
        m = jax.tree_util.tree_map(jnp.zeros_like, params)
        v = jax.tree_util.tree_map(jnp.zeros_like, params)
        _, adam_step, _ = _functions(*self._shape(n_qubits))

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
        self.readout_stats_ = None
        if self.standardise_readout:  # freeze the readout statistics of the whole training set
            o = self._readout(Xs)
            self.readout_stats_ = (float(o.mean()), float(o.std()))
        return self

    def _readout(self, Xs: np.ndarray) -> np.ndarray:
        _, _, circuit = _functions(*self._shape(self.n_features_in_))
        w = jnp.asarray(self.params_["w"])
        return np.concatenate([np.asarray(circuit(w, jnp.asarray(c)))[: len(c0)]
                               for c, c0 in self._chunks(Xs)])

    @staticmethod
    def _chunks(Xs: np.ndarray):
        # fixed-size chunks so the jitted functions compile only once per shape
        for start in range(0, len(Xs), _PREDICT_CHUNK):
            chunk = Xs[start : start + _PREDICT_CHUNK]
            pad = _PREDICT_CHUNK - len(chunk)
            yield (np.vstack([chunk, np.zeros((pad, Xs.shape[1]))]) if pad else chunk), chunk

    def decision_function(self, X):
        Xs = self.scale * np.asarray(X, dtype=float)
        logits, _, _ = _functions(*self._shape(self.n_features_in_))
        params = jax.tree_util.tree_map(jnp.asarray, self.params_)
        stats = None if self.readout_stats_ is None else tuple(jnp.asarray(v) for v in self.readout_stats_)
        return np.concatenate([np.asarray(logits(params, jnp.asarray(padded), stats))[: len(chunk)]
                               for padded, chunk in self._chunks(Xs)])

    def _shape(self, n_qubits: int) -> tuple:
        return (n_qubits, self.n_layers, self.n_repeats, self.entangling, self.readout, self.log_output_scale,
                self.standardise_readout)

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)
