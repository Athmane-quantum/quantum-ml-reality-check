"""Run one benchmark configuration over all seeds and return one row per (seed, model)."""

from __future__ import annotations

import json
from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score

from .data import make_preprocessor, make_split, subsample
from .experiments import BenchmarkConfig, ConcentrationConfig
from .models import ModelSpec, classical_models, quantum_models, timed
from .quantum import bloch_vectors, fidelity_kernel, iqp_states, projected_kernel

RESULT_FIELDS = [
    "experiment", "seed", "model", "family", "auprc", "accuracy", "balanced_accuracy",
    "test_positive_rate", "n_test", "cv_auprc", "n_configs", "seconds", "best_params",
]


def metrics(y_true: np.ndarray, scores: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "auprc": float(average_precision_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def models_for(cfg: BenchmarkConfig) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    if "classical" in cfg.families:
        specs += classical_models()
    if "quantum" in cfg.families:
        specs += quantum_models(cfg.vqc_steps, cfg.vqc_grid)
    return specs


def run_benchmark(cfg: BenchmarkConfig, X_pool, y_pool, log: Callable[[str], None] = print) -> list[dict]:
    rows = []
    specs = models_for(cfg)
    for seed in cfg.seeds:
        split = make_split(
            X_pool, y_pool, n_samples=cfg.n_samples, positive_fraction=cfg.positive_fraction,
            test_size=cfg.test_size, n_components=cfg.n_components, n_folds=cfg.n_folds, seed=seed,
        )
        for spec in specs:
            result = timed(spec.run, split, seed)
            row = {
                "experiment": cfg.name,
                "seed": seed,
                "model": spec.name,
                "family": spec.family,
                **metrics(split.y_test, result.scores, result.y_pred),
                "test_positive_rate": float(split.y_test.mean()),
                "n_test": int(len(split.y_test)),
                "cv_auprc": result.cv_score,
                "n_configs": result.n_configs,
                "seconds": result.seconds,
                "best_params": json.dumps(result.best_params, sort_keys=True),
            }
            rows.append(row)
            log(f"  [{cfg.name}] seed={seed} {spec.name:<15} AUPRC={row['auprc']:.3f} "
                f"acc={row['accuracy']:.3f} ({result.seconds:.1f}s, {result.n_configs} configs)")
    return rows


CONCENTRATION_FIELDS = [
    "experiment", "n_qubits", "scale", "fidelity_offdiag_mean", "fidelity_offdiag_std",
    "projected_offdiag_mean", "projected_offdiag_std", "haar_mean",
]


def run_concentration(cfg: ConcentrationConfig, X_pool, y_pool, log: Callable[[str], None] = print) -> list[dict]:
    """Off-diagonal kernel statistics vs qubit count and bandwidth (no labels used)."""
    rng = np.random.default_rng(cfg.seed)
    X, _ = subsample(X_pool, y_pool, cfg.n_samples, cfg.positive_fraction, rng)
    iu = np.triu_indices(cfg.n_samples, k=1)
    rows = []
    for n in cfg.qubits:
        Z = make_preprocessor(n).fit_transform(X)
        for s in cfg.scales:
            S = iqp_states(Z, s)
            Kf = fidelity_kernel(S, S)[iu]
            B = bloch_vectors(Z, s)
            # projected kernel with a fixed gamma = 1 (no data-dependent rescaling) to expose concentration
            Kp = projected_kernel(B, B, 1.0)[iu]
            rows.append({
                "experiment": cfg.name, "n_qubits": n, "scale": s,
                "fidelity_offdiag_mean": float(Kf.mean()), "fidelity_offdiag_std": float(Kf.std()),
                "projected_offdiag_mean": float(Kp.mean()), "projected_offdiag_std": float(Kp.std()),
                "haar_mean": 1.0 / 2**n,
            })
            log(f"  [{cfg.name}] n={n:<2} scale={s:<4} fidelity mean={Kf.mean():.4f} std={Kf.std():.4f} "
                f"| projected std={Kp.std():.4f}")
    return rows
