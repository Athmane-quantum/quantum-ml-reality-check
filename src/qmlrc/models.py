"""Models and their hyperparameter grids.

Fairness rules (see DESIGN.md):
* every model is tuned by grid search on the same CV folds of the training set,
  with the same selection metric (average precision = AUPRC);
* the grid size (number of configurations tried) is reported for every model;
* the test set is used once, after tuning, to score the refitted best model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .data import Split
from .quantum import bloch_vectors, fidelity_kernel, iqp_states, projected_kernel
from .shots import clip_to_psd, sample_bloch_vectors, sample_fidelity_kernel
from .vqc import VariationalQuantumClassifier

SCORING = "average_precision"

# Grid-edge rule (DESIGN.md §4): an axis end whose CV-selected value sits on the edge in at least 2 of
# 5 seeds is extended by one value, at most once, for classical and quantum models alike. Extensions:
# session 1 (most axes), session 3 (SVM_C at both ends, VQC depth 1). See JOURNAL.md.

SVM_C = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
CLASS_WEIGHT = [None, "balanced"]
# Bandwidths are multiplicative factors, like gamma = factor * 'scale' for the RBF kernel.
RBF_GAMMA_FACTORS = [0.1, 0.3, 1.0, 3.0, 10.0]
FIDELITY_SCALES = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
PROJECTED_SCALES = [0.03, 0.1, 0.3, 1.0]
PROJECTED_GAMMA_FACTORS = [0.1, 1.0, 10.0, 100.0]
VQC_GRID = {"scale": [0.03, 0.1, 0.3, 1.0], "n_layers": [1, 2, 4, 6]}
# Twin VQCs of the entanglement ablation: same readout and training, they differ only in entanglement
VQC_CONTROL = {"readout": "mean", "log_output_scale": True, "standardise_readout": True}
SHOTS = (100, 1000, 10000)
LOGREG_GRID = {"C": [1e-4, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], "class_weight": CLASS_WEIGHT}
# class_weight is left out: it slows sklearn's histogram GBT ~10x on this data,
# and reweighting matters little for a ranking metric such as AUPRC.
GRADBOOST_GRID = {
    "learning_rate": [0.01, 0.03, 0.1, 0.3, 1.0],
    "max_leaf_nodes": [3, 7, 31, 63],
    "min_samples_leaf": [2, 5, 20, 50],
    "l2_regularization": [0.0, 1.0, 10.0, 100.0],
}
MLP_GRID = {"hidden_layer_sizes": [(16,), (64,), (32, 32)], "alpha": [1e-4, 1e-2, 1.0, 10.0]}


def grid_axes() -> dict[str, dict[str, list]]:
    """Numeric grid axes per model, keyed as in ``best_params``, for the grid-edge check."""
    return {
        "LogReg": {"C": LOGREG_GRID["C"]},
        "RBF-SVM": {"C": SVM_C, "gamma_factor": RBF_GAMMA_FACTORS},
        "GradBoost": dict(GRADBOOST_GRID),
        "MLP": {"alpha": MLP_GRID["alpha"]},
        "QSVM-fidelity": {"scale": FIDELITY_SCALES, "C": SVM_C},
        "QSVM-projected": {"scale": PROJECTED_SCALES, "gamma_factor": PROJECTED_GAMMA_FACTORS, "C": SVM_C},
        "VQC": dict(VQC_GRID),
        "QSVM-fidelity-sep": {"scale": FIDELITY_SCALES, "C": SVM_C},
        "QSVM-projected-sep": {"scale": PROJECTED_SCALES, "gamma_factor": PROJECTED_GAMMA_FACTORS, "C": SVM_C},
        "VQC-sep": dict(VQC_GRID),
        "VQC-mean": dict(VQC_GRID),
    }


@dataclass
class ModelResult:
    scores: np.ndarray  # decision scores on the test set (higher = more likely positive)
    y_pred: np.ndarray  # hard predictions on the test set
    best_params: dict
    n_configs: int  # number of hyperparameter configurations tried
    cv_score: float  # mean CV AUPRC of the selected configuration
    seconds: float = 0.0  # wall-clock: tuning + final fit + test prediction


@dataclass
class ModelSpec:
    name: str
    family: str  # "classical" or "quantum"
    run: Callable[[Split, int], ModelResult]
    description: str = ""
    tags: list[str] = field(default_factory=list)


def _sklearn_grid(estimator, grid: dict | list[dict]) -> Callable[[Split, int], ModelResult]:
    def run(split: Split, seed: int) -> ModelResult:
        est = clone(estimator)
        if "random_state" in est.get_params():
            est.set_params(random_state=seed)
        search = GridSearchCV(est, grid, scoring=SCORING, cv=split.cv_splits, n_jobs=1, refit=True)
        search.fit(split.X_train, split.y_train)
        best = search.best_estimator_
        scores = best.decision_function(split.X_test) if hasattr(best, "decision_function") else best.predict_proba(split.X_test)[:, 1]
        return ModelResult(
            scores=np.asarray(scores),
            y_pred=best.predict(split.X_test),
            best_params=_clean(search.best_params_),
            n_configs=len(search.cv_results_["params"]),
            cv_score=float(search.best_score_),
        )

    return run


def _rbf_svm(split: Split, seed: int) -> ModelResult:
    # gamma grid expressed relative to sklearn's 'scale' heuristic 1 / (n_features * Var[X])
    base = 1.0 / (split.X_train.shape[1] * split.X_train.var())
    grid = {"C": SVM_C, "class_weight": CLASS_WEIGHT, "gamma": [f * base for f in RBF_GAMMA_FACTORS]}
    result = _sklearn_grid(SVC(kernel="rbf"), grid)(split, seed)
    result.best_params["gamma_factor"] = round(result.best_params.pop("gamma") / base, 6)
    return result


def _precomputed_svm(kernels: dict, y_train, cv_splits) -> tuple[dict, GridSearchCV, int]:
    """Tune C and class_weight for each candidate kernel; return the best overall."""
    best_key, best_search, n_configs = None, None, 0
    for key, K_train in kernels.items():
        search = GridSearchCV(
            SVC(kernel="precomputed"), {"C": SVM_C, "class_weight": CLASS_WEIGHT},
            scoring=SCORING, cv=cv_splits, n_jobs=1, refit=True,
        )
        search.fit(K_train, y_train)
        n_configs += len(search.cv_results_["params"])
        if best_search is None or search.best_score_ > best_search.best_score_:
            best_key, best_search = key, search
    return best_key, best_search, n_configs


def _rng(seed: int, *keys: int) -> np.random.Generator:
    return np.random.default_rng([seed, *keys])


def make_fidelity_qsvm(entangling: bool = True, shots: int | None = None) -> Callable[[Split, int], ModelResult]:
    """Fidelity-kernel SVM; with ``shots``, every kernel entry is a finite-shot estimate (qmlrc.shots)."""

    def run(split: Split, seed: int) -> ModelResult:
        states = {s: iqp_states(split.X_train, s, entangling=entangling) for s in FIDELITY_SCALES}
        kernels = {}
        for i, (s, S) in enumerate(states.items()):
            K = fidelity_kernel(S, S)
            if shots:
                K = clip_to_psd(sample_fidelity_kernel(K, shots, _rng(seed, shots, 0, i), symmetric=True))
            kernels[s] = K
        scale, search, n_configs = _precomputed_svm(kernels, split.y_train, split.cv_splits)
        K_test = fidelity_kernel(iqp_states(split.X_test, scale, entangling=entangling), states[scale])
        if shots:
            K_test = sample_fidelity_kernel(K_test, shots, _rng(seed, shots, 1, FIDELITY_SCALES.index(scale)))
        best = search.best_estimator_
        return ModelResult(
            scores=best.decision_function(K_test),
            y_pred=best.predict(K_test),
            best_params={"scale": scale, **_clean(search.best_params_)},
            n_configs=n_configs,
            cv_score=float(search.best_score_),
        )

    return run


def projected_gamma(bloch: np.ndarray, factor: float) -> float:
    """gamma = factor * base, the analogue of sklearn's 'scale': the exponent is
    gamma * |dr|^2 / 2 summed over the 3n Bloch coordinates."""
    base = 2.0 / (bloch.shape[1] * max(bloch.var(), 1e-12))
    return factor * base


def make_projected_qsvm(entangling: bool = True, shots: int | None = None) -> Callable[[Split, int], ModelResult]:
    """Projected-kernel SVM; with ``shots``, Bloch vectors are finite-shot estimates (qmlrc.shots)."""

    def run(split: Split, seed: int) -> ModelResult:
        blochs = {}
        for i, s in enumerate(PROJECTED_SCALES):
            B = bloch_vectors(split.X_train, s, entangling=entangling)
            blochs[s] = sample_bloch_vectors(B, shots, _rng(seed, shots, 0, i)) if shots else B
        kernels, gammas = {}, {}
        for s, B in blochs.items():
            for g in PROJECTED_GAMMA_FACTORS:
                gammas[(s, g)] = projected_gamma(B, g)
                kernels[(s, g)] = projected_kernel(B, B, gammas[(s, g)])
        (scale, factor), search, n_configs = _precomputed_svm(kernels, split.y_train, split.cv_splits)
        B_test = bloch_vectors(split.X_test, scale, entangling=entangling)
        if shots:
            B_test = sample_bloch_vectors(B_test, shots, _rng(seed, shots, 1, PROJECTED_SCALES.index(scale)))
        K_test = projected_kernel(B_test, blochs[scale], gammas[(scale, factor)])
        best = search.best_estimator_
        return ModelResult(
            scores=best.decision_function(K_test),
            y_pred=best.predict(K_test),
            best_params={"scale": scale, "gamma_factor": factor, **_clean(search.best_params_)},
            n_configs=n_configs,
            cv_score=float(search.best_score_),
        )

    return run


def make_vqc_runner(
    steps: int = 150, grid: dict | None = None, **options
) -> Callable[[Split, int], ModelResult]:
    """Grid search over ``grid`` for a VariationalQuantumClassifier with fixed ``options``
    (entangling, readout, log_output_scale, standardise_readout)."""
    grid = grid or VQC_GRID

    def run(split: Split, seed: int) -> ModelResult:
        from sklearn.metrics import average_precision_score

        configs = [dict(zip(grid, values)) for values in product(*grid.values())]
        best_cfg, best_score = None, -np.inf
        for cfg in configs:
            fold_scores = []
            for train_idx, val_idx in split.cv_splits:
                model = VariationalQuantumClassifier(steps=steps, random_state=seed, **options, **cfg)
                model.fit(split.X_train[train_idx], split.y_train[train_idx])
                fold_scores.append(
                    average_precision_score(split.y_train[val_idx], model.decision_function(split.X_train[val_idx]))
                )
            if np.mean(fold_scores) > best_score:
                best_cfg, best_score = cfg, float(np.mean(fold_scores))
        model = VariationalQuantumClassifier(steps=steps, random_state=seed, **options, **best_cfg)
        model.fit(split.X_train, split.y_train)
        return ModelResult(
            scores=model.decision_function(split.X_test),
            y_pred=model.predict(split.X_test),
            best_params=dict(best_cfg),
            n_configs=len(configs),
            cv_score=best_score,
        )

    return run


def _clean(params: dict) -> dict:
    return {k: (v if v is not None else "none") for k, v in params.items()}


def classical_models() -> list[ModelSpec]:
    return [
        ModelSpec(
            "LogReg", "classical",
            _sklearn_grid(LogisticRegression(max_iter=5000), LOGREG_GRID),
            "L2 logistic regression; C x class_weight",
        ),
        ModelSpec("RBF-SVM", "classical", _rbf_svm, "SVC, RBF kernel; C x gamma x class_weight"),
        ModelSpec(
            "GradBoost", "classical",
            _sklearn_grid(HistGradientBoostingClassifier(), GRADBOOST_GRID),
            "Histogram gradient-boosted trees; learning_rate x max_leaf_nodes x min_samples_leaf x l2",
        ),
        ModelSpec(
            "MLP", "classical",
            _sklearn_grid(MLPClassifier(max_iter=2000), MLP_GRID),
            "Small multilayer perceptron (Adam); width x L2 penalty",
        ),
    ]


def quantum_models(vqc_steps: int = 150, vqc_grid: dict | None = None) -> list[ModelSpec]:
    return [
        ModelSpec("QSVM-fidelity", "quantum", make_fidelity_qsvm(),
                  "SVC on |<phi(x)|phi(x')>|^2, IQP/ZZ feature map (2 reps); bandwidth x C x class_weight"),
        ModelSpec("QSVM-projected", "quantum", make_projected_qsvm(),
                  "SVC on projected kernel (1-qubit reduced states), same feature map; bandwidth x gamma x C x class_weight"),
        ModelSpec("VQC", "quantum", make_vqc_runner(vqc_steps, vqc_grid),
                  "IQP feature map (1 rep) + StronglyEntanglingLayers, Adam; bandwidth x depth"),
    ]


def separable_models(vqc_steps: int = 150, vqc_grid: dict | None = None) -> list[ModelSpec]:
    """Unentangled twins of the quantum models, same grids (DESIGN.md, entanglement ablation)."""
    return [
        ModelSpec("QSVM-fidelity-sep", "quantum-separable", make_fidelity_qsvm(entangling=False),
                  "QSVM-fidelity with the MultiRZ (entangling) gates removed from the feature map"),
        ModelSpec("QSVM-projected-sep", "quantum-separable", make_projected_qsvm(entangling=False),
                  "QSVM-projected with the MultiRZ (entangling) gates removed from the feature map"),
        ModelSpec("VQC-sep", "quantum-separable",
                  make_vqc_runner(vqc_steps, vqc_grid, entangling=False, **VQC_CONTROL),
                  "VQC without entangling gates (Rot layers); standardised mean_i <Z_i> readout"),
    ]


def vqc_control_models(vqc_steps: int = 150, vqc_grid: dict | None = None) -> list[ModelSpec]:
    """Entangled VQC with exactly the readout and training of VQC-sep, so the pair differs only in
    entanglement (session 3; replaces the session-2 VQC/VQC-sep pairing, which also differed in readout)."""
    return [
        ModelSpec("VQC-mean", "quantum", make_vqc_runner(vqc_steps, vqc_grid, entangling=True, **VQC_CONTROL),
                  "VQC (StronglyEntanglingLayers) with standardised mean_i <Z_i> readout"),
    ]


def shot_models(shots: tuple[int, ...] = SHOTS) -> list[ModelSpec]:
    """Quantum kernel SVMs whose kernels are finite-shot estimates, re-tuned on the noisy kernels."""
    specs = []
    for r in shots:
        specs += [
            ModelSpec(f"QSVM-fidelity@{r}", "quantum-shots", make_fidelity_qsvm(shots=r),
                      f"QSVM-fidelity, every kernel entry estimated from {r} shots"),
            ModelSpec(f"QSVM-projected@{r}", "quantum-shots", make_projected_qsvm(shots=r),
                      f"QSVM-projected, every Pauli expectation estimated from {r} shots"),
        ]
    return specs


# entangled model -> its separable twin (same readout and training)
TWINS = {"QSVM-fidelity": "QSVM-fidelity-sep", "QSVM-projected": "QSVM-projected-sep", "VQC-mean": "VQC-sep"}


def timed(run: Callable[[Split, int], ModelResult], split: Split, seed: int) -> ModelResult:
    start = time.perf_counter()
    result = run(split, seed)
    result.seconds = time.perf_counter() - start
    return result
