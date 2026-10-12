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
from .vqc import VariationalQuantumClassifier

SCORING = "average_precision"

# Grids as of run 2 (session 1). After run 1, every axis whose CV-selected value sat on the edge of
# the grid in at least 2 of 5 seeds was extended by one value, for classical and quantum models
# alike (see JOURNAL.md, 2026-10-12).

SVM_C = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
CLASS_WEIGHT = [None, "balanced"]
# Bandwidths are multiplicative factors, like gamma = factor * 'scale' for the RBF kernel.
RBF_GAMMA_FACTORS = [0.1, 0.3, 1.0, 3.0, 10.0]
FIDELITY_SCALES = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
PROJECTED_SCALES = [0.03, 0.1, 0.3, 1.0]
PROJECTED_GAMMA_FACTORS = [0.1, 1.0, 10.0, 100.0]
VQC_GRID = {"scale": [0.03, 0.1, 0.3, 1.0], "n_layers": [2, 4, 6]}


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


def _fidelity_qsvm(split: Split, seed: int) -> ModelResult:
    states = {s: iqp_states(split.X_train, s) for s in FIDELITY_SCALES}
    kernels = {s: fidelity_kernel(S, S) for s, S in states.items()}
    scale, search, n_configs = _precomputed_svm(kernels, split.y_train, split.cv_splits)
    K_test = fidelity_kernel(iqp_states(split.X_test, scale), states[scale])
    best = search.best_estimator_
    return ModelResult(
        scores=best.decision_function(K_test),
        y_pred=best.predict(K_test),
        best_params={"scale": scale, **_clean(search.best_params_)},
        n_configs=n_configs,
        cv_score=float(search.best_score_),
    )


def _projected_qsvm(split: Split, seed: int) -> ModelResult:
    blochs = {s: bloch_vectors(split.X_train, s) for s in PROJECTED_SCALES}
    kernels, gammas = {}, {}
    for s, B in blochs.items():
        # analogue of 'scale': exponent is gamma * |dr|^2 / 2 summed over 3n Bloch coordinates
        base = 2.0 / (B.shape[1] * max(B.var(), 1e-12))
        for g in PROJECTED_GAMMA_FACTORS:
            gammas[(s, g)] = g * base
            kernels[(s, g)] = projected_kernel(B, B, g * base)
    (scale, factor), search, n_configs = _precomputed_svm(kernels, split.y_train, split.cv_splits)
    K_test = projected_kernel(bloch_vectors(split.X_test, scale), blochs[scale], gammas[(scale, factor)])
    best = search.best_estimator_
    return ModelResult(
        scores=best.decision_function(K_test),
        y_pred=best.predict(K_test),
        best_params={"scale": scale, "gamma_factor": factor, **_clean(search.best_params_)},
        n_configs=n_configs,
        cv_score=float(search.best_score_),
    )


def make_vqc_runner(steps: int = 150, grid: dict | None = None) -> Callable[[Split, int], ModelResult]:
    grid = grid or VQC_GRID

    def run(split: Split, seed: int) -> ModelResult:
        from sklearn.metrics import average_precision_score

        configs = [dict(zip(grid, values)) for values in product(*grid.values())]
        best_cfg, best_score = None, -np.inf
        for cfg in configs:
            fold_scores = []
            for train_idx, val_idx in split.cv_splits:
                model = VariationalQuantumClassifier(steps=steps, random_state=seed, **cfg)
                model.fit(split.X_train[train_idx], split.y_train[train_idx])
                fold_scores.append(
                    average_precision_score(split.y_train[val_idx], model.decision_function(split.X_train[val_idx]))
                )
            if np.mean(fold_scores) > best_score:
                best_cfg, best_score = cfg, float(np.mean(fold_scores))
        model = VariationalQuantumClassifier(steps=steps, random_state=seed, **best_cfg)
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
            _sklearn_grid(
                LogisticRegression(max_iter=5000),
                {"C": [1e-4, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], "class_weight": CLASS_WEIGHT},
            ),
            "L2 logistic regression; C x class_weight",
        ),
        ModelSpec("RBF-SVM", "classical", _rbf_svm, "SVC, RBF kernel; C x gamma x class_weight"),
        ModelSpec(
            "GradBoost", "classical",
            _sklearn_grid(
                HistGradientBoostingClassifier(),
                # class_weight is left out: it slows sklearn's histogram GBT ~10x on this data,
                # and reweighting matters little for a ranking metric such as AUPRC.
                {
                    "learning_rate": [0.03, 0.1, 0.3],
                    "max_leaf_nodes": [7, 31, 63],
                    "min_samples_leaf": [5, 20, 50],
                    "l2_regularization": [0.0, 1.0, 10.0],
                },
            ),
            "Histogram gradient-boosted trees; learning_rate x max_leaf_nodes x min_samples_leaf x l2",
        ),
        ModelSpec(
            "MLP", "classical",
            _sklearn_grid(
                MLPClassifier(max_iter=2000),
                {"hidden_layer_sizes": [(16,), (64,), (32, 32)], "alpha": [1e-4, 1e-2, 1.0, 10.0]},
            ),
            "Small multilayer perceptron (Adam); width x L2 penalty",
        ),
    ]


def quantum_models(vqc_steps: int = 150, vqc_grid: dict | None = None) -> list[ModelSpec]:
    return [
        ModelSpec("QSVM-fidelity", "quantum", _fidelity_qsvm,
                  "SVC on |<phi(x)|phi(x')>|^2, IQP/ZZ feature map (2 reps); bandwidth x C x class_weight"),
        ModelSpec("QSVM-projected", "quantum", _projected_qsvm,
                  "SVC on projected kernel (1-qubit reduced states), same feature map; bandwidth x gamma x C x class_weight"),
        ModelSpec("VQC", "quantum", make_vqc_runner(vqc_steps, vqc_grid),
                  "IQP feature map (1 rep) + StronglyEntanglingLayers, Adam; bandwidth x depth"),
    ]


def timed(run: Callable[[Split, int], ModelResult], split: Split, seed: int) -> ModelResult:
    start = time.perf_counter()
    result = run(split, seed)
    result.seconds = time.perf_counter() - start
    return result
