import numpy as np
import pytest

from qmlrc.benchmark import metrics
from qmlrc.models import (
    classical_models, quantum_models, separable_models, shot_models, timed, vqc_control_models,
)

TINY_VQC = {"scale": [0.3], "n_layers": [1, 2]}
SPECS = (classical_models() + quantum_models(5, TINY_VQC) + vqc_control_models(5, TINY_VQC)
         + separable_models(5, TINY_VQC) + shot_models((100,)))
EXPECTED_CONFIGS = {"LogReg": 16, "RBF-SVM": 80, "GradBoost": 144, "MLP": 12,
                    "QSVM-fidelity": 96, "QSVM-projected": 256, "VQC": 2, "VQC-mean": 2,
                    "QSVM-fidelity-sep": 96, "QSVM-projected-sep": 256, "VQC-sep": 2,
                    "QSVM-fidelity@100": 96, "QSVM-projected@100": 256}


@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_every_model_runs_and_reports_its_budget(spec, small_split):
    result = timed(spec.run, small_split, seed=0)
    n_test = len(small_split.y_test)
    assert result.scores.shape == (n_test,) and np.all(np.isfinite(result.scores))
    assert result.y_pred.shape == (n_test,)
    assert result.n_configs == EXPECTED_CONFIGS[spec.name]
    assert result.seconds > 0
    m = metrics(small_split.y_test, result.scores, result.y_pred)
    assert 0.0 <= m["auprc"] <= 1.0 and 0.0 <= m["accuracy"] <= 1.0


def test_metrics_on_perfect_and_random_rankings():
    y = np.array([0, 0, 0, 1, 1])
    assert metrics(y, np.array([0, 1, 2, 3, 4.0]), (np.arange(5) > 2).astype(int))["auprc"] == 1.0
    assert metrics(y, np.zeros(5), np.zeros(5, dtype=int))["accuracy"] == 0.6


def test_results_do_not_depend_on_number_of_workers(pool):
    from qmlrc.benchmark import run_benchmark
    from qmlrc.experiments import BenchmarkConfig

    X, y = pool
    cfg = BenchmarkConfig(name="tiny", dataset="synthetic", n_samples=120, positive_fraction=0.2,
                          n_components=3, seeds=(0, 1), families=("classical",))
    one = run_benchmark(cfg, X, y, log=lambda _: None, jobs=1)
    two = run_benchmark(cfg, X, y, log=lambda _: None, jobs=2)
    key = lambda r: (r["seed"], r["model"], round(r["auprc"], 12), r["best_params"])  # noqa: E731
    assert [key(r) for r in one] == [key(r) for r in two]
