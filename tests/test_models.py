import numpy as np
import pytest

from qmlrc.benchmark import metrics
from qmlrc.models import classical_models, quantum_models, timed

SPECS = classical_models() + quantum_models(vqc_steps=5, vqc_grid={"scale": [0.3], "n_layers": [1, 2]})
EXPECTED_CONFIGS = {"LogReg": 14, "RBF-SVM": 60, "GradBoost": 24, "MLP": 9,
                    "QSVM-fidelity": 60, "QSVM-projected": 108, "VQC": 2}


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
