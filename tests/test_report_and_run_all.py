import json
from pathlib import Path

from qmlrc import DISCLAIMER
from qmlrc.report import load_reading, read_csv, results_hash
from qmlrc.run_all import main


def test_quick_run_all_end_to_end(tmp_path: Path):
    assert main(["--quick", "--out-dir", str(tmp_path)]) == 0
    text = (tmp_path / "RESULTS.md").read_text()
    assert DISCLAIMER in text
    for name in ["LogReg", "RBF-SVM", "GradBoost", "MLP", "QSVM-fidelity", "QSVM-projected", "VQC",
                 "VQC-mean", "QSVM-fidelity-sep", "QSVM-projected-sep", "VQC-sep"]:
        assert f"| {name} |" in text
    for section in ["## Entanglement ablation", "## Finite-shot quantum kernels", "## Geometric difference"]:
        assert section in text
    shots = read_csv(tmp_path / "experiments" / "creditcard_q8_shots" / "results.csv")
    assert {r["model"] for r in shots} == {f"{m}@{r}" for m in ("QSVM-fidelity", "QSVM-projected")
                                           for r in (100, 1000, 10000)}
    geo = read_csv(tmp_path / "experiments" / "geometric_difference" / "results.csv")
    assert len(geo) == 4 and all(r["g"] > 0 for r in geo)
    rows = read_csv(tmp_path / "experiments" / "creditcard_q8" / "results.csv")
    assert {r["model"] for r in rows} >= {"RBF-SVM", "QSVM-fidelity"}

    # rebuilding the report from the CSVs gives the same fingerprint
    fingerprint = results_hash(
        rows
        + read_csv(tmp_path / "experiments" / "creditcard_full_classical" / "results.csv")
        + read_csv(tmp_path / "experiments" / "creditcard_q8_ablation" / "results.csv")
        + read_csv(tmp_path / "experiments" / "creditcard_q8_shots" / "results.csv")
    )
    assert main(["--quick", "--report-only", "--out-dir", str(tmp_path)]) == 0
    assert fingerprint in (tmp_path / "RESULTS.md").read_text()


def test_reading_is_flagged_stale_when_numbers_change(tmp_path: Path):
    path = tmp_path / "READING.md"
    path.write_text("<!-- results-hash: abc -->\nClassical wins.\n")
    assert load_reading(path, "abc") == "Classical wins."
    assert "STALE READING" in load_reading(path, "def")


def test_edge_lines_flag_values_on_grid_edges():
    from qmlrc.report import edge_lines

    rows = [{"model": "M", "seed": s, "best_params": json.dumps({"a": a, "b": 2})} for s, a in enumerate([1, 1, 3])]
    lines = edge_lines(rows, {"M": {"a": [1, 2, 3], "b": [1, 2, 3]}})
    assert lines == ["- M: `a` took its lowest grid value (1) in 2/3 seeds."]


def test_verdict_rule_is_symmetric():
    import numpy as np

    from qmlrc.report import verdict

    assert verdict(np.array([0.02, 0.03, 0.025, 0.02, 0.03])) == "quantum better"
    assert verdict(-np.array([0.02, 0.03, 0.025, 0.02, 0.03])) == "classical better"
    assert verdict(np.array([0.02, -0.03, 0.025, 0.02, -0.03])) == "no clear difference"
    assert verdict(np.array([0.1, 0.1, 0.1, 0.1, -0.5]), "entangled", "separable") == "no clear difference"
    assert verdict(np.array([0.02, 0.03, 0.025, 0.02, 0.0]), "entangled", "separable") == "entangled better"
