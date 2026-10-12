import json
from pathlib import Path

from qmlrc import DISCLAIMER
from qmlrc.report import load_reading, read_csv, results_hash
from qmlrc.run_all import main


def test_quick_run_all_end_to_end(tmp_path: Path):
    assert main(["--quick", "--out-dir", str(tmp_path)]) == 0
    text = (tmp_path / "RESULTS.md").read_text()
    assert DISCLAIMER in text
    for name in ["LogReg", "RBF-SVM", "GradBoost", "MLP", "QSVM-fidelity", "QSVM-projected", "VQC"]:
        assert f"| {name} |" in text
    rows = read_csv(tmp_path / "experiments" / "creditcard_q8" / "results.csv")
    assert {r["model"] for r in rows} >= {"RBF-SVM", "QSVM-fidelity"}

    # rebuilding the report from the CSVs gives the same fingerprint
    fingerprint = results_hash(rows + read_csv(tmp_path / "experiments" / "creditcard_full_classical" / "results.csv"))
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
