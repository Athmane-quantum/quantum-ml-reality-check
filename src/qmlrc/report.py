"""Turn raw result rows into RESULTS.md. Every number in RESULTS.md comes from these rows."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import DISCLAIMER

READING_HASH_PREFIX = "results-hash:"


def write_csv(rows: list[dict], fields: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for k, v in row.items():
            try:
                row[k] = int(v) if k in {"seed", "n_test", "n_configs", "n_qubits"} else float(v)
            except (TypeError, ValueError):
                pass
    return rows


def results_hash(rows: list[dict]) -> str:
    """Fingerprint of the metric values (rounded to 3 decimals, timings excluded)."""
    keys = sorted(
        (r["experiment"], r["seed"], r["model"], round(r["auprc"], 3), round(r["accuracy"], 3))
        for r in rows
    )
    return hashlib.sha256(json.dumps(keys).encode()).hexdigest()[:16]


def _ms(values) -> str:
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.3f} ± {values.std(ddof=1) if len(values) > 1 else 0.0:.3f}"


def _by_model(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["model"]].append(r)
    return out


def summary_table(rows: list[dict]) -> str:
    lines = [
        "| Model | Family | AUPRC (test) | Accuracy | Balanced acc. | CV AUPRC | Configs tried | Time / seed (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model, rs in _by_model(rows).items():
        lines.append(
            f"| {model} | {rs[0]['family']} | {_ms([r['auprc'] for r in rs])} | {_ms([r['accuracy'] for r in rs])} "
            f"| {_ms([r['balanced_accuracy'] for r in rs])} | {_ms([r['cv_auprc'] for r in rs])} "
            f"| {int(rs[0]['n_configs'])} | {np.mean([r['seconds'] for r in rs]):.1f} |"
        )
    return "\n".join(lines)


def per_seed_table(rows: list[dict], metric: str = "auprc") -> str:
    seeds = sorted({r["seed"] for r in rows})
    lines = ["| Model | " + " | ".join(f"seed {s}" for s in seeds) + " |", "|---|" + "---|" * len(seeds)]
    for model, rs in _by_model(rows).items():
        val = {r["seed"]: r[metric] for r in rs}
        lines.append(f"| {model} | " + " | ".join(f"{val[s]:.3f}" for s in seeds) + " |")
    return "\n".join(lines)


def params_table(rows: list[dict], family: str = "quantum") -> str:
    lines = ["| Model | Seed | Selected hyperparameters | CV AUPRC |", "|---|---|---|---|"]
    for r in rows:
        if r["family"] == family:
            params = ", ".join(f"{k}={v}" for k, v in json.loads(r["best_params"]).items())
            lines.append(f"| {r['model']} | {r['seed']} | {params} | {r['cv_auprc']:.3f} |")
    return "\n".join(lines)


def comparison_lines(rows: list[dict]) -> list[str]:
    """Automatic, paired-by-seed comparisons of each quantum model with the classical ones."""
    models = _by_model(rows)
    classical = {m: rs for m, rs in models.items() if rs[0]["family"] == "classical"}
    quantum = {m: rs for m, rs in models.items() if rs[0]["family"] == "quantum"}
    if not classical or not quantum:
        return []
    mean = {m: float(np.mean([r["auprc"] for r in rs])) for m, rs in models.items()}
    best_c = max(classical, key=mean.get)
    best_q = max(quantum, key=mean.get)
    out = [
        f"- Highest mean test AUPRC overall: **{max(mean, key=mean.get)}** ({mean[max(mean, key=mean.get)]:.3f}).",
        f"- Best classical model: **{best_c}** ({mean[best_c]:.3f}). Best quantum model: **{best_q}** ({mean[best_q]:.3f}).",
    ]
    references = [("RBF-SVM", "pre-specified like-for-like baseline: same SVM, classical kernel")] if "RBF-SVM" in classical else []
    if best_c != "RBF-SVM":
        references.append((best_c, "best classical model, picked after seeing test scores, which slightly favours classical"))
    for q, qrs in quantum.items():
        qa = {r["seed"]: r["auprc"] for r in qrs}
        for ref, why in references:
            ca = {r["seed"]: r["auprc"] for r in classical[ref]}
            seeds = sorted(set(qa) & set(ca))
            diff = np.array([qa[s] - ca[s] for s in seeds])
            wins = int((diff > 0).sum())
            sd = diff.std(ddof=1) if len(diff) > 1 else 0.0
            out.append(
                f"- {q} minus {ref} ({why}): AUPRC difference {diff.mean():+.3f} ± {sd:.3f} "
                f"(paired over {len(seeds)} seeds); {q} is higher on {wins}/{len(seeds)} seeds."
            )
    return out


def concentration_table(rows: list[dict]) -> str:
    lines = [
        "| Qubits | Bandwidth s | Fidelity kernel: mean off-diag. | Fidelity kernel: std off-diag. | Haar-random mean 1/2^n | Projected kernel (gamma=1): std off-diag. |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['n_qubits']} | {r['scale']} | {r['fidelity_offdiag_mean']:.4f} | {r['fidelity_offdiag_std']:.4f} "
            f"| {r['haar_mean']:.4f} | {r['projected_offdiag_std']:.4f} |"
        )
    return "\n".join(lines)


def load_reading(path: Path, current_hash: str) -> str:
    """Hand-written interpretation; flagged as stale if it was written for other numbers."""
    if not path.exists():
        return "_No plain-language reading has been written for these results yet._"
    text = path.read_text()
    first, _, body = text.partition("\n")
    written_for = first.replace("<!--", "").replace("-->", "").replace(READING_HASH_PREFIX, "").strip()
    if written_for != current_hash:
        warning = (
            f"> **STALE READING.** This text was written for results `{written_for}`, "
            f"but the current results are `{current_hash}`. Re-check every statement below against the tables.\n\n"
        )
        return warning + body.strip()
    return body.strip()


def render_results(context: dict) -> str:
    q8, full, conc = context["q8_rows"], context["full_rows"], context["conc_rows"]
    cfg = context["q8_config"]
    if cfg.dataset == "creditcard":
        source = "credit-card fraud, OpenML dataset 1597 (ULB / Worldline, 284,807 transactions, 492 frauds)"
        n_raw = 29
    else:
        source = f"synthetic smoke-test data (`{cfg.dataset}`), not a real benchmark"
        n_raw = "all"
    pos = np.mean([r["test_positive_rate"] for r in q8])
    n_test = int(q8[0]["n_test"])
    parts = [
        "# Results — benchmark v0",
        "",
        f"*{DISCLAIMER}*",
        "",
        "This file is generated by `python -m qmlrc.run_all`. Do not edit it by hand: edit the code or "
        "`experiments/READING.md` and re-run. Raw per-seed numbers are in `experiments/*/results.csv`.",
        "",
        "## How these numbers were produced",
        "",
        f"- Command: `{context['command']}`",
        f"- Date (UTC): {context['date']} · git commit: `{context['commit']}` · total wall-clock: {context['total_seconds'] / 60:.1f} min",
        f"- Machine: {context['platform']}, {context['cpus']} CPU threads visible. Times are wall-clock on this machine "
        "and are **classical simulation times**, not quantum hardware times.",
        f"- Versions: {context['versions']}",
        f"- Seeds: {', '.join(str(s) for s in cfg.seeds)} (each seed redraws the subsample, the split, the CV folds and the model initialisation).",
        f"- Results fingerprint: `{context['hash']}`",
        "",
        "## Setup (identical for every model)",
        "",
        f"- Data: {source}. "
        f"Each seed draws {cfg.n_samples} rows without replacement at {cfg.positive_fraction:.0%} fraud, "
        f"then a stratified {1 - cfg.test_size:.0%}/{cfg.test_size:.0%} train/test split ({n_test} test rows).",
        f"- Features: standardise → PCA to {cfg.n_components} components → standardise, fitted on the training rows only. "
        f"{cfg.n_components} features = {cfg.n_components} qubits.",
        f"- Tuning: grid search with {cfg.n_folds}-fold stratified CV on the training rows, selection by AUPRC; "
        "the same folds for every model. The test rows are used once, after tuning.",
        f"- Reference point: a random ranking has expected AUPRC equal to the fraud rate of the test set ({pos:.3f}).",
        "",
        f"## Main table: {cfg.n_components} features, all models",
        "",
        "Mean ± standard deviation over seeds. AUPRC is the primary metric. Accuracy is reported as requested but is "
        f"easy to inflate at {cfg.positive_fraction:.0%} positives (predicting 'never fraud' scores {1 - pos:.3f}).",
        "",
        summary_table(q8),
        "",
        "### Automatic comparisons (paired by seed)",
        "",
        *comparison_lines(q8),
        "",
        "### Test AUPRC per seed",
        "",
        per_seed_table(q8),
        "",
        "### Hyperparameters selected for the quantum models",
        "",
        "Bandwidth `scale` multiplies the features before encoding; small values make the quantum feature map "
        "close to a low-order (near-classical) function of the inputs.",
        "",
        params_table(q8),
        "",
        f"## Reference: classical models on {n_raw} features",
        "",
        "Same rows, splits and folds as above, but without the PCA-to-8 step. This is **not** a like-for-like "
        "comparison with the quantum models; it measures how much the qubit-count restriction costs.",
        "",
        summary_table(full),
        "",
        "## Kernel concentration vs qubit count",
        "",
        "Off-diagonal values of the fidelity kernel over 300 credit-card rows (seed 0), and of the projected kernel "
        "with a fixed gamma = 1. A spread (std) that shrinks towards 0 as qubits are added means the kernel "
        "matrix approaches a constant, so an SVM can no longer tell the points apart without exponentially many shots "
        "on hardware (here the simulation is exact, so the effect shows up as poorer generalisation instead).",
        "",
        concentration_table(conc),
        "",
        "## Plain-language reading",
        "",
        "_Written by the agent after inspecting the tables above. Kept in `experiments/READING.md`._",
        "",
        context["reading"],
        "",
    ]
    return "\n".join(parts)
