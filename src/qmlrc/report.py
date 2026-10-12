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
        "| Model | Family | AUPRC (test) | Accuracy | Balanced acc. | CV AUPRC | Configs tried | Time / seed (s, 1 thread) |",
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


def params_table(rows: list[dict], family: str | None = "quantum") -> str:
    lines = ["| Model | Seed | Selected hyperparameters | CV AUPRC |", "|---|---|---|---|"]
    for r in rows:
        if family is None or r["family"] == family:
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


def edge_lines(rows: list[dict], axes: dict[str, dict[str, list]], min_seeds: int = 2) -> list[str]:
    """Grid axes whose CV-selected value sat on an edge of the grid in at least ``min_seeds`` seeds."""
    hits: dict[tuple, int] = defaultdict(int)
    for r in rows:
        params = json.loads(r["best_params"])
        for axis, values in axes.get(r["model"].split("@")[0], {}).items():
            value = params.get(axis)
            if value == min(values):
                hits[(r["model"], axis, "lowest", min(values))] += 1
            elif value == max(values):
                hits[(r["model"], axis, "highest", max(values))] += 1
    n_seeds = len({r["seed"] for r in rows})
    return [
        f"- {model}: `{axis}` took its {side} grid value ({value}) in {count}/{n_seeds} seeds."
        for (model, axis, side, value), count in sorted(hits.items())
        if count >= min_seeds
    ]


def verdict(diffs: np.ndarray, first: str = "quantum", second: str = "classical") -> str:
    """DESIGN.md section 6, for diffs = first - second: a side wins if it is ahead on >= 80% of
    seeds and its mean paired lead exceeds the standard deviation of the paired differences."""
    n = len(diffs)
    sd = diffs.std(ddof=1) if n > 1 else 0.0
    if (diffs > 0).sum() >= 0.8 * n and diffs.mean() > sd:
        return f"{first} better"
    if (diffs < 0).sum() >= 0.8 * n and -diffs.mean() > sd:
        return f"{second} better"
    return "no clear difference"


def ablation_table(rows: list[dict], sep_rows: list[dict], twins: dict[str, str]) -> str:
    models = _by_model(rows + sep_rows)
    lines = [
        "| Entangled model | AUPRC | Separable twin | AUPRC | Entangled − separable (paired) | Seeds entangled ahead | Verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for ent, sep in twins.items():
        if ent not in models or sep not in models:
            continue
        a = {r["seed"]: r["auprc"] for r in models[ent]}
        b = {r["seed"]: r["auprc"] for r in models[sep]}
        seeds = sorted(set(a) & set(b))
        d = np.array([a[s_] - b[s_] for s_ in seeds])
        sd = d.std(ddof=1) if len(d) > 1 else 0.0
        lines.append(
            f"| {ent} | {_ms([a[s_] for s_ in seeds])} | {sep} | {_ms([b[s_] for s_ in seeds])} "
            f"| {d.mean():+.3f} ± {sd:.3f} | {int((d > 0).sum())}/{len(d)} | {verdict(d, 'entangled', 'separable')} |"
        )
    return "\n".join(lines)


def geometry_table(geo_rows: list[dict]) -> str:
    lines = [
        "| Quantum kernel model | g, mean ± sd over seeds | g / √N | √N | Closest classical kernel (most frequent) | Seeds |",
        "|---|---|---|---|---|---|",
    ]
    for model, rs in _by_model(geo_rows).items():
        kernels = defaultdict(int)
        for r in rs:
            kernels[r["classical_kernel"]] += 1
        top = max(kernels, key=kernels.get)
        lines.append(
            f"| {model} | {_ms([r['g'] for r in rs])} | {np.mean([r['g_over_sqrt_n'] for r in rs]):.3f} "
            f"| {rs[0]['sqrt_n']:.1f} | {top} ({kernels[top]}/{len(rs)}) | {len(rs)} |"
        )
    return "\n".join(lines)


def verdict_table(rows: list[dict]) -> str:
    models = _by_model(rows)
    classical = [m for m, rs in models.items() if rs[0]["family"] == "classical"]
    quantum = [m for m, rs in models.items() if rs[0]["family"] == "quantum"]
    lines = ["| Quantum model | " + " | ".join(f"vs {c}" for c in classical) + " |", "|---|" + "---|" * len(classical)]
    for q in quantum:
        qa = {r["seed"]: r["auprc"] for r in models[q]}
        cells = []
        for c in classical:
            ca = {r["seed"]: r["auprc"] for r in models[c]}
            seeds = sorted(set(qa) & set(ca))
            d = np.array([qa[s_] - ca[s_] for s_ in seeds])
            cells.append(f"{d.mean():+.3f} ({int((d > 0).sum())}/{len(d)}) · {verdict(d)}")
        lines.append(f"| {q} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def shots_table(exact_rows: list[dict], shot_rows: list[dict]) -> str:
    """AUPRC of each kernel model with exact kernels and with finite-shot kernels, plus the paired
    verdict (DESIGN.md §6) of exact minus R shots."""
    by = _by_model(exact_rows + shot_rows)
    shots = sorted({int(m.split("@")[1]) for m in by if "@" in m})
    bases = [m for m in by if "@" not in m and any(f"{m}@{r}" in by for r in shots)]
    head = "| Model | Exact kernel | " + " | ".join(f"{r:,} shots" for r in shots) + " |"
    lines = [head, "|---|---|" + "---|" * len(shots)]
    for b in bases:
        exact = {r_["seed"]: r_["auprc"] for r_ in by[b]}
        cells = []
        for r in shots:
            rows = by.get(f"{b}@{r}", [])
            noisy = {r_["seed"]: r_["auprc"] for r_ in rows}
            seeds = sorted(set(exact) & set(noisy))
            d = np.array([exact[s_] - noisy[s_] for s_ in seeds])
            cells.append(f"{_ms([noisy[s_] for s_ in seeds])} ({verdict(d, 'exact', 'shots')})")
        lines.append(f"| {b} | {_ms(list(exact.values()))} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def shot_cost_table(n_train: int, n_test: int, shots: list[int]) -> str:
    from .shots import fidelity_shot_count, projected_shot_count

    lines = ["| Model | " + " | ".join(f"{r:,} shots" for r in shots) + " |", "|---|" + "---|" * len(shots)]
    lines.append("| QSVM-fidelity | " + " | ".join(f"{fidelity_shot_count(n_train, n_test, r):.2e}" for r in shots) + " |")
    lines.append("| QSVM-projected | " + " | ".join(f"{projected_shot_count(n_train, n_test, r):.2e}" for r in shots) + " |")
    return "\n".join(lines)


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
    sep, geo = context["sep_rows"], context["geo_rows"]
    cfg = context["q8_config"]
    conc_cfg = context["conc_config"]
    n_geo = int(round(geo[0]["sqrt_n"] ** 2)) if geo else 0
    n_test = int(q8[0]["n_test"])
    n_train = cfg.n_samples - n_test
    shots_list = sorted({int(r["model"].split("@")[1]) for r in context["shot_rows"]})
    if cfg.dataset == "creditcard":
        source = "credit-card fraud, OpenML dataset 1597 (ULB / Worldline, 284,807 transactions, 492 frauds)"
        n_raw = 29
    else:
        source = f"synthetic smoke-test data (`{cfg.dataset}`), not a real benchmark"
        n_raw = "all"
    pos = np.mean([r["test_positive_rate"] for r in q8])
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
        f"- Machine: {context['platform']}, {context['cpus']} CPU threads visible. Each (seed, model) task ran in its own "
        f"single-threaded process pinned to one core, {context.get('jobs', 1)} at a time, so the times below are single-core "
        "wall-clock times on this machine. For quantum models they are **classical simulation times**, not quantum "
        "hardware times.",
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
        "Decision rule of DESIGN.md §6, applied to every quantum/classical pair. Each cell: mean paired AUPRC "
        "difference (quantum minus classical), seeds where the quantum model is ahead, verdict. A side is "
        "declared better only if it is ahead on at least 4 of 5 seeds *and* its mean lead exceeds the standard "
        "deviation of the paired differences.",
        "",
        verdict_table(q8),
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
        "### Grid-edge check",
        "",
        "Axes whose CV-selected value sat on the edge of its grid in at least 2 of the seeds (DESIGN.md §4). "
        "An edge hit can mean the grid is too narrow, or just that the CV surface is flat there.",
        "",
        f"Main table ({cfg.n_components} features):",
        "",
        *(edge_lines(q8, context["grid_axes"]) or ["- None."]),
        "",
        "Entanglement ablation:",
        "",
        *(edge_lines(sep, context["grid_axes"]) or ["- None."]),
        "",
        "Finite-shot kernels:",
        "",
        *(edge_lines(context["shot_rows"], context["grid_axes"]) or ["- None."]),
        "",
        f"Reference run ({n_raw} features, classical only):",
        "",
        *(edge_lines(full, context["grid_axes"]) or ["- None."]),
        "",
        "## Entanglement ablation",
        "",
        "Twins that differ only in entanglement (`creditcard_q8_ablation`), in the spirit of Bowles et al. "
        "(arXiv:2403.07059). The kernel twins use the same IQP map with its MultiRZ gates removed. VQC-sep "
        "(single-qubit rotations, no CNOTs) is compared with VQC-mean, an entangled VQC with exactly the same "
        "readout and training (standardised mean of ⟨Z_i⟩, log-scale output weight). The main table's VQC reads "
        "⟨Z_0⟩ only and is not part of this comparison. Same rows, splits, folds and grids as the main table. "
        "If a twin without entanglement does as well, the model's performance does not come from entanglement.",
        "",
        summary_table(sep),
        "",
        ablation_table(q8, sep, context["twins"]),
        "",
        "Selected hyperparameters:",
        "",
        params_table(sep, family=None),
        "",
        "## Finite-shot quantum kernels",
        "",
        "The same two kernel SVMs, with every kernel value estimated from a finite number of measurements, as a "
        "quantum computer would (`creditcard_q8_shots`; estimators in `qmlrc/shots.py`). Fidelity kernel: each "
        "entry is the frequency of the all-zeros outcome of the overlap circuit. Projected kernel: each Pauli "
        "expectation is estimated from that many shots. The models are re-tuned on the noisy kernels, with the "
        "same grids. In each cell, the verdict compares exact and finite-shot kernels with the DESIGN.md §6 rule.",
        "",
        shots_table(q8, context["shot_rows"]),
        "",
        f"Circuit executions needed to estimate the kernels of *one* fitted model ({n_train:,} training and "
        f"{n_test:,} test rows, one bandwidth), before any hyperparameter search:",
        "",
        shot_cost_table(n_train, n_test, shots_list),
        "",
        "## Geometric difference (Huang et al.)",
        "",
        f"For each seed, the Gram matrix of each tuned quantum kernel on the {n_geo:,} training rows is compared with "
        "a linear kernel and nine Gaussian kernels, following Appendix L of Huang et al. (arXiv:2011.01938): "
        "both normalised to trace N, regularised geometric difference "
        "g = √‖√K_Q K_C (K_C + λI)⁻² √K_Q‖, minimised over classical kernels and λ with training-error term "
        "g_tra < 0.045. Their bound gives s_C ≤ g² s_Q: when g is much smaller than √N, a classical kernel "
        "model is guaranteed to predict about as well as the quantum one *for any labels on these inputs*; "
        "only g of order √N leaves room for a quantum prediction advantage.",
        "",
        geometry_table(geo),
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
        f"Off-diagonal entries of the fidelity kernel |⟨φ(x)|φ(x')⟩|² for {conc_cfg.n_samples} rows of the benchmark "
        f"data (seed {conc_cfg.seed}; same preprocessing, with PCA to n components for n qubits), and of the "
        "projected kernel with a fixed γ = 1. What to look for: *exponential concentration* "
        "(Thanasilp et al., arXiv:2208.11060) is the off-diagonal mean and spread shrinking towards the Haar-random "
        "value as qubits are added at a fixed bandwidth. The Gram matrix then drifts towards the identity, every "
        "point looks unrelated to every other, and on hardware resolving the tiny values would take exponentially "
        "many shots. The simulation here is exact, so the only cost is generalisation, which a smaller bandwidth s "
        "can recover. The Haar column is the mean for fully random states.",
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
