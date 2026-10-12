"""Reproduce every result in RESULTS.md from scratch.

    python -m qmlrc.run_all            # full benchmark (about 10 min with 4 CPU cores), rewrites results + RESULTS.md
    python -m qmlrc.run_all --quick    # synthetic smoke test (~1 min), writes to --out-dir only
    python -m qmlrc.run_all --report-only   # rebuild RESULTS.md from the committed CSVs

Run from the repository root.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

from . import experiments as ex
from .benchmark import (
    CONCENTRATION_FIELDS, GEOMETRY_FIELDS, RESULT_FIELDS, run_benchmark, run_concentration, run_geometry,
)
from .data import load_pool
from .models import TWINS, grid_axes
from .report import load_reading, read_csv, render_results, results_hash, write_csv

PACKAGES = ["numpy", "scipy", "scikit-learn", "pennylane", "jax"]


def _commit(root: Path) -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=root, capture_output=True, text=True).stdout.strip()
        return sha + ("+uncommitted-src-changes" if dirty else "")
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="tiny synthetic run for smoke tests")
    parser.add_argument("--report-only", action="store_true", help="skip computation, rebuild RESULTS.md from CSVs")
    parser.add_argument("--out-dir", default=".", help="repository root to write into (default: current directory)")
    parser.add_argument("--jobs", type=int, default=None,
                        help="parallel single-threaded workers (default: 1 with --quick, else min(4, CPUs))")
    args = parser.parse_args(argv)

    root = Path(args.out_dir).resolve()
    jobs = args.jobs or (1 if args.quick else min(4, os.cpu_count() or 1))
    configs = ex.quick() if args.quick else ex.FULL
    by_name = {cfg.name: cfg for cfg in configs}
    q8_cfg, full_cfg = by_name["creditcard_q8"], by_name["creditcard_full_classical"]
    sep_cfg, geo_cfg = by_name["creditcard_q8_ablation"], by_name["geometric_difference"]
    shots_cfg = by_name["creditcard_q8_shots"]
    conc_cfg = by_name["kernel_concentration"]
    paths = {cfg.name: root / "experiments" / cfg.name / "results.csv" for cfg in configs}
    log = lambda msg: print(msg, flush=True)  # noqa: E731

    info_path = root / "experiments" / "run_info.json"
    if not args.report_only:
        start = time.perf_counter()
        pools: dict[str, tuple] = {}
        results: dict[str, list[dict]] = {}
        seconds: dict[str, float] = {}

        def pool_for(dataset: str):
            if dataset not in pools:
                log(f"Loading dataset '{dataset}' ...")
                pools[dataset] = load_pool(dataset)
            return pools[dataset]

        for cfg in configs:
            log(f"Running {cfg.name}: {cfg.description}")
            t0 = time.perf_counter()
            if isinstance(cfg, ex.BenchmarkConfig):
                X, y = pool_for(cfg.dataset)
                results[cfg.name] = run_benchmark(cfg, X, y, log, jobs=jobs)
                write_csv(results[cfg.name], RESULT_FIELDS, paths[cfg.name])
            elif isinstance(cfg, ex.GeometryConfig):
                split_cfg = by_name[cfg.benchmarks[0]]
                X, y = pool_for(split_cfg.dataset)
                rows = [r for name in cfg.benchmarks for r in results[name]]
                write_csv(run_geometry(cfg, split_cfg, rows, X, y, log, jobs=jobs), GEOMETRY_FIELDS, paths[cfg.name])
            else:
                X, y = pool_for(cfg.dataset)
                write_csv(run_concentration(cfg, X, y, log), CONCENTRATION_FIELDS, paths[cfg.name])
            seconds[cfg.name] = round(time.perf_counter() - t0, 1)
        info = {
            "command": "python -m qmlrc.run_all" + (" --quick" if args.quick else "") + (f" --jobs {args.jobs}" if args.jobs else ""),
            "date": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "commit": _commit(root),
            "total_seconds": round(time.perf_counter() - start, 1),
            "experiment_seconds": seconds,
            "platform": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
            "cpus": os.cpu_count(),
            "jobs": jobs,
            "versions": ", ".join(f"{p} {version(p)}" for p in PACKAGES),
        }
        info_path.write_text(json.dumps(info, indent=2) + "\n")
    info = json.loads(info_path.read_text())

    q8_rows, full_rows = read_csv(paths[q8_cfg.name]), read_csv(paths[full_cfg.name])
    sep_rows, geo_rows = read_csv(paths[sep_cfg.name]), read_csv(paths[geo_cfg.name])
    shot_rows = read_csv(paths[shots_cfg.name])
    conc_rows = read_csv(paths[conc_cfg.name])
    fingerprint = results_hash(q8_rows + full_rows + sep_rows + shot_rows)
    context = {
        **info,
        "q8_rows": q8_rows,
        "full_rows": full_rows,
        "sep_rows": sep_rows,
        "geo_rows": geo_rows,
        "shot_rows": shot_rows,
        "conc_rows": conc_rows,
        "q8_config": q8_cfg,
        "conc_config": conc_cfg,
        "hash": fingerprint,
        "grid_axes": grid_axes(),
        "twins": TWINS,
        "reading": load_reading(root / "experiments" / "READING.md", fingerprint),
    }
    (root / "RESULTS.md").write_text(render_results(context))
    log(f"Wrote {root / 'RESULTS.md'} (results fingerprint {fingerprint}); computation took {info['total_seconds'] / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
