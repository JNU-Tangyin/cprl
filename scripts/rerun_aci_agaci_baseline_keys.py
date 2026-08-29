#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import os
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TSL_DIR = PROJECT_ROOT / "time_series_library"
if str(TSL_DIR) not in sys.path:
    sys.path.append(str(TSL_DIR))

import exp.exp_conformal as exp_mod
from exp.exp_conformal import ExpConformal, get_args


OFFICIAL_31_GRID = [
    0.0,
    0.000005,
    0.00005,
    0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
    0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
]


SETTING_RE = re.compile(
    r"(?P<data>.+?)\.csv_lags(?P<lags>\d+)_model(?P<model>.+?)"
    r"_cp(?P<mode>aci|agaci)_mode(?P<run>.+?)(?:_base(?P<base_seed>\d+))?_seed(?P<cp_seed>\d+)$"
)


def _no_plot(*args, **kwargs):
    return None


def _read_done(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    with path.open(newline="") as f:
        return {
            (r["setting"], r["cp_mode"], r["target_coverage"])
            for r in csv.DictReader(f)
        }


def _target_to_alpha(target_coverage: str) -> float:
    return float(f"{1.0 - float(target_coverage):.12g}")


def _cache_dataset(data: str) -> str:
    return "exchange" if data == "exchange_rate" else data


def _data_path(data: str) -> str:
    return f"dataset/{data}.csv"


def _cache_path(data: str, model: str, seed: str) -> str:
    ds = _cache_dataset(data)
    root = PROJECT_ROOT / f"forecast_cache_seed{seed}"
    matches = sorted(root.glob(f"*{ds}_cache_{model}_*/forecast_full.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(f"cache match count={len(matches)} for data={data} model={model} seed={seed}")
    return str(matches[0].relative_to(PROJECT_ROOT))


def _load_jobs(baseline_csv: Path) -> list[dict]:
    jobs = {}
    with baseline_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["cp_mode"] not in {"aci", "agaci"}:
                continue
            match = SETTING_RE.match(row["setting"])
            if not match:
                raise ValueError(f"Cannot parse setting: {row['setting']}")
            parsed = match.groupdict()
            key = (row["setting"], row["cp_mode"], row["target_coverage"])
            jobs[key] = {
                "setting": row["setting"],
                "cp_mode": row["cp_mode"],
                "target_coverage": row["target_coverage"],
                "alpha": _target_to_alpha(row["target_coverage"]),
                "data": parsed["data"],
                "lags": int(parsed["lags"]),
                "model": parsed["model"],
                "run_mode": parsed["run"],
                "base_seed": int(parsed["base_seed"]) if parsed.get("base_seed") else int(parsed["cp_seed"]),
                "cp_seed": int(parsed["cp_seed"]),
            }
    return [jobs[k] for k in sorted(jobs)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", default="results/baseline_all_v2")
    parser.add_argument("--out-dir", default="results/baseline_all_replacement_aci_agaci_full_20260605")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    baseline_dir = PROJECT_ROOT / args.baseline_dir
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    conformal_csv = out_dir / "conformal_replacement.csv"
    adaptive_csv = out_dir / "adaptive_replacement.csv"
    progress_csv = out_dir / "progress.csv"

    # Disable heavy side effects. The metric computation is unchanged.
    exp_mod.plot_prediction_intervals = _no_plot
    exp_mod.plot_series = _no_plot
    exp_mod.ResultLogger.to_excel = lambda self: {}

    original_argv = sys.argv[:]
    sys.argv = ["rerun_aci_agaci_baseline_keys.py"]
    base_args = get_args()
    sys.argv = original_argv

    base_args.calib_print_every = 0
    base_args.test_print_every = 0
    base_args.dynamics_stride = 10**9
    base_args.use_gpu = 0
    base_args.num_workers = 0
    base_args.aci_gamma = 0.01
    base_args.aci_clip_alpha = 1
    base_args.agaci_gammas = OFFICIAL_31_GRID
    base_args.conformal_csv_path = str(conformal_csv.relative_to(PROJECT_ROOT))
    base_args.adaptive_csv_path = str(adaptive_csv.relative_to(PROJECT_ROOT))
    base_args.results_dir = str(out_dir.relative_to(PROJECT_ROOT))

    jobs = _load_jobs(baseline_dir / "conformal_results.csv")
    if args.limit is not None:
        jobs = jobs[: args.limit]

    done = _read_done(conformal_csv)
    total = len(jobs)
    completed_now = 0

    for idx, job in enumerate(jobs, 1):
        key = (job["setting"], job["cp_mode"], job["target_coverage"])
        if key in done:
            continue

        run_args = copy.deepcopy(base_args)
        run_args.data_path = _data_path(job["data"])
        run_args.cache_path = _cache_path(job["data"], job["model"], str(job["base_seed"]))
        run_args.x_lag = job["lags"]
        run_args.lags = job["lags"]
        run_args.base_model = job["model"]
        run_args.alpha = job["alpha"]
        run_args.run_mode = job["run_mode"]
        run_args.cp_mode = job["cp_mode"]
        run_args.seed = job["cp_seed"]
        run_args.comment = (
            f"official_full_replacement_{job['data']}_{job['model']}_{job['cp_mode']}"
            f"_base{job['base_seed']}_cp{job['cp_seed']}_tc{job['target_coverage']}"
        )

        log_path = log_dir / (
            f"{idx:04d}_{job['data']}_{job['model']}_{job['cp_mode']}"
            f"_base{job['base_seed']}_cp{job['cp_seed']}_tc{job['target_coverage']}.log"
        )
        with log_path.open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            exp = ExpConformal(run_args)
            exp.run(setting=job["setting"])

        done.add(key)
        completed_now += 1
        with progress_csv.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([idx, total, job["setting"], job["cp_mode"], job["target_coverage"]])

        if completed_now % 25 == 0:
            print(f"[Progress] completed_now={completed_now} done={len(done)}/{total}", flush=True)

    print(f"[Done] completed_now={completed_now} done={len(done)}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
