#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from multiprocessing import Process
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_EXP = ROOT / "run_exp.py"
REBUILD_BASELINES = ROOT / "scripts" / "rebuild_baseline_results.py"

SETTING_RE = re.compile(
    r"^(?P<dataset>.+?\.csv)_lags(?P<lags>\d+)_model(?P<base_model>.+?)"
    r"_cp(?P<cp_mode>.+?)_mode(?P<run_mode>.+?)(?:_base(?P<base_seed>\d+))?_seed(?P<seed>\d+)$"
)

DATASET_PATHS = {
    "exchange_rate": "time_series_library/dataset/exchange_rate/exchange_rate.csv",
    "ETTh1": "time_series_library/dataset/ETT-small/ETTh1.csv",
    "ETTh2": "time_series_library/dataset/ETT-small/ETTh2.csv",
    "ETTm1": "time_series_library/dataset/ETT-small/ETTm1.csv",
    "ETTm2": "time_series_library/dataset/ETT-small/ETTm2.csv",
    "weather": "time_series_library/dataset/weather/weather.csv",
}
NEW_DATASET_MANIFEST = ROOT / "time_series_library" / "dataset" / "new_benchmarks" / "manifest.csv"


def load_new_dataset_paths() -> dict[str, str]:
    if not NEW_DATASET_MANIFEST.exists():
        return {}
    try:
        manifest = pd.read_csv(NEW_DATASET_MANIFEST, dtype=str).fillna("")
    except Exception:
        return {}
    if not {"dataset", "data_path"}.issubset(manifest.columns):
        return {}
    return {
        str(row.dataset): str(row.data_path)
        for row in manifest.itertuples(index=False)
        if str(row.dataset).strip() and str(row.data_path).strip()
    }


DATASET_PATHS.update(load_new_dataset_paths())


@dataclass(frozen=True)
class Template:
    dataset: str
    lags: int
    base_model: str
    cp_mode: str
    run_mode: str
    base_seed: int | None = None


@dataclass(frozen=True)
class Job:
    template: Template
    seed: int
    target: float


def target_key(target: float) -> str:
    return f"{target:.12g}"


def alpha_for_target(target: float) -> str:
    return f"{1.0 - target:.12g}"


def build_setting(job: Job) -> str:
    t = job.template
    base_part = f"_base{t.base_seed}" if t.base_seed is not None else ""
    return f"{t.dataset}_lags{t.lags}_model{t.base_model}_cp{t.cp_mode}_mode{t.run_mode}{base_part}_seed{job.seed}"


def dataset_to_path(dataset: str) -> str:
    key = dataset[:-4] if dataset.lower().endswith(".csv") else dataset
    if dataset in DATASET_PATHS:
        return DATASET_PATHS[dataset]
    if key in DATASET_PATHS:
        return DATASET_PATHS[key]
    if NEW_DATASET_MANIFEST.exists():
        manifest = pd.read_csv(NEW_DATASET_MANIFEST, dtype=str).fillna("")
        hit = manifest[manifest["dataset"].eq(key)]
        if not hit.empty and hit.iloc[0].get("data_path"):
            return str(hit.iloc[0]["data_path"])
    return key


def dataset_aliases(dataset: str) -> list[str]:
    key = dataset.lower().removesuffix(".csv")
    return {"exchange_rate": ["exchange_rate", "exchange"]}.get(key, [key])


def build_cache_index(seeds: Iterable[int]) -> dict[tuple[int, str, str], Path]:
    out: dict[tuple[int, str, str], Path] = {}
    setting_re = re.compile(
        r"^long_term_forecast_(?P<dataset>.+?)_cache_(?P<model>[^_]+)_.+$",
        re.IGNORECASE,
    )
    for seed in seeds:
        root = ROOT / f"forecast_cache_seed{seed}"
        if not root.exists():
            continue
        for path in root.rglob("forecast_full.npz"):
            m = setting_re.match(path.parent.name)
            if not m:
                continue
            out[(seed, m.group("dataset").lower(), m.group("model").lower())] = path
    return out


def discover_cache(index: dict[tuple[int, str, str], Path], seed: int, dataset: str, model: str) -> Path | None:
    model_l = model.lower()
    for alias in dataset_aliases(dataset):
        hit = index.get((seed, alias, model_l))
        if hit is not None:
            return hit
    return None


def load_existing_keys(paths: Iterable[Path]) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for path in paths:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, dtype=str).fillna("")
        except Exception:
            continue
        needed = {"setting", "cp_mode", "target_coverage"}
        if not needed.issubset(df.columns):
            continue
        for row in df[["setting", "cp_mode", "target_coverage"]].drop_duplicates().itertuples(index=False):
            keys.add((str(row.setting), str(row.cp_mode), str(row.target_coverage)))
    return keys


def parse_templates(
    results_csv: Path,
    datasets: set[str],
    models: set[str],
    cp_modes: set[str],
    base_seeds: set[int] | None = None,
) -> list[Template]:
    df = pd.read_csv(results_csv, dtype=str).fillna("")
    df = df[df["target_coverage"].eq("0.9") & df["cp_mode"].isin(cp_modes)]
    out: list[Template] = []
    seen: set[tuple[str, int, str, str, str]] = set()
    for row in df.itertuples(index=False):
        m = SETTING_RE.match(str(row.setting))
        if not m:
            continue
        gd = m.groupdict()
        if gd["dataset"] not in datasets or gd["base_model"] not in models:
            continue
        base_seed = int(gd["base_seed"]) if gd.get("base_seed") else None
        if base_seeds is not None and base_seed not in base_seeds:
            continue
        key = (gd["dataset"], int(gd["lags"]), gd["base_model"], gd["cp_mode"], gd["run_mode"], base_seed)
        if key in seen:
            continue
        seen.add(key)
        out.append(Template(*key))
    return out


def write_worker_status(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def run_worker(worker_id: int, jobs: list[Job], cache_index: dict[tuple[int, str, str], Path], supplement_dir: Path) -> None:
    shard_dir = supplement_dir / "shards" / f"worker_{worker_id:02d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    conformal_csv = shard_dir / "conformal_results.csv"
    adaptive_csv = shard_dir / "adaptive_conformal_results.csv"
    log_dir = shard_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    status_path = shard_dir / "status.log"
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/tmp/cprl_mpl")
    env.setdefault("CUDA_VISIBLE_DEVICES", "")

    done = 0
    for job in jobs:
        setting = build_setting(job)
        cache_seed = job.template.base_seed if job.template.base_seed is not None else job.seed
        cache_path = discover_cache(cache_index, cache_seed, job.template.dataset, job.template.base_model)
        if cache_path is None:
            write_worker_status(status_path, f"SKIP cache_missing {setting} target={target_key(job.target)}")
            continue
        data_path = dataset_to_path(job.template.dataset)
        log_path = log_dir / f"{setting}_target{target_key(job.target)}.log"
        cmd = [
            sys.executable,
            str(RUN_EXP),
            "--data_path", data_path,
            "--cache_path", str(cache_path),
            "--x_lag", str(job.template.lags),
            "--lags", str(job.template.lags),
            "--base_model", job.template.base_model,
            "--cp_mode", job.template.cp_mode,
            "--run_mode", job.template.run_mode,
            "--alpha", alpha_for_target(job.target),
            "--seed", str(job.seed),
            "--results_dir", str(shard_dir),
            "--conformal_csv_path", str(conformal_csv),
            "--adaptive_csv_path", str(adaptive_csv),
        ]
        if job.template.base_seed is not None:
            cmd.extend(["--base_seed", str(job.template.base_seed)])
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            write_worker_status(status_path, f"FAIL rc={proc.returncode} {setting} target={target_key(job.target)} log={log_path}")
            raise SystemExit(proc.returncode)
        done += 1
        if done % 100 == 0:
            write_worker_status(status_path, f"DONE {done}/{len(jobs)}")
    write_worker_status(status_path, f"COMPLETE {done}/{len(jobs)}")


def concat_csv(paths: list[Path], out_path: Path, dedupe_cols: list[str]) -> None:
    frames = []
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            frames.append(pd.read_csv(path))
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    keep_cols = [c for c in dedupe_cols if c in df.columns]
    if keep_cols:
        df = df.drop_duplicates(subset=keep_cols, keep="last")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", default="results/baseline_all")
    parser.add_argument("--supplement-dir", default="results/baseline_all_missing_targets")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--cp-modes", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--base-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--targets", nargs="+", type=float, default=[0.8, 0.85, 0.95, 0.99])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    baseline_dir = (ROOT / args.baseline_dir).resolve()
    supplement_dir = (ROOT / args.supplement_dir).resolve()
    supplement_dir.mkdir(parents=True, exist_ok=True)

    main_conformal = baseline_dir / "conformal_results.csv"
    main_adaptive = baseline_dir / "adaptive_conformal_results.csv"
    shard_conformal = sorted(supplement_dir.glob("shards/worker_*/conformal_results.csv"))
    shard_adaptive = sorted(supplement_dir.glob("shards/worker_*/adaptive_conformal_results.csv"))

    if not args.merge_only:
        existing_keys = load_existing_keys([main_conformal, *shard_conformal])
        base_seed_filter = set(args.base_seeds) if args.base_seeds is not None else None
        templates = parse_templates(main_conformal, set(args.datasets), set(args.models), set(args.cp_modes), base_seed_filter)
        jobs: list[Job] = []
        for target in args.targets:
            if abs(target - 0.9) < 1e-12:
                continue
            for seed in args.seeds:
                for template in templates:
                    job = Job(template, seed, target)
                    key = (build_setting(job), template.cp_mode, target_key(target))
                    if key not in existing_keys:
                        jobs.append(job)
        if args.limit is not None:
            jobs = jobs[: args.limit]
        print(f"[Setup] templates={len(templates)} pending={len(jobs)} workers={args.workers}", flush=True)
        if args.dry_run:
            return 0

        cache_index = build_cache_index(args.seeds)
        n_workers = max(1, min(args.workers, len(jobs) or 1))
        chunk_size = math.ceil(len(jobs) / n_workers) if jobs else 0
        procs: list[Process] = []
        for worker_id in range(n_workers):
            chunk = jobs[worker_id * chunk_size : (worker_id + 1) * chunk_size]
            if not chunk:
                continue
            p = Process(target=run_worker, args=(worker_id, chunk, cache_index, supplement_dir))
            p.start()
            procs.append(p)
        failures = 0
        for p in procs:
            p.join()
            if p.exitcode:
                failures += 1
        if failures:
            raise SystemExit(f"{failures} worker(s) failed")

    shard_conformal = sorted(supplement_dir.glob("shards/worker_*/conformal_results.csv"))
    shard_adaptive = sorted(supplement_dir.glob("shards/worker_*/adaptive_conformal_results.csv"))
    backup_dir = supplement_dir / "premerge_backup"
    backup_dir.mkdir(exist_ok=True)
    for src in [main_conformal, main_adaptive]:
        if src.exists():
            shutil.copy2(src, backup_dir / src.name)
    concat_csv([main_conformal, *shard_conformal], main_conformal, ["setting", "cp_mode", "target_coverage"])
    concat_csv([main_adaptive, *shard_adaptive], main_adaptive, ["setting", "cp_mode", "target_coverage"])
    subprocess.run(
        [sys.executable, str(REBUILD_BASELINES), "--output_dir", str(baseline_dir.relative_to(ROOT)), "--rewrite_main_results"],
        cwd=str(ROOT),
        check=True,
    )
    print("[Done] merged and rebuilt baseline outputs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
