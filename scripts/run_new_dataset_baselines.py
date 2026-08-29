#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_CACHE = ROOT / "scripts" / "run_cache_experiments.py"
MANIFEST = ROOT / "time_series_library" / "dataset" / "new_benchmarks" / "manifest.csv"
DEFAULT_RESULTS_DIR = "results/baseline_all_v2"
DEFAULT_CP_MODES = ["acp", "aci", "agaci", "dfpi", "spci", "hopcpt", "nex", "cpid", "bellman"]
DEFAULT_MODELS = [
    "Autoformer",
    "Crossformer",
    "DLinear",
    "iTransformer",
    "PatchTST",
    "TimesNet",
    "Transformer",
    "TSMixer",
]


def load_datasets() -> list[str]:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")
    df = pd.read_csv(MANIFEST, dtype=str).fillna("")
    if "dataset" not in df.columns:
        raise ValueError(f"Missing dataset column in {MANIFEST}")
    datasets = [str(x) for x in df["dataset"].tolist() if str(x).strip()]
    if not datasets:
        raise ValueError(f"No datasets found in {MANIFEST}")
    return datasets


def main() -> int:
    parser = argparse.ArgumentParser(description="Run baseline CP experiments for all new benchmark datasets.")
    parser.add_argument("--datasets", nargs="*", default=[], help="Optional explicit dataset list. Default: manifest order.")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="Base forecasters to run.")
    parser.add_argument("--cp_modes", nargs="*", default=DEFAULT_CP_MODES, help="CP modes to run.")
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR, help="Output results directory.")
    parser.add_argument("--base_seed", type=int, default=2021, help="Base forecaster cache seed.")
    parser.add_argument("--cp_seed", type=int, default=2011, help="CP seed to evaluate.")
    parser.add_argument("--start_from", default="", help="Start from this dataset name (inclusive).")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of datasets processed.")
    args = parser.parse_args()

    datasets = args.datasets or load_datasets()
    if args.start_from:
        if args.start_from not in datasets:
            raise ValueError(f"start_from dataset not found: {args.start_from}")
        datasets = datasets[datasets.index(args.start_from):]
    if args.limit is not None:
        datasets = datasets[: args.limit]

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/private/tmp/cprl_mpl")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    total = len(datasets)
    print(f"[Setup] datasets={total} models={len(args.models)} cp_modes={len(args.cp_modes)} results_dir={args.results_dir}")
    for idx, dataset in enumerate(datasets, 1):
        print(f"[Dataset {idx}/{total}] {dataset}")
        cmd = [
            sys.executable,
            str(RUN_CACHE),
            "baseline",
            "--dataset",
            dataset,
            "--models",
            *args.models,
            "--cp_modes",
            *args.cp_modes,
            "--results_dir",
            args.results_dir,
            "--base_seed",
            str(args.base_seed),
            "--cp_seed",
            str(args.cp_seed),
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
        if proc.returncode != 0:
            print(f"[Error] dataset failed: {dataset} (exit={proc.returncode})", file=sys.stderr)
            return proc.returncode

    print("[Done] all datasets completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
