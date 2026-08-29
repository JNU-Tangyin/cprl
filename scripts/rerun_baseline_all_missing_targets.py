#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_EXP = ROOT / "run_exp.py"
REBUILD_BASELINES = ROOT / "scripts" / "rebuild_baseline_results.py"
LOG_DIR = ROOT / "experiment_logs"
COMMAND_LOG_DIR = LOG_DIR / "commands"

SETTING_RE = re.compile(
    r"^(?P<dataset>.+?\.csv)_lags(?P<lags>\d+)_model(?P<base_model>.+?)_cp(?P<cp_mode>.+?)_mode(?P<run_mode>.+?)_seed(?P<seed>\d+)$"
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
DEFAULT_DATASETS = [
    "ETTh1.csv",
    "ETTh2.csv",
    "ETTm1.csv",
    "ETTm2.csv",
    "exchange_rate.csv",
    "weather.csv",
    *[f"{name}.csv" for name in load_new_dataset_paths()],
]
DEFAULT_MODELS = [
    "Autoformer",
    "Crossformer",
    "DLinear",
    "iTransformer",
    "PatchTST",
    "Pyraformer",
    "TimesNet",
    "Transformer",
    "TSMixer",
]
DEFAULT_SEEDS = [2021, 2022, 2023, 2024, 2025]
_CACHE_INDEX: dict[int, dict[tuple[str, str], Path]] = {}


def dataset_aliases_for_lookup(dataset_l: str) -> list[str]:
    dataset_aliases = {
        "exchange_rate": ["exchange_rate", "exchange"],
    }
    return dataset_aliases.get(dataset_l, [dataset_l])


@dataclass(frozen=True)
class TemplateRow:
    setting: str
    dataset: str
    lags: int
    base_model: str
    cp_mode: str
    run_mode: str
    seed: int


def build_setting(job: TemplateRow) -> str:
    return (
        f"{job.dataset}_lags{job.lags}_model{job.base_model}"
        f"_cp{job.cp_mode}_mode{job.run_mode}_seed{job.seed}"
    )


def load_existing_result_keys(results_csv: Path) -> set[tuple[str, str, str]]:
    if not results_csv.exists():
        return set()
    df = pd.read_csv(results_csv, dtype=str).fillna("")
    needed = {"setting", "cp_mode", "target_coverage"}
    if not needed.issubset(df.columns):
        return set()
    keys: set[tuple[str, str, str]] = set()
    for _, row in df[["setting", "cp_mode", "target_coverage"]].drop_duplicates().iterrows():
        keys.add((str(row["setting"]), str(row["cp_mode"]), str(row["target_coverage"])))
    return keys


def dataset_to_path(dataset: str) -> str:
    dataset = dataset.strip()
    dataset_key = dataset[:-4] if dataset.lower().endswith(".csv") else dataset
    if dataset in DATASET_PATHS:
        return DATASET_PATHS[dataset]
    if dataset_key in DATASET_PATHS:
        return DATASET_PATHS[dataset_key]
    if NEW_DATASET_MANIFEST.exists():
        try:
            manifest = pd.read_csv(NEW_DATASET_MANIFEST, dtype=str).fillna("")
            hit = manifest[manifest["dataset"] == dataset_key]
            if not hit.empty and hit.iloc[0].get("data_path"):
                return str(hit.iloc[0]["data_path"])
        except Exception:
            pass
    return dataset_key


def discover_cache(base_seed: int, dataset: str, model: str) -> Optional[Path]:
    cached = _CACHE_INDEX.get(base_seed)
    if cached is not None:
        dataset_l = dataset.lower().removesuffix(".csv")
        model_l = model.lower()
        for dataset_key in dataset_aliases_for_lookup(dataset_l):
            hit = cached.get((dataset_key, model_l))
            if hit is not None:
                return hit

    root = ROOT / f"forecast_cache_seed{base_seed}"
    if not root.exists():
        return None

    dataset_l = dataset.lower()
    dataset_l = dataset_l.removesuffix(".csv")
    dataset_keys = dataset_aliases_for_lookup(dataset_l)
    model_l = model.lower()
    setting_re = re.compile(
        r"^long_term_forecast_(?P<dataset>.+?)_cache_(?P<model>[^_]+)_.+$",
        re.IGNORECASE,
    )
    candidates: List[Path] = []
    index: dict[tuple[str, str], Path] = {}
    for path in root.rglob("forecast_full.npz"):
        parent_name = path.parent.name
        m = setting_re.match(parent_name)
        if not m:
            continue

        parsed_dataset = m.group("dataset").lower()
        parsed_model = m.group("model").lower()
        key = (parsed_dataset, parsed_model)
        index[key] = path
        if parsed_dataset in dataset_keys and parsed_model == model_l:
            candidates.append(path)

    _CACHE_INDEX[base_seed] = index
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: (len(str(p)), str(p)))[0]


def parse_templates(results_csv: Path, cp_modes: set[str], datasets: set[str], models: set[str]) -> list[TemplateRow]:
    df = pd.read_csv(results_csv, dtype=str).fillna("")
    df = df[df["cp_mode"].isin(cp_modes) & (df["target_coverage"] == "0.9")]
    allowed_datasets = set(datasets)
    allowed_models = set(models)
    templates: list[TemplateRow] = []
    seen: set[tuple[str, str, str, str, int]] = set()
    for _, row in df.iterrows():
        setting = str(row["setting"])
        m = SETTING_RE.match(setting)
        if not m:
            continue
        parsed = m.groupdict()
        if not parsed["dataset"]:
            continue
        dataset = parsed["dataset"]
        if dataset not in allowed_datasets:
            continue
        model = parsed["base_model"]
        if model not in allowed_models:
            continue
        key = (dataset, model, parsed["cp_mode"], parsed["run_mode"], int(parsed["lags"]))
        if key in seen:
            continue
        seen.add(key)
        templates.append(
            TemplateRow(
                setting=setting,
                dataset=dataset,
                lags=int(parsed["lags"]),
                base_model=model,
                cp_mode=parsed["cp_mode"],
                run_mode=parsed["run_mode"],
                seed=int(parsed["seed"]),
            )
        )
    return templates


def target_to_alpha(target_coverage: float) -> float:
    return float(f"{1.0 - target_coverage:.12g}")


def run_one(template: TemplateRow, target_coverage: float, results_dir: Path) -> None:
    cache_path = discover_cache(template.seed, template.dataset, template.base_model)
    if cache_path is None:
        print(f"[Skip] cache missing for {template.dataset} / {template.base_model} / seed{template.seed}")
        return

    alpha = target_to_alpha(target_coverage)
    data_path = dataset_to_path(template.dataset)
    data_path_abs = (ROOT / data_path).resolve() if not Path(data_path).is_absolute() else Path(data_path)
    if not data_path_abs.exists():
        print(f"[Skip] data path missing for {template.dataset}: {data_path}")
        return
    conformal_csv = results_dir / "conformal_results.csv"
    adaptive_csv = results_dir / "adaptive_conformal_results.csv"

    cmd = [
        sys.executable,
        str(RUN_EXP),
        "--data_path", str(data_path),
        "--cache_path", str(cache_path),
        "--x_lag", str(template.lags),
        "--lags", str(template.lags),
        "--base_model", template.base_model,
        "--cp_mode", template.cp_mode,
        "--run_mode", template.run_mode,
        "--alpha", str(alpha),
        "--seed", str(template.seed),
        "--results_dir", str(results_dir),
        "--conformal_csv_path", str(conformal_csv),
        "--adaptive_csv_path", str(adaptive_csv),
    ]

    cmd_log_dir = COMMAND_LOG_DIR
    cmd_log_dir.mkdir(parents=True, exist_ok=True)
    log_name = (
        f"baseline_all_missing_{template.dataset}_{template.base_model}_{template.cp_mode}"
        f"_seed{template.seed}_tc{target_coverage}.log"
    )
    log_path = cmd_log_dir / log_name

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/private/tmp/cprl_mpl")
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"job failed: {template.setting} target={target_coverage} see {log_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rerun missing target coverage rows for baseline_all.")
    parser.add_argument("--baseline-dir", default="results/baseline_all")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help="Datasets to rerun. Default: the 6 non-FRED/non-BCI datasets.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Forecasters to rerun. Default: 9 baseline forecasters.",
    )
    parser.add_argument(
        "--cp_modes",
        nargs="*",
        default=["bellman", "cpid", "aci", "agaci"],
        help="CP modes to rerun. Default: bellman cpid aci agaci",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seeds to rerun. Default: 2021 2022 2023 2024 2025",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        type=float,
        default=[0.8, 0.85, 0.95, 0.99],
        help="Target coverages to rerun. Default: 0.8 0.85 0.95 0.99",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    baseline_dir = (ROOT / args.baseline_dir).resolve()
    if not baseline_dir.exists():
        raise FileNotFoundError(f"baseline dir not found: {baseline_dir}")

    existing_keys = load_existing_result_keys(baseline_dir / "conformal_results.csv")
    templates = parse_templates(
        baseline_dir / "conformal_results.csv",
        set(args.cp_modes),
        set(args.datasets),
        set(args.models),
    )
    if args.limit is not None:
        templates = templates[: args.limit]
    if not templates:
        raise RuntimeError("no template rows found")

    jobs: list[tuple[TemplateRow, float]] = []
    for target in args.targets:
        if abs(target - 0.9) < 1e-12:
            continue
        for seed in args.seeds:
            for template in templates:
                job = TemplateRow(
                    setting=template.setting,
                    dataset=template.dataset,
                    lags=template.lags,
                    base_model=template.base_model,
                    cp_mode=template.cp_mode,
                    run_mode=template.run_mode,
                    seed=seed,
                )
                key = (build_setting(job), job.cp_mode, str(target))
                if key in existing_keys:
                    continue
                jobs.append((job, target))

    total = len(jobs)
    print(
        f"[Setup] templates={len(templates)} datasets={len(args.datasets)} models={len(args.models)} "
        f"cp_modes={args.cp_modes} seeds={args.seeds} targets={args.targets} existing={len(existing_keys)} "
        f"pending={total}"
    )
    done = 0
    for job, target in jobs:
        print(
            f"[Run {done+1}/{total}] dataset={job.dataset} model={job.base_model} "
            f"cp_mode={job.cp_mode} seed={job.seed} target={target}",
            flush=True,
        )
        run_one(job, target, baseline_dir)
        done += 1

    print("[Rebuild] dedupe and rewrite main result files")
    subprocess.run(
        [
            sys.executable,
            str(REBUILD_BASELINES),
            "--output_dir",
            str(baseline_dir.relative_to(ROOT)),
            "--rewrite_main_results",
        ],
        cwd=str(ROOT),
        check=True,
    )
    print(f"[Done] completed={done}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
