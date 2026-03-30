from __future__ import annotations

import os
import csv
import json
import random
import argparse
from itertools import product
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.backends

from exp.exp_conformal import ExpConformal


# ============================================================
# Fixed experiment setup
# ============================================================

# DATASETS = ["ETTh1", "exchange"]
DATASETS = ["exchange"]
BASE_MODEL = "Transformer"

# base forecaster seed is fixed
FORECAST_SEED = 2021
FORECAST_CACHE_ROOT = f"./forecast_cache_seed{FORECAST_SEED}"

# ACP / CEM seeds
ACP_SEEDS = [1, 2, 3, 4, 5]

# OFAT grid
OFAT_PARAM_GRID = {
    "new_regime_threshold": [1.6, 1.9, 2.2, 2.5, 2.8, 3.1],
    "min_state_duration":   [1, 3, 5, 8, 12, 16],
    "lambda_spectral":      [0.0, 0.1, 0.3, 0.5, 0.8, 1.2],
    "lambda_cov":           [0.5, 1.0, 2.5, 5.0, 10.0, 20.0],
    "cem_lr":               [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
    "cem_pop":              [8, 16, 24, 32, 48],
}

# interaction grid
INTERACTION_GRID = {
    "regime_interaction": {
        "param_x": "new_regime_threshold",
        "values_x": [1.8, 2.2, 2.6, 3.0],
        "param_y": "min_state_duration",
        "values_y": [1, 5, 10, 15],
    },
    "spectral_control_interaction": {
        "param_x": "lambda_spectral",
        "values_x": [0.0, 0.3, 0.6, 1.0],
        "param_y": "lambda_cov",
        "values_y": [0.5, 2.5, 5.0, 10.0],
    },
}

# output dirs
RESULT_ROOT = "./results/sensitivity_analysis"
RAW_DIR = os.path.join(RESULT_ROOT, "raw_runs")
SUMMARY_DIR = os.path.join(RESULT_ROOT, "summary")

CONFORMAL_CSV = os.path.join(RESULT_ROOT, "conformal_results.csv")
ADAPTIVE_CSV = os.path.join(RESULT_ROOT, "adaptive_conformal_results.csv")


# ============================================================
# Utilities
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    """
    Match run.py logic:
    random + numpy + torch are seeded before ExpConformal runs.
    This controls ACP/CEM randomness while forecast cache is fixed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(args) -> None:
    """
    Match run.py logic.
    For sensitivity analysis, CPU is safest by default.
    """
    if args.gpu_type == "cpu":
        args.device = torch.device("cpu")
        return

    if args.gpu_type == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = torch.device("mps")
        else:
            args.device = torch.device("cpu")
        return

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device(f"cuda:{args.gpu}")
    else:
        args.device = torch.device("cpu")


def infer_dataset_name_from_cache(cache_path: str) -> str:
    parent = Path(cache_path).resolve().parent.name
    parts = parent.split("_")

    if "cache" in parts:
        idx = parts.index("cache")
        if idx - 1 >= 0:
            return parts[idx - 1]

    return parent


def infer_base_model_from_cache(cache_path: str) -> str:
    parent = Path(cache_path).resolve().parent.name
    parts = parent.split("_")

    if "cache" in parts:
        idx = parts.index("cache")
        if idx + 1 < len(parts):
            return parts[idx + 1]

    return "cache"


def resolve_cache_path(dataset: str, base_model: str, forecast_seed: int = FORECAST_SEED) -> str:
    """
    Find the unique forecast_full.npz under forecast_cache_seed2021
    for the requested dataset and base forecaster.
    """
    root = Path(f"./forecast_cache_seed{forecast_seed}")
    if not root.exists():
        raise FileNotFoundError(f"Cache root not found: {root}")

    matches: List[str] = []
    token = f"_{dataset}_cache_{base_model}_"

    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        if token in sub.name:
            npz_path = sub / "forecast_full.npz"
            if npz_path.exists():
                matches.append(str(npz_path))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No cache found for dataset={dataset}, base_model={base_model} under {root}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple caches found for dataset={dataset}, base_model={base_model}: {matches}"
        )
    return matches[0]


def cleanup_old_setting_rows(csv_path: str, setting: str) -> None:
    """
    Remove existing rows with the same setting to avoid duplicates
    when re-running experiments.
    """
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    if "setting" not in df.columns:
        return
    df = df[df["setting"] != setting].copy()
    df.to_csv(csv_path, index=False)


def extract_latest_row(csv_path: str, setting: str) -> Dict[str, Any]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    sub = df[df["setting"] == setting].copy()
    if sub.empty:
        raise ValueError(f"Setting not found in {csv_path}: {setting}")
    return sub.iloc[-1].to_dict()


def safe_std(vals: List[float]) -> float:
    if len(vals) <= 1:
        return 0.0
    return float(np.std(vals, ddof=1))


def summarize_metric_group(df: pd.DataFrame, group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(group_cols, dropna=False)

    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {k: v for k, v in zip(group_cols, keys)}
        row["n_runs"] = len(sub)

        for m in metric_cols:
            vals = sub[m].dropna().astype(float).tolist()
            row[f"{m}_mean"] = float(np.mean(vals)) if len(vals) > 0 else np.nan
            row[f"{m}_std"] = safe_std(vals) if len(vals) > 0 else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def save_manifest() -> None:
    ensure_dir(RESULT_ROOT)
    manifest = {
        "datasets": DATASETS,
        "base_model": BASE_MODEL,
        "forecast_seed": FORECAST_SEED,
        "forecast_cache_root": FORECAST_CACHE_ROOT,
        "acp_seeds": ACP_SEEDS,
        "ofat_param_grid": OFAT_PARAM_GRID,
        "interaction_grid": INTERACTION_GRID,
    }
    path = os.path.join(RESULT_ROOT, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


# ============================================================
# Build args
# ============================================================

def build_args(dataset: str, acp_seed: int, overrides: Dict[str, Any]):
    """
    Reproduce run.py argument style, but inside the sensitivity script.
    """
    cache_path = resolve_cache_path(dataset, BASE_MODEL, FORECAST_SEED)

    parser = argparse.ArgumentParser(description="Sensitivity Analysis Runner")

    # task / protocol
    parser.add_argument("--task", type=str, default="conformal")
    parser.add_argument("--run_mode", type=str, default="online")
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--seed", type=int, default=acp_seed)

    # naming / output
    parser.add_argument("--exp_name", type=str, default="sensitivity")
    parser.add_argument("--des", type=str, default="debug")
    parser.add_argument("--results_dir", type=str, default=RESULT_ROOT)
    parser.add_argument("--comment", type=str, default="sensitivity_analysis")

    # cache input
    parser.add_argument("--cache_path", type=str, default=cache_path)
    parser.add_argument("--x_lag", type=int, default=24)

    # conformal
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--cp_mode", type=str, default="acp")

    parser.add_argument("--spectral_window", type=int, default=64)
    parser.add_argument("--n_latent_states", type=int, default=3)
    parser.add_argument("--calib_window_size", type=int, default=200)
    parser.add_argument("--lambda_spectral", type=float, default=0.5)
    parser.add_argument("--min_calib_size", type=int, default=30)

    # diagnostics
    parser.add_argument("--unc_window", type=int, default=256)
    parser.add_argument("--calib_window", type=int, default=200)
    parser.add_argument("--calib_print_every", type=int, default=200)
    parser.add_argument("--test_window", type=int, default=100)
    parser.add_argument("--test_print_every", type=int, default=100)
    parser.add_argument("--dynamics_stride", type=int, default=1)
    parser.add_argument("--worst_window", type=int, default=100)

    # device
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu_type", type=str, default="cpu", choices=["cuda", "mps", "cpu"])

    # ablation
    parser.add_argument("--ablation", type=int, default=0)
    parser.add_argument("--ablation_mode", type=str, default="M0")

    # ===== ACP config fields consumed by build_conformal_predictor =====
    parser.add_argument("--initial_alpha", type=float, default=0.1)
    parser.add_argument("--target_coverage", type=float, default=0.9)
    parser.add_argument("--window_size", type=int, default=64)

    parser.add_argument("--max_regimes", type=int, default=8)
    parser.add_argument("--new_regime_threshold", type=float, default=2.2)
    parser.add_argument("--new_regime_patience", type=int, default=3)

    parser.add_argument("--sticky_bonus", type=float, default=0.5)
    parser.add_argument("--min_state_duration", type=int, default=5)

    parser.add_argument("--ewma_beta", type=float, default=0.94)
    parser.add_argument("--jump_q", type=float, default=0.95)
    parser.add_argument("--feature_ema", type=float, default=0.05)

    parser.add_argument("--min_regime_calib_size", type=int, default=20)
    parser.add_argument("--min_spectral_size", type=int, default=30)
    parser.add_argument("--min_regime_eval_size", type=int, default=20)
    parser.add_argument("--min_regime_cov_size", type=int, default=20)

    parser.add_argument("--eval_window_size", type=int, default=150)

    parser.add_argument("--coverage_window", type=int, default=50)
    parser.add_argument("--lambda_cov", type=float, default=5.0)

    parser.add_argument("--k_update_every", type=int, default=20)
    parser.add_argument("--k_min", type=float, default=1e-3)
    parser.add_argument("--k_max", type=float, default=100.0)
    parser.add_argument("--k_fallback", type=float, default=1.0)

    parser.add_argument("--alpha_min", type=float, default=0.01)
    parser.add_argument("--alpha_max", type=float, default=0.3)
    parser.add_argument("--cem_pop", type=int, default=16)
    parser.add_argument("--cem_elite_frac", type=float, default=0.25)
    parser.add_argument("--cem_noise", type=float, default=0.02)
    parser.add_argument("--cem_lr", type=float, default=0.3)
    parser.add_argument("--cem_n_iters", type=int, default=3)

    parser.add_argument("--use_spectral", type=bool, default=True)
    parser.add_argument("--use_regime", type=bool, default=True)
    parser.add_argument("--use_cem", type=bool, default=True)
    parser.add_argument("--width_weight", type=float, default=1.0)

    # optional result paths
    parser.add_argument("--conformal_csv_path", type=str, default=CONFORMAL_CSV)
    parser.add_argument("--adaptive_csv_path", type=str, default=ADAPTIVE_CSV)

    args = parser.parse_args([])

    # apply overrides from sensitivity sweep
    for k, v in overrides.items():
        setattr(args, k, v)

    # mimic run.py post-processing
    args.dataset_name = infer_dataset_name_from_cache(args.cache_path)
    args.base_model = infer_base_model_from_cache(args.cache_path)

    return args


def make_setting(args, ii: int = 0) -> str:
    """
    Match run.py setting rule.
    """
    return "{}_{}_model{}_cp{}_a{}_cw{}_xlag{}_seed{}_mode{}_{}_{}".format(
        args.dataset_name,
        args.exp_name,
        args.base_model,
        args.cp_mode,
        args.alpha,
        args.calib_window_size,
        args.x_lag,
        args.seed,
        args.run_mode,
        f"{args.des}-{ii}",
        getattr(args, "ablation_mode", "M0"),
    )


# ============================================================
# Single run
# ============================================================

def run_single_setting(
    *,
    dataset: str,
    acp_seed: int,
    overrides: Dict[str, Any],
    des_tag: str,
) -> Dict[str, Any]:
    set_seed(acp_seed)

    args = build_args(dataset, acp_seed, overrides)
    args.exp_name = "sensitivity"
    args.des = des_tag

    select_device(args)

    setting = make_setting(args, ii=0)

    cleanup_old_setting_rows(CONFORMAL_CSV, setting)
    cleanup_old_setting_rows(ADAPTIVE_CSV, setting)

    exp = ExpConformal(args)
    exp.run(setting)

    conf_row = extract_latest_row(CONFORMAL_CSV, setting)
    adp_row = extract_latest_row(ADAPTIVE_CSV, setting)

    result = {
        "setting": setting,
        "dataset": dataset,
        "base_model": args.base_model,
        "forecast_seed": FORECAST_SEED,
        "acp_seed": acp_seed,
        "cache_path": args.cache_path,
    }
    result.update(overrides)

    for key in [
        "coverage",
        "coverage_gap",
        "avg_width",
        "ces",
        "rcs",
        "point_mse",
        "point_mae",
    ]:
        if key in conf_row:
            result[key] = conf_row[key]

    for key in [
        "worst_window_coverage",
        "width_step_mean",
        "width_std",
        "control_alpha_step_mean",
        "control_alpha_std",
    ]:
        if key in adp_row:
            result[key] = adp_row[key]

    return result


# ============================================================
# OFAT
# ============================================================

def run_ofat() -> None:
    print("========== Running OFAT sensitivity ==========")
    rows = []

    for dataset in DATASETS:
        for param_name, param_values in OFAT_PARAM_GRID.items():
            for param_value in param_values:
                overrides = {param_name: param_value}

                for acp_seed in ACP_SEEDS:
                    des_tag = f"ofat_{param_name}_{param_value}"
                    print(f"[OFAT] dataset={dataset} | {param_name}={param_value} | acp_seed={acp_seed}")

                    row = run_single_setting(
                        dataset=dataset,
                        acp_seed=acp_seed,
                        overrides=overrides,
                        des_tag=des_tag,
                    )
                    row["experiment_type"] = "ofat"
                    row["param_name"] = param_name
                    row["param_value"] = param_value
                    rows.append(row)

    raw_df = pd.DataFrame(rows)
    ensure_dir(RAW_DIR)
    raw_path = os.path.join(RAW_DIR, "ofat_raw.csv")
    raw_df.to_csv(raw_path, index=False)

    metric_cols = [
        "coverage",
        "coverage_gap",
        "avg_width",
        "ces",
        "rcs",
        "point_mse",
        "point_mae",
        "worst_window_coverage",
        "width_step_mean",
        "width_std",
        "control_alpha_step_mean",
        "control_alpha_std",
    ]
    metric_cols = [c for c in metric_cols if c in raw_df.columns]

    summary_df = summarize_metric_group(
        raw_df,
        group_cols=["experiment_type", "dataset", "param_name", "param_value"],
        metric_cols=metric_cols,
    )
    ensure_dir(SUMMARY_DIR)
    summary_path = os.path.join(SUMMARY_DIR, "ofat_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"[OFAT] raw saved to: {raw_path}")
    print(f"[OFAT] summary saved to: {summary_path}")


# ============================================================
# Interaction
# ============================================================

def run_interactions() -> None:
    print("========== Running interaction sensitivity ==========")
    rows = []

    for dataset in DATASETS:
        for interaction_name, cfg in INTERACTION_GRID.items():
            param_x = cfg["param_x"]
            param_y = cfg["param_y"]

            for value_x, value_y in product(cfg["values_x"], cfg["values_y"]):
                overrides = {
                    param_x: value_x,
                    param_y: value_y,
                }

                for acp_seed in ACP_SEEDS:
                    des_tag = f"{interaction_name}_{param_x}_{value_x}_{param_y}_{value_y}"
                    print(
                        f"[INT] dataset={dataset} | {interaction_name} | "
                        f"{param_x}={value_x} | {param_y}={value_y} | acp_seed={acp_seed}"
                    )

                    row = run_single_setting(
                        dataset=dataset,
                        acp_seed=acp_seed,
                        overrides=overrides,
                        des_tag=des_tag,
                    )
                    row["experiment_type"] = "interaction"
                    row["interaction_name"] = interaction_name
                    row["param_x"] = param_x
                    row["value_x"] = value_x
                    row["param_y"] = param_y
                    row["value_y"] = value_y
                    rows.append(row)

    raw_df = pd.DataFrame(rows)
    ensure_dir(RAW_DIR)
    raw_path = os.path.join(RAW_DIR, "interaction_raw.csv")
    raw_df.to_csv(raw_path, index=False)

    metric_cols = [
        "coverage",
        "coverage_gap",
        "avg_width",
        "ces",
        "rcs",
        "point_mse",
        "point_mae",
        "worst_window_coverage",
        "width_step_mean",
        "width_std",
        "control_alpha_step_mean",
        "control_alpha_std",
    ]
    metric_cols = [c for c in metric_cols if c in raw_df.columns]

    summary_df = summarize_metric_group(
        raw_df,
        group_cols=[
            "experiment_type",
            "interaction_name",
            "dataset",
            "param_x",
            "value_x",
            "param_y",
            "value_y",
        ],
        metric_cols=metric_cols,
    )
    ensure_dir(SUMMARY_DIR)
    summary_path = os.path.join(SUMMARY_DIR, "interaction_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"[Interaction] raw saved to: {raw_path}")
    print(f"[Interaction] summary saved to: {summary_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    ensure_dir(RESULT_ROOT)
    ensure_dir(RAW_DIR)
    ensure_dir(SUMMARY_DIR)

    save_manifest()
    run_ofat()
    run_interactions()

    print("✅ Sensitivity analysis finished.")


if __name__ == "__main__":
    main()