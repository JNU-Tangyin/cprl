#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Iterable, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFORMAL_NAME = "conformal_results.csv"
ADAPTIVE_NAME = "adaptive_conformal_results.csv"
RESULT_KEY_COLS = ["setting", "cp_mode", "target_coverage"]

SETTING_RE = re.compile(
    r"^(?P<dataset>.+?\.csv)_lags\d+_model(?P<base_model>.+?)_cp(?P<cp_mode>.+?)_mode.+?"
    r"(?:_base(?P<base_seed>\d+))?_seed(?P<cp_seed>\d+)"
)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_first_csv(paths: Iterable[Path]) -> pd.DataFrame:
    for path in paths:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def result_key_cols(df: pd.DataFrame) -> List[str]:
    return [col for col in RESULT_KEY_COLS if col in df.columns]


def dedupe_by_result_key(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    use = [df for df in frames if not df.empty]
    if not use:
        return pd.DataFrame()
    merged = pd.concat(use, ignore_index=True)
    key_cols = result_key_cols(merged)
    if not key_cols:
        return merged
    return merged.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)


def ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = ""
    return out


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def parse_setting(setting: str) -> dict:
    m = SETTING_RE.match(str(setting))
    if not m:
        return {"dataset": "", "base_model": "", "cp_mode": "", "base_seed": "", "cp_seed": ""}
    out = m.groupdict()
    if not out.get("base_seed"):
        out["base_seed"] = out.get("cp_seed", "")
    return out


def rebuild_seed_summaries(out_dir: Path, conformal: pd.DataFrame, adaptive: pd.DataFrame) -> None:
    if conformal.empty:
        pd.DataFrame().to_csv(out_dir / "seed_summary_raw.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "seed_summary_agg.csv", index=False)
        return

    conf = conformal.copy()
    if "coverage_gap" in conf.columns:
        conf = conf.drop(columns=["coverage_gap"])

    if adaptive.empty:
        merged = conf.copy()
    else:
        adp = adaptive.copy()
        key_cols = result_key_cols(adp)
        if key_cols:
            adp = adp.drop_duplicates(subset=key_cols, keep="last")
        rename = {c: f"adaptive_{c}" for c in adp.columns if c not in RESULT_KEY_COLS}
        adp = adp.rename(columns=rename)
        merge_cols = [col for col in RESULT_KEY_COLS if col in conf.columns and col in adp.columns]
        merged = conf.merge(adp, on=merge_cols, how="left")
        merged["adaptive_setting"] = merged["setting"]
        merged["adaptive_cp_mode"] = merged["cp_mode"]
        merged["adaptive_target_coverage"] = merged["target_coverage"]

    parsed = merged["setting"].map(parse_setting).apply(pd.Series)
    merged["phase"] = "baseline"
    merged["dataset"] = parsed["dataset"].str.replace(".csv", "", regex=False)
    merged["base_seed"] = pd.to_numeric(parsed["base_seed"], errors="coerce")
    merged["cp_seed"] = pd.to_numeric(parsed["cp_seed"], errors="coerce")
    merged["cache_seed"] = merged["base_seed"]
    merged["base_model"] = parsed["base_model"]
    merged["ablation_mode"] = ""
    merged["sensitivity_param"] = ""
    merged["sensitivity_value"] = ""

    regime_dir = out_dir / "regime_metrics"
    merged["regime_metrics_path"] = merged["setting"].map(
        lambda s: str(regime_dir / f"{s}.csv") if (regime_dir / f"{s}.csv").exists() else ""
    )

    raw_cols = [
        "setting",
        "cp_mode",
        "target_coverage",
        "runtime_seconds",
        "coverage",
        "abs_coverage_gap",
        "under_coverage_gap",
        "over_coverage_gap",
        "coverage_bias",
        "coverage_bias_direction",
        "avg_width",
        "ces",
        "rcs",
        "point_mse",
        "point_mae",
        "comment",
        "adaptive_setting",
        "adaptive_cp_mode",
        "adaptive_target_coverage",
        "adaptive_runtime_seconds",
        "adaptive_worst_window_coverage",
        "adaptive_width_step_mean",
        "adaptive_width_std",
        "adaptive_control_alpha_step_mean",
        "adaptive_control_alpha_std",
        "adaptive_comment",
        "regime_metrics_path",
        "phase",
        "dataset",
        "cache_seed",
        "base_seed",
        "cp_seed",
        "base_model",
        "ablation_mode",
        "sensitivity_param",
        "sensitivity_value",
    ]
    merged = ensure_cols(merged, raw_cols)
    merged = merged[raw_cols]
    merged.to_csv(out_dir / "seed_summary_raw.csv", index=False)

    metric_cols = [
        "runtime_seconds",
        "coverage",
        "abs_coverage_gap",
        "under_coverage_gap",
        "over_coverage_gap",
        "coverage_bias",
        "avg_width",
        "ces",
        "rcs",
        "point_mse",
        "point_mae",
        "adaptive_runtime_seconds",
        "adaptive_worst_window_coverage",
        "adaptive_width_step_mean",
        "adaptive_width_std",
        "adaptive_control_alpha_step_mean",
        "adaptive_control_alpha_std",
    ]
    merged = coerce_numeric(merged, metric_cols)
    use_metrics = [c for c in metric_cols if c in merged.columns]
    agg = merged.groupby(
        ["dataset", "base_model", "cp_mode", "target_coverage"], dropna=False
    )[use_metrics].agg(["mean", "std"]).reset_index()
    agg.columns = [
        "_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col)
        for col in agg.columns
    ]
    agg.to_csv(out_dir / "seed_summary_agg.csv", index=False)
    agg.to_excel(out_dir / "seed_summary_agg.xlsx", index=False)


def rebuild_regime_summaries(out_dir: Path) -> None:
    regime_dir = out_dir / "regime_metrics"
    rows: List[pd.DataFrame] = []
    if regime_dir.exists():
        for path in sorted(regime_dir.glob("*.csv")):
            df = pd.read_csv(path)
            if df.empty or "setting" not in df.columns:
                continue
            parsed = parse_setting(str(df["setting"].iloc[0]))
            df["dataset"] = parsed["dataset"].replace(".csv", "")
            df["base_seed"] = int(parsed["base_seed"]) if parsed["base_seed"] else ""
            df["cp_seed"] = int(parsed["cp_seed"]) if parsed["cp_seed"] else ""
            df["cache_seed"] = df["base_seed"]
            df["base_model"] = parsed["base_model"]
            df["cp_mode"] = df["cp_mode"] if "cp_mode" in df.columns else parsed["cp_mode"]
            df["ablation_mode"] = ""
            df["sensitivity_param"] = ""
            df["sensitivity_value"] = ""
            rows.append(df)

    raw_path = out_dir / "regime_metrics_by_seed.csv"
    agg_path = out_dir / "regime_metrics_agg.csv"
    if not rows:
        pd.DataFrame().to_csv(raw_path, index=False)
        pd.DataFrame().to_csv(agg_path, index=False)
        return

    raw = pd.concat(rows, ignore_index=True)
    raw.to_csv(raw_path, index=False)

    metrics = [c for c in ["count", "coverage", "avg_width", "ces"] if c in raw.columns]
    agg = raw.groupby(
        [
            "dataset",
            "base_model",
            "cp_mode",
            "target_coverage",
            "ablation_mode",
            "sensitivity_param",
            "sensitivity_value",
            "regime_id",
        ],
        dropna=False,
    )[metrics].agg(["mean", "std"]).reset_index()
    agg.columns = [
        "_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col)
        for col in agg.columns
    ]
    agg.to_csv(agg_path, index=False)


def merge_regime_files(output_dir: Path, input_dirs: List[Path]) -> None:
    out_regime = output_dir / "regime_metrics"
    out_regime.mkdir(parents=True, exist_ok=True)
    for src_dir in input_dirs:
        src_regime = src_dir / "regime_metrics"
        if not src_regime.exists():
            continue
        for path in sorted(src_regime.glob("*.csv")):
            dst = out_regime / path.name
            if path.resolve() == dst.resolve():
                continue
            shutil.copy2(path, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Deduplicate and rebuild unified baseline result files.")
    parser.add_argument("--output_dir", required=True, help="Destination results dir, e.g. results/baseline_all_v2")
    parser.add_argument(
        "--input_dirs",
        nargs="*",
        default=[],
        help="Additional dirs to merge after the existing output_dir contents; later dirs win on duplicate result keys.",
    )
    parser.add_argument(
        "--rewrite_main_results",
        action="store_true",
        help="Rewrite conformal/adaptive main CSV and XLSX files after deduplication.",
    )
    args = parser.parse_args()

    output_dir = (ROOT / args.output_dir).resolve()
    input_dirs = [(ROOT / p).resolve() for p in args.input_dirs]
    output_dir.mkdir(parents=True, exist_ok=True)

    conf_frames = [read_csv_if_exists(output_dir / CONFORMAL_NAME)]
    adp_frames = [read_csv_if_exists(output_dir / ADAPTIVE_NAME)]
    for d in input_dirs:
        conf_frames.append(read_first_csv([d / CONFORMAL_NAME, d / "conformal_replacement.csv"]))
        adp_frames.append(read_first_csv([d / ADAPTIVE_NAME, d / "adaptive_replacement.csv"]))

    conformal = dedupe_by_result_key(conf_frames)
    adaptive = dedupe_by_result_key(adp_frames)

    if "coverage_gap" in conformal.columns:
        conformal = conformal.drop(columns=["coverage_gap"])
    if args.rewrite_main_results:
        conformal.to_csv(output_dir / CONFORMAL_NAME, index=False)
        adaptive.to_csv(output_dir / ADAPTIVE_NAME, index=False)
        conformal.to_excel(output_dir / "conformal_results.xlsx", index=False)
        adaptive.to_excel(output_dir / "adaptive_conformal_results.xlsx", index=False)

    merge_regime_files(output_dir, [output_dir] + input_dirs)
    rebuild_seed_summaries(output_dir, conformal, adaptive)
    rebuild_regime_summaries(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
