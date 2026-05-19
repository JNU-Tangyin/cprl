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

SETTING_RE = re.compile(
    r"^(?P<dataset>.+?\.csv)_lags\d+_model(?P<base_model>.+?)_cp(?P<cp_mode>.+?)_mode.+?_seed(?P<cache_seed>\d+)"
)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def dedupe_by_setting(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    use = [df for df in frames if not df.empty]
    if not use:
        return pd.DataFrame()
    merged = pd.concat(use, ignore_index=True)
    if "setting" not in merged.columns:
        return merged
    return merged.drop_duplicates(subset=["setting"], keep="last").reset_index(drop=True)


def ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = ""
    return out


def parse_setting(setting: str) -> dict:
    m = SETTING_RE.match(str(setting))
    if not m:
        return {"dataset": "", "base_model": "", "cp_mode": "", "cache_seed": ""}
    return m.groupdict()


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
        if "setting" in adp.columns:
            adp = adp.drop_duplicates(subset=["setting"], keep="last")
        rename = {c: f"adaptive_{c}" for c in adp.columns if c != "setting"}
        adp = adp.rename(columns=rename)
        merged = conf.merge(adp, on="setting", how="left")

    parsed = merged["setting"].map(parse_setting).apply(pd.Series)
    merged["phase"] = "baseline"
    merged["dataset"] = parsed["dataset"].str.replace(".csv", "", regex=False)
    merged["cache_seed"] = pd.to_numeric(parsed["cache_seed"], errors="coerce")
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
        "base_model",
        "ablation_mode",
        "sensitivity_param",
        "sensitivity_value",
    ]
    merged = ensure_cols(merged, raw_cols)
    merged = merged[raw_cols]
    merged.to_csv(out_dir / "seed_summary_raw.csv", index=False)

    metric_cols = [
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
        "adaptive_worst_window_coverage",
        "adaptive_width_step_mean",
        "adaptive_width_std",
        "adaptive_control_alpha_step_mean",
        "adaptive_control_alpha_std",
    ]
    use_metrics = [c for c in metric_cols if c in merged.columns]
    agg = merged.groupby(
        ["dataset", "base_model", "cp_mode", "target_coverage"], dropna=False
    )[use_metrics].agg(["mean", "std"]).reset_index()
    agg.columns = [
        "_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col)
        for col in agg.columns
    ]
    agg.to_csv(out_dir / "seed_summary_agg.csv", index=False)


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
            df["cache_seed"] = int(parsed["cache_seed"]) if parsed["cache_seed"] else ""
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
    parser.add_argument("--output_dir", required=True, help="Destination results dir, e.g. results/baseline_all")
    parser.add_argument(
        "--input_dirs",
        nargs="*",
        default=[],
        help="Additional dirs to merge after the existing output_dir contents; later dirs win on duplicate settings.",
    )
    args = parser.parse_args()

    output_dir = (ROOT / args.output_dir).resolve()
    input_dirs = [(ROOT / p).resolve() for p in args.input_dirs]
    output_dir.mkdir(parents=True, exist_ok=True)

    conf_frames = [read_csv_if_exists(output_dir / CONFORMAL_NAME)]
    adp_frames = [read_csv_if_exists(output_dir / ADAPTIVE_NAME)]
    for d in input_dirs:
        conf_frames.append(read_csv_if_exists(d / CONFORMAL_NAME))
        adp_frames.append(read_csv_if_exists(d / ADAPTIVE_NAME))

    conformal = dedupe_by_setting(conf_frames)
    adaptive = dedupe_by_setting(adp_frames)

    if "coverage_gap" in conformal.columns:
        conformal = conformal.drop(columns=["coverage_gap"])
    conformal.to_csv(output_dir / CONFORMAL_NAME, index=False)
    adaptive.to_csv(output_dir / ADAPTIVE_NAME, index=False)

    merge_regime_files(output_dir, [output_dir] + input_dirs)
    rebuild_seed_summaries(output_dir, conformal, adaptive)
    rebuild_regime_summaries(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
