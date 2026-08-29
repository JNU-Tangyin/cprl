#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from statsmodels.tsa.stattools import adfuller, kpss


ROOT = Path(__file__).resolve().parents[1]
ACP_ROOT = ROOT.parent


@dataclass(frozen=True)
class SeriesSpec:
    dataset: str
    path: Path
    target: str = "OT"
    date_col: str = "date"


DEFAULT_SERIES = [
    SeriesSpec("ETTh1", ROOT / "time_series_library/dataset/ETT-small/ETTh1.csv"),
    SeriesSpec("ETTh2", ROOT / "time_series_library/dataset/ETT-small/ETTh2.csv"),
    SeriesSpec("ETTm1", ROOT / "time_series_library/dataset/ETT-small/ETTm1.csv"),
    SeriesSpec("ETTm2", ROOT / "time_series_library/dataset/ETT-small/ETTm2.csv"),
    SeriesSpec("Weather", ROOT / "time_series_library/dataset/weather/weather.csv"),
    SeriesSpec("Exchange", ROOT / "time_series_library/dataset/exchange_rate/exchange_rate.csv"),
    SeriesSpec(
        "BCI-AMD-Vol",
        ACP_ROOT / "Datasets/BCI/Variation of Local Fractional Coverage/AMD-fc.csv",
        target="1e2Vt",
        date_col="Date",
    ),
    SeriesSpec(
        "BCI-Amazon-Vol",
        ACP_ROOT / "Datasets/BCI/Variation of Local Fractional Coverage/Amazon-fc.csv",
        target="1e2Vt",
        date_col="Date",
    ),
    SeriesSpec(
        "BCI-Nvidia-Vol",
        ACP_ROOT / "Datasets/BCI/Variation of Local Fractional Coverage/Nvidia-fc.csv",
        target="1e2Vt",
        date_col="Date",
    ),
]


def clean_series(values: Iterable[float]) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def rolling_std_cv(y: np.ndarray, window: int) -> float:
    if len(y) < window:
        return float("nan")
    stds = pd.Series(y).rolling(window=window).std().dropna().to_numpy(dtype=float)
    stds = stds[np.isfinite(stds)]
    mean_std = float(np.mean(stds)) if len(stds) else float("nan")
    if not np.isfinite(mean_std) or mean_std == 0:
        return float("nan")
    return float(np.std(stds, ddof=1) / mean_std)


def normalized_wasserstein_shifts(y: np.ndarray, window: int, stride: int) -> np.ndarray:
    if len(y) < 2 * window:
        return np.array([], dtype=float)
    scale = float(np.std(y, ddof=1))
    if not np.isfinite(scale) or scale == 0:
        return np.array([], dtype=float)
    out = []
    for start in range(0, len(y) - 2 * window + 1, stride):
        left = y[start : start + window]
        right = y[start + window : start + 2 * window]
        out.append(wasserstein_distance(left, right) / scale)
    return np.array(out, dtype=float)


def shift_metrics(dataset: str, y: np.ndarray, window: int, stride: int) -> dict[str, object]:
    shifts = normalized_wasserstein_shifts(y, window, stride)
    return {
        "dataset": dataset,
        "n_obs": len(y),
        "window": window,
        "stride": stride,
        "n_shift_windows": len(shifts),
        "mean_nw": float(np.mean(shifts)) if len(shifts) else float("nan"),
        "p90_nw": float(np.percentile(shifts, 90)) if len(shifts) else float("nan"),
        "max_nw": float(np.max(shifts)) if len(shifts) else float("nan"),
        "rolling_std_cv": rolling_std_cv(y, window),
    }


def stationarity_tests(dataset: str, y: np.ndarray) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset": dataset,
        "n_obs": len(y),
        "adf_stat": float("nan"),
        "adf_pvalue": float("nan"),
        "kpss_stat": float("nan"),
        "kpss_pvalue": float("nan"),
        "interpretation": "insufficient data",
    }
    if len(y) < 20:
        return row

    try:
        adf_stat, adf_pvalue, *_ = adfuller(y, autolag="AIC")
        row["adf_stat"] = float(adf_stat)
        row["adf_pvalue"] = float(adf_pvalue)
    except Exception as exc:
        row["adf_error"] = type(exc).__name__

    try:
        kpss_stat, kpss_pvalue, *_ = kpss(y, regression="c", nlags="auto")
        row["kpss_stat"] = float(kpss_stat)
        row["kpss_pvalue"] = float(kpss_pvalue)
    except Exception as exc:
        row["kpss_error"] = type(exc).__name__

    adf_supports_nonstationary = np.isfinite(row["adf_pvalue"]) and float(row["adf_pvalue"]) >= 0.05
    kpss_supports_nonstationary = np.isfinite(row["kpss_pvalue"]) and float(row["kpss_pvalue"]) < 0.05
    if adf_supports_nonstationary and kpss_supports_nonstationary:
        row["interpretation"] = "strong evidence"
    elif adf_supports_nonstationary or kpss_supports_nonstationary:
        row["interpretation"] = "moderate evidence"
    else:
        row["interpretation"] = "weak evidence"
    return row


def load_csv_series(spec: SeriesSpec) -> tuple[np.ndarray, str]:
    df = pd.read_csv(spec.path)
    if spec.target not in df.columns:
        raise KeyError(f"{spec.path} missing target column {spec.target!r}")
    return clean_series(df[spec.target]), spec.target


def parse_fred_ts(path: Path) -> dict[str, np.ndarray]:
    series: dict[str, np.ndarray] = {}
    in_data = False
    with path.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if not in_data:
                if line.lower() == "@data":
                    in_data = True
                continue
            name, _timestamp, values = line.split(":", 2)
            series[name] = clean_series(values.split(","))
    return series


def aggregate_rows(dataset: str, rows: list[dict[str, object]], fields: list[str]) -> dict[str, object]:
    out: dict[str, object] = {"dataset": dataset, "n_series": len(rows)}
    for field in fields:
        vals = pd.to_numeric(pd.Series([r.get(field) for r in rows]), errors="coerce").dropna()
        out[field] = float(vals.mean()) if len(vals) else float("nan")
    if "interpretation" in rows[0]:
        counts = pd.Series([r.get("interpretation", "") for r in rows]).value_counts()
        out["interpretation"] = counts.index[0] if len(counts) else ""
        for label in ["strong evidence", "moderate evidence", "weak evidence"]:
            out[f"n_{label.replace(' ', '_')}"] = int(counts.get(label, 0))
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    priority = [
        "dataset",
        "n_obs",
        "n_series",
        "window",
        "stride",
        "n_shift_windows",
        "mean_nw",
        "p90_nw",
        "max_nw",
        "rolling_std_cv",
        "adf_stat",
        "adf_pvalue",
        "kpss_stat",
        "kpss_pvalue",
        "interpretation",
    ]
    ordered = [f for f in priority if f in fieldnames] + [f for f in fieldnames if f not in priority]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute target-series non-stationarity diagnostics.")
    parser.add_argument("--out_dir", default="results/nonstationarity")
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--include_fred", action="store_true", default=True)
    parser.add_argument("--fred_path", default=str(ACP_ROOT / "Datasets/fred_md/fred_md_dataset.ts"))
    args = parser.parse_args()

    out_dir = (ROOT / args.out_dir).resolve()
    shift_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []

    for spec in DEFAULT_SERIES:
        if not spec.path.exists():
            print(f"[Skip] Missing {spec.dataset}: {spec.path}")
            continue
        y, target = load_csv_series(spec)
        srow = shift_metrics(spec.dataset, y, args.window, args.stride)
        srow["target"] = target
        trow = stationarity_tests(spec.dataset, y)
        trow["target"] = target
        shift_rows.append(srow)
        test_rows.append(trow)

    fred_path = Path(args.fred_path)
    fred_shift_detail: list[dict[str, object]] = []
    fred_test_detail: list[dict[str, object]] = []
    if args.include_fred and fred_path.exists():
        for name, y in parse_fred_ts(fred_path).items():
            dataset = f"FRED-MD-{name}"
            fred_shift_detail.append(shift_metrics(dataset, y, args.window, args.stride) | {"target": name})
            fred_test_detail.append(stationarity_tests(dataset, y) | {"target": name})
        shift_rows.append(
            aggregate_rows(
                "FRED-MD-Avg107",
                fred_shift_detail,
                ["n_obs", "n_shift_windows", "mean_nw", "p90_nw", "max_nw", "rolling_std_cv"],
            )
            | {"target": "all_107_series"}
        )
        test_rows.append(
            aggregate_rows(
                "FRED-MD-Avg107",
                fred_test_detail,
                ["n_obs", "adf_stat", "adf_pvalue", "kpss_stat", "kpss_pvalue"],
            )
            | {"target": "all_107_series"}
        )
    elif args.include_fred:
        print(f"[Skip] Missing FRED-MD: {fred_path}")

    write_csv(out_dir / "shift_metrics.csv", shift_rows)
    write_csv(out_dir / "stationarity_tests.csv", test_rows)
    if fred_shift_detail:
        write_csv(out_dir / "fred_md_shift_metrics_by_series.csv", fred_shift_detail)
    if fred_test_detail:
        write_csv(out_dir / "fred_md_stationarity_tests_by_series.csv", fred_test_detail)

    print(f"[Done] wrote {out_dir / 'shift_metrics.csv'}")
    print(f"[Done] wrote {out_dir / 'stationarity_tests.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
