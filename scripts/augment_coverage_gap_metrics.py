#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def compute_direction(signed_bias: float, tol: float = 1e-12) -> str:
    if signed_bias > tol:
        return "overcoverage"
    if signed_bias < -tol:
        return "undercoverage"
    return "balanced"


def augment_csv(path: Path) -> bool:
    df = pd.read_csv(path)
    if df.empty:
        return False

    required = {"coverage", "target_coverage"}
    if not required.issubset(df.columns):
        return False

    coverage = pd.to_numeric(df["coverage"], errors="coerce")
    target = pd.to_numeric(df["target_coverage"], errors="coerce")
    signed_bias = coverage - target

    df["abs_coverage_gap"] = signed_bias.abs()
    df["under_coverage_gap"] = (target - coverage).clip(lower=0.0)
    df["over_coverage_gap"] = (coverage - target).clip(lower=0.0)
    df["coverage_bias"] = signed_bias
    df["coverage_bias_direction"] = [
        compute_direction(v) if np.isfinite(v) else "unknown"
        for v in signed_bias.to_numpy(dtype=float)
    ]

    if "coverage_gap" in df.columns:
        df = df.drop(columns=["coverage_gap"])

    cols = list(df.columns)
    if "coverage" in cols:
        insert_at = cols.index("coverage") + 1
        ordered_new = [
            "abs_coverage_gap",
            "under_coverage_gap",
            "over_coverage_gap",
            "coverage_bias",
            "coverage_bias_direction",
        ]
        remaining = [c for c in cols if c not in ordered_new]
        cols = remaining[:insert_at] + ordered_new + remaining[insert_at:]
        # de-duplicate if script re-run
        seen = set()
        cols = [c for c in cols if not (c in seen or seen.add(c))]
        df = df[cols]

    df.to_csv(path, index=False)
    return True


def main() -> int:
    updated = 0
    for path in sorted(ROOT.joinpath("results").rglob("conformal_results.csv")):
        if augment_csv(path):
            updated += 1
            print(path)
    print(f"updated={updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
