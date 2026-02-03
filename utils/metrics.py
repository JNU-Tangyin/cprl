import numpy as np
from typing import List, Tuple

# ---------- 基础 conformal 指标 ----------

def compute_coverage(y_true: np.ndarray, intervals: List[Tuple[float, float]]) -> float:
    covered = sum(1 for y, (lo, hi) in zip(y_true, intervals) if lo <= y <= hi)
    return covered / len(y_true)

def compute_average_width(intervals: List[Tuple[float, float]]) -> float:
    return float(np.mean([hi - lo for lo, hi in intervals]))


# ---------- composite 指标 ----------

def compute_w_ref(
    y_true: np.ndarray,
    method: str = "iqr",
) -> float:
    y = np.asarray(y_true, dtype=float)
    y = y[np.isfinite(y)]

    if y.size == 0:
        return 1.0

    if method == "iqr":
        q75, q25 = np.percentile(y, [75, 25])
        w = float(q75 - q25)
    elif method == "std":
        w = float(np.std(y))
    else:
        raise ValueError(f"Unknown w_ref method: {method}")

    # numerical safety
    if not np.isfinite(w) or w <= 1e-12:
        w = float(np.std(y) + 1e-6)

    return max(w, 1e-12)


def compute_ces(
    coverage: float,
    target_coverage: float,
    avg_width: float,
    w_ref: float,
    alpha: float,
) -> float:
    r = avg_width / max(w_ref, 1e-12)
    g = abs(coverage - target_coverage) / max(alpha, 1e-12)
    return 1.0 / (1.0 + r + g)

def compute_rcs(
    coverage: float,
    target_coverage: float,
    avg_width: float,
    w_ref: float,
    alpha: float,
) -> float:
    r = avg_width / max(w_ref, 1e-12)
    g = abs(coverage - target_coverage) / max(alpha, 1e-12)
    return (1.0 / (1.0 + r)) * np.exp(-g)


# ---------- adaptive 专用诊断 ----------

def compute_worst_window_coverage(
    y_true: np.ndarray,
    intervals: List[Tuple[float, float]],
    window: int = 100,
) -> float:
    """Worst (minimum) sliding-window coverage on test stream."""
    n = len(y_true)
    if n == 0:
        return float("nan")
    if n < window:
        return compute_coverage(y_true, intervals)
    worst = 1.0
    for i in range(0, n - window + 1):
        c = compute_coverage(y_true[i:i+window], intervals[i:i+window])
        if c < worst:
            worst = c
    return float(worst)

def compute_step_mean(series: List[float]) -> float:
    a = np.asarray(series, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return float("nan")
    return float(np.mean(np.abs(np.diff(a))))


def compute_alpha_step_mean(series: List[float]) -> float:
    return compute_step_mean(series)

def compute_series_std(series: List[float]) -> float:
    a = np.asarray(series, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.std(a)) if a.size else float("nan")


