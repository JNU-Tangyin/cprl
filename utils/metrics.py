# utils/metrics.py
import numpy as np
from typing import List, Tuple, Optional, Sequence


def _as_interval_arrays(
    intervals: Optional[List[Tuple[float, float]]] = None,
    lower: Optional[Sequence[float]] = None,
    upper: Optional[Sequence[float]] = None,
):
    """
    Normalize interval input into two numpy arrays: lower, upper.
    Supports either:
      1) intervals = [(lo, hi), ...]
      2) lower=array_like, upper=array_like
    """
    if intervals is not None:
        if len(intervals) == 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        lo = np.asarray([x[0] for x in intervals], dtype=float)
        hi = np.asarray([x[1] for x in intervals], dtype=float)
        return lo, hi

    if lower is None or upper is None:
        raise ValueError("Provide either intervals or both lower and upper.")

    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)

    if lo.shape != hi.shape:
        raise ValueError("lower and upper must have the same shape.")

    return lo, hi


# ---------- 基础 conformal 指标 ----------

def compute_coverage(
    y_true: np.ndarray,
    intervals: Optional[List[Tuple[float, float]]] = None,
    lower: Optional[Sequence[float]] = None,
    upper: Optional[Sequence[float]] = None,
) -> float:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    lo, hi = _as_interval_arrays(intervals=intervals, lower=lower, upper=upper)

    if len(y) != len(lo):
        raise ValueError("y_true and intervals length mismatch.")
    if len(y) == 0:
        return float("nan")

    return float(np.mean((lo <= y) & (y <= hi)))


def compute_average_width(
    intervals: Optional[List[Tuple[float, float]]] = None,
    lower: Optional[Sequence[float]] = None,
    upper: Optional[Sequence[float]] = None,
) -> float:
    lo, hi = _as_interval_arrays(intervals=intervals, lower=lower, upper=upper)
    if len(lo) == 0:
        return float("nan")
    return float(np.mean(hi - lo))


def compute_winkler_score(
    y_true: np.ndarray,
    alpha: float,
    intervals: Optional[List[Tuple[float, float]]] = None,
    lower: Optional[Sequence[float]] = None,
    upper: Optional[Sequence[float]] = None,
) -> float:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    lo, hi = _as_interval_arrays(intervals=intervals, lower=lower, upper=upper)

    if len(y) != len(lo):
        raise ValueError("y_true and intervals length mismatch.")
    if len(y) == 0:
        return float("nan")

    width = hi - lo
    score = width.copy()

    below = y < lo
    above = y > hi

    score[below] += (2.0 / max(alpha, 1e-12)) * (lo[below] - y[below])
    score[above] += (2.0 / max(alpha, 1e-12)) * (y[above] - hi[above])

    return float(np.mean(score))


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
    return float(1.0 / (1.0 + r + g))


def compute_rcs(
    coverage: float,
    target_coverage: float,
    avg_width: float,
    w_ref: float,
    alpha: float,
) -> float:
    r = avg_width / max(w_ref, 1e-12)
    g = abs(coverage - target_coverage) / max(alpha, 1e-12)
    return float((1.0 / (1.0 + r)) * np.exp(-g))


# ---------- adaptive 专用诊断 ----------

def compute_worst_window_coverage(
    y_true: np.ndarray,
    intervals: Optional[List[Tuple[float, float]]] = None,
    lower: Optional[Sequence[float]] = None,
    upper: Optional[Sequence[float]] = None,
    window: int = 100,
) -> float:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    lo, hi = _as_interval_arrays(intervals=intervals, lower=lower, upper=upper)

    n = len(y)
    if n == 0:
        return float("nan")
    if len(lo) != n:
        raise ValueError("y_true and intervals length mismatch.")

    if n < window:
        return compute_coverage(y, lower=lo, upper=hi)

    worst = 1.0
    for i in range(0, n - window + 1):
        c = compute_coverage(y[i:i + window], lower=lo[i:i + window], upper=hi[i:i + window])
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