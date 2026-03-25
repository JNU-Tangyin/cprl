import numpy as np
from typing import List, Tuple, Dict, Any

def normalize_series(series: np.ndarray) -> np.ndarray:
    """Normalize a time series to zero mean and unit variance"""
    return (series - np.mean(series)) / (np.std(series) + 1e-8)

def compute_coverage(y_true: np.ndarray, intervals: List[Tuple[float, float]]) -> float:
    """
    Compute the empirical coverage of prediction intervals.
    
    Args:
        y_true: Array of true values
        intervals: List of (lower, upper) prediction intervals
        
    Returns:
        float: Empirical coverage (fraction of true values within intervals)
    """
    if len(y_true) != len(intervals):
        raise ValueError("Length of y_true must match length of intervals")
    
    covered = sum(1 for y, (lo, hi) in zip(y_true, intervals) if lo <= y <= hi)
    return covered / len(y_true)

def compute_average_width(intervals: List[Tuple[float, float]]) -> float:
    """Compute the average width of prediction intervals"""
    return np.mean([hi - lo for lo, hi in intervals])

def compute_w_ref(
    y_true: np.ndarray,
    method: str = "iqr",
) -> float:
    """Compute reference width for normalisation (IQR or std of y_true)."""
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
    """Coverage-Efficiency Score."""
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
    """Relative Coverage Score."""
    r = avg_width / max(w_ref, 1e-12)
    g = abs(coverage - target_coverage) / max(alpha, 1e-12)
    return (1.0 / (1.0 + r)) * np.exp(-g)

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

def compute_alpha_step_mean(series: List[float]) -> float:
    """Compute mean absolute step of a series."""
    a = np.asarray(series, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return float("nan")
    return float(np.mean(np.abs(np.diff(a))))

def compute_series_std(series: List[float]) -> float:
    """Compute standard deviation of a series."""
    a = np.asarray(series, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.std(a)) if a.size else float("nan")

def create_lagged_features(series: np.ndarray, lags: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create lagged features for time series forecasting.
    
    Args:
        series: Time series data
        lags: Number of lagged features to create
        
    Returns:
        Tuple of (X, y) where X has shape (n_samples, lags) and y has shape (n_samples,)
    """
    X, y = [], []
    for i in range(len(series) - lags):
        X.append(series[i:i+lags])
        y.append(series[i+lags])
    return np.array(X), np.array(y)