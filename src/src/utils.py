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