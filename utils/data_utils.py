import numpy as np
from typing import Tuple

def normalize_series(series: np.ndarray) -> np.ndarray:
    return (series - np.mean(series)) / (np.std(series) + 1e-8)

def create_lagged_features(series: np.ndarray, lags: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(series) - lags):
        X.append(series[i:i+lags])
        y.append(series[i+lags])
    return np.array(X), np.array(y)
