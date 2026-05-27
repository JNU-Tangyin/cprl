from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _to_1d(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _maybe_get_array(data, *keys) -> Optional[np.ndarray]:
    for key in keys:
        if key in data.files:
            return np.asarray(data[key])
    return None


def _sort_triplet(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    time_idx: np.ndarray,
):
    order = np.argsort(time_idx)
    return y_true[order], y_pred[order], time_idx[order]


def _build_lag_feature_from_past_true(
    y_true_series: np.ndarray,
    lag: int,
) -> np.ndarray:
    y = _to_1d(y_true_series)
    T = len(y)

    if lag <= 0:
        return np.zeros((T, 0), dtype=float)

    X = np.zeros((T, lag), dtype=float)
    for t in range(T):
        start = max(0, t - lag)
        past = y[start:t]
        if len(past) > 0:
            X[t, lag - len(past):] = past
    return X


def _build_test_lag_feature_with_val_history(
    val_y_true: np.ndarray,
    test_y_true: np.ndarray,
    lag: int,
) -> np.ndarray:
    val_hist = _to_1d(val_y_true)
    test_hist = _to_1d(test_y_true)
    T = len(test_hist)

    if lag <= 0:
        return np.zeros((T, 0), dtype=float)

    full_hist = np.concatenate([val_hist, test_hist], axis=0)
    val_len = len(val_hist)
    X = np.zeros((T, lag), dtype=float)

    for j in range(T):
        cur = val_len + j
        start = max(0, cur - lag)
        past = full_hist[start:cur]
        if len(past) > 0:
            X[j, lag - len(past):] = past
    return X


def load_cache_for_conformal(
    cache_path: str,
    x_lag: int = 24,
) -> Dict[str, np.ndarray]:
    data = np.load(cache_path, allow_pickle=True)

    val_y_true = _maybe_get_array(data, "val_y_true", "val_y_true_full")
    val_y_pred = _maybe_get_array(data, "val_y_pred", "val_y_pred_full")
    test_y_true = _maybe_get_array(data, "test_y_true", "test_y_true_full")
    test_y_pred = _maybe_get_array(data, "test_y_pred", "test_y_pred_full")

    if val_y_true is None or val_y_pred is None:
        raise KeyError(
            "Missing val_y_true/val_y_pred or val_y_true_full/val_y_pred_full in cache."
        )
    if test_y_true is None or test_y_pred is None:
        raise KeyError(
            "Missing test_y_true/test_y_pred or test_y_true_full/test_y_pred_full in cache."
        )

    val_y_true = _to_1d(val_y_true)
    val_y_pred = _to_1d(val_y_pred)
    test_y_true = _to_1d(test_y_true)
    test_y_pred = _to_1d(test_y_pred)

    if len(val_y_true) != len(val_y_pred):
        raise ValueError("val_y_true and val_y_pred length mismatch.")
    if len(test_y_true) != len(test_y_pred):
        raise ValueError("test_y_true and test_y_pred length mismatch.")

    val_time_idx = _maybe_get_array(data, "val_time_idx")
    test_time_idx = _maybe_get_array(data, "test_time_idx")

    if val_time_idx is None:
        val_time_idx = np.arange(len(val_y_true), dtype=int)
    else:
        val_time_idx = _to_1d(val_time_idx)

    if test_time_idx is None:
        test_time_idx = np.arange(len(test_y_true), dtype=int)
    else:
        test_time_idx = _to_1d(test_time_idx)

    if len(val_time_idx) != len(val_y_true):
        raise ValueError("val_time_idx and val_y_true length mismatch.")
    if len(test_time_idx) != len(test_y_true):
        raise ValueError("test_time_idx and test_y_true length mismatch.")

    val_y_true, val_y_pred, val_time_idx = _sort_triplet(
        val_y_true, val_y_pred, val_time_idx
    )
    test_y_true, test_y_pred, test_time_idx = _sort_triplet(
        test_y_true, test_y_pred, test_time_idx
    )

    val_x = _build_lag_feature_from_past_true(val_y_true, x_lag)
    test_x = _build_test_lag_feature_with_val_history(val_y_true, test_y_true, x_lag)

    return {
        "val_y_true": val_y_true,
        "val_y_pred": val_y_pred,
        "test_y_true": test_y_true,
        "test_y_pred": test_y_pred,
        "val_time_idx": val_time_idx,
        "test_time_idx": test_time_idx,
        "val_step": np.arange(len(val_y_true), dtype=int),
        "test_step": np.arange(len(test_y_true), dtype=int),
        "val_x": val_x,
        "test_x": test_x,
    }
