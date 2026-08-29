# src/base_conformal/nex_cp.py
from __future__ import annotations

import numpy as np
from typing import Union, Optional, Tuple

Number = Union[int, float]


def weighted_quantile_with_point_mass_at_inf(
    values: np.ndarray,
    weights: np.ndarray,
    q: float,
    w_inf: float,
) -> float:
    """
    Implements: Q_q( sum_i weights_i * δ_{values_i} + w_inf * δ_{+∞} )
    Assumes: weights >= 0, and sum(weights) + w_inf = 1.

    If q is larger than the total finite mass, returns +∞.
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)

    if v.size != w.size:
        raise ValueError(f"values and weights must have same length, got {v.size} vs {w.size}")
    if v.size == 0:
        return float("inf")

    q = float(q)
    if q <= 0.0:
        return float(np.min(v))

    # sort by value
    order = np.argsort(v)
    v_sorted = v[order]
    w_sorted = w[order]

    # CDF over finite points only
    cdf = np.cumsum(w_sorted)
    finite_mass = float(cdf[-1])

    # if q exceeds total finite mass, quantile is +∞
    if q > finite_mass + 1e-15:
        return float("inf")

    k = int(np.searchsorted(cdf, q, side="left"))
    k = min(max(k, 0), v_sorted.size - 1)
    return float(v_sorted[k])


class NexCP:
    """
    NexCP (Nonexchangeable split conformal with weighted residual quantiles),
    aligned with Eq. (10)-(11) in "Conformal Prediction Beyond Exchangeability".

    Paper definition (split, symmetric algorithm):
      - Choose weights w1,...,wn in [0,1]
      - Normalize: \tilde{w}_i = w_i / (sum_j w_j + 1), and \tilde{w}_{n+1} = 1/(sum_j w_j + 1)
      - Interval: mu(x_{n+1}) ± Q_{1-α}( sum_i \tilde{w}_i δ_{R_i} + \tilde{w}_{n+1} δ_{+∞} )
        where R_i = |y_i - mu(x_i)| for a *pre-trained* model mu.
    :contentReference[oaicite:1]{index=1}

    In this code:
      - residuals R_i are collected online from (y_true, y_pred).
      - We instantiate a common drift-aware weight choice via exponential decay:
            w_i ∝ gamma^{age_i}
        (paper allows arbitrary fixed weights; this is a typical choice).
      - window_size is an OPTIONAL engineering choice:
            if provided, only last window_size residuals are used.
        If you want strict "use all past residuals", set window_size=None.
    """

    def __init__(
        self,
        alpha: float,
        window_size: Optional[int] = None,
        min_calib_size: int = 30,
        gamma: float = 0.99,
        warmup_scale: float = 3.0,
    ):
        self.alpha = float(alpha)
        self.initial_alpha = float(alpha)

        self.window_size = None if window_size is None else int(window_size)
        self.min_calib_size = int(min_calib_size)

        self.gamma = float(gamma)
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1].")

        # warmup fallback width multiplier (engineering; not in paper)
        self.warmup_scale = float(warmup_scale)

        self.residuals = []

    def initialize(self, initial_data=None):
        self.alpha = self.initial_alpha
        self.residuals = []

    def start_test(self):
        """
        API compatibility: NexCP split method does not require freezing
        if you keep updating residuals online.
        If you want a frozen split version, you'd stop calling update().
        """
        return

    def _get_residual_array(self) -> np.ndarray:
        if self.window_size is None:
            r = np.asarray(self.residuals, dtype=float)
        else:
            r = np.asarray(self.residuals[-self.window_size :], dtype=float)
        return r.reshape(-1)

    def _current_q(self, model_uncertainty: Number = 1.0) -> float:
        # engineering warmup (not part of paper)
        if len(self.residuals) < self.min_calib_size:
            return self.warmup_scale * float(model_uncertainty)

        r = self._get_residual_array()
        n = int(r.size)
        if n == 0:
            return self.warmup_scale * float(model_uncertainty)

        # Exponential-decay weights:
        # residuals stored in time order (oldest -> newest).
        # "Most recent" should get largest weight.
        ages = np.arange(n - 1, -1, -1, dtype=float)  # newest -> 0, oldest -> n-1
        w_raw = self.gamma ** ages

        # Eq.(10): normalize with "+1" mass at +∞
        denom = float(np.sum(w_raw) + 1.0)
        w_tilde = w_raw / denom
        w_inf = 1.0 / denom

        # Eq.(11): weighted quantile at level (1 - alpha)
        q_level = 1.0 - float(self.alpha)
        qhat = weighted_quantile_with_point_mass_at_inf(
            values=r,
            weights=w_tilde,
            q=q_level,
            w_inf=w_inf,
        )
        return float(qhat)

    def predict(
        self,
        base_prediction: Number,
        model_uncertainty: Number = 1.0,
        **kwargs,
    ) -> Tuple[float, float]:
        mu = float(base_prediction)
        q = float(self._current_q(model_uncertainty=model_uncertainty))

        if not np.isfinite(q):
            # If quantile is +∞, interval is infinite (this is allowed by the paper's construction)
            return -float("inf"), float("inf")

        return mu - q, mu + q

    def update(
        self,
        y_true: Number,
        y_pred: Number,
        prediction_interval=None,
        **kwargs,
    ):
        # split conformal residual score: R_t = |y_t - mu_t|
        r = abs(float(y_true) - float(y_pred))
        self.residuals.append(float(r))
