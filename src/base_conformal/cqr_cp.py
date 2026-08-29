# src/base_conformal/cqr_cp.py
from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

from sklearn.linear_model import QuantileRegressor

Number = Union[int, float]


@dataclass
class CQRConfig:
    alpha: float = 0.1

    # minimum calibration size to allow split I1/I2
    min_calib_size: int = 60

    # I1 proportion inside calibration set (Romano split conformal style)
    split_ratio: float = 0.5

    # QuantileRegressor regularization strength (L2)
    qr_l2: float = 0.0
    solver: str = "highs"

    # feature processing (engineering; doesn't change methodology)
    standardize_x: bool = True
    eps: float = 1e-8

    # fallback (before frozen)
    fallback_width: float = 3.0

    # if you prefer sequential split for time series (optional)
    sequential_split: bool = False

    random_state: int = 1103


class CQRCP:
    """
    Conformalized Quantile Regression (CQR), Romano et al. (NeurIPS 2019).

    Protocol:
      - During calibration stream: collect (x_t, y_t).
      - At start_test(): split the collected calibration set into I1/I2.
          * Fit two quantile regressors on I1: q_lo(x), q_hi(x)
          * Compute conformity scores on I2:
                E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))
          * Compute correction Q_hat = empirical quantile at level
                k = ceil((m+1)*(1-α)) / m   (order statistic form)
          * Freeze model: no more updates after start_test()
      - Test-time interval:
            [ q_lo(x) - Q_hat, q_hi(x) + Q_hat ]

    Time-series adaptation used here:
      - Use the *lag vector x* (your ExpConformal passes x=lags) as covariate.
        This keeps CQR meaningful in time-series (conditional intervals).
    """

    def __init__(
        self,
        alpha: float,
        min_calib_size: int = 60,
        split_ratio: float = 0.5,
        qr_l2: float = 0.0,
        solver: str = "highs",
        standardize_x: bool = True,
        sequential_split: bool = False,
        fallback_width: float = 3.0,
        random_state: int = 1103,
        **kwargs,
    ):
        self.cfg = CQRConfig(
            alpha=float(alpha),
            min_calib_size=int(min_calib_size),
            split_ratio=float(split_ratio),
            qr_l2=float(qr_l2),
            solver=str(solver),
            standardize_x=bool(standardize_x),
            sequential_split=bool(sequential_split),
            fallback_width=float(fallback_width),
            random_state=int(random_state),
        )

        self.initial_alpha = float(alpha)
        self.alpha = float(alpha)

        # quantiles for base QR models
        self._q_lo = self.alpha / 2.0
        self._q_hi = 1.0 - self.alpha / 2.0

        # collected calibration data
        self._X_calib: List[np.ndarray] = []
        self._Y_calib: List[float] = []

        # fitted models and correction
        self._model_lo: Optional[QuantileRegressor] = None
        self._model_hi: Optional[QuantileRegressor] = None
        self._Qhat: Optional[float] = None
        self._frozen: bool = False

        # feature standardization stats (fit on I1 only)
        self._x_mean: Optional[np.ndarray] = None
        self._x_std: Optional[np.ndarray] = None

    def initialize(self, initial_data=None):
        self.alpha = self.initial_alpha
        self._X_calib = []
        self._Y_calib = []
        self._model_lo = None
        self._model_hi = None
        self._Qhat = None
        self._frozen = False
        self._x_mean = None
        self._x_std = None

    @staticmethod
    def _as_row(x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return x.reshape(1, -1)
        if x.ndim == 2:
            return x
        return x.reshape(1, -1)

    def _fit_standardizer(self, X: np.ndarray):
        self._x_mean = X.mean(axis=0, keepdims=True)
        self._x_std = X.std(axis=0, keepdims=True)
        self._x_std = np.maximum(self._x_std, self.cfg.eps)

    def _transform_x(self, X: np.ndarray) -> np.ndarray:
        if not self.cfg.standardize_x:
            return X
        if self._x_mean is None or self._x_std is None:
            return X
        return (X - self._x_mean) / self._x_std

    @staticmethod
    def _conformal_order_stat_quantile(values: np.ndarray, alpha: float) -> float:
        """
        Split conformal order-stat quantile:
          k = ceil((m + 1) * (1 - alpha))
          q = k-th smallest value (1-indexed)
        """
        v = np.asarray(values, dtype=float).reshape(-1)
        m = int(v.size)
        if m <= 0:
            raise ValueError("Empty conformity score set for conformal quantile.")
        k = int(np.ceil((m + 1) * (1.0 - float(alpha))))
        k = max(1, min(m, k))
        return float(np.sort(v)[k - 1])

    def start_test(self):
        """
        Fit QR on I1, compute conformal correction on I2, then freeze.
        """
        n = len(self._Y_calib)
        if n < self.cfg.min_calib_size:
            raise ValueError(f"Not enough calibration samples for CQR: {n} < {self.cfg.min_calib_size}")

        X = np.asarray(self._X_calib, dtype=float)
        Y = np.asarray(self._Y_calib, dtype=float).reshape(-1)

        # --- split into I1 / I2 ---
        if self.cfg.sequential_split:
            # sequential split (time-series friendly): first part I1, second part I2
            n1 = int(np.floor(self.cfg.split_ratio * n))
            n1 = min(max(n1, 1), n - 1)
            I1 = np.arange(0, n1)
            I2 = np.arange(n1, n)
        else:
            # random split (Romano original split conformal style)
            rng = np.random.default_rng(self.cfg.random_state)
            idx = np.arange(n)
            rng.shuffle(idx)
            n1 = int(np.floor(self.cfg.split_ratio * n))
            n1 = min(max(n1, 1), n - 1)
            I1, I2 = idx[:n1], idx[n1:]

        X1, Y1 = X[I1], Y[I1]
        X2, Y2 = X[I2], Y[I2]

        # standardize using I1 only (engineering, stable across scales)
        if self.cfg.standardize_x:
            self._fit_standardizer(X1)
            X1s = self._transform_x(X1)
            X2s = self._transform_x(X2)
        else:
            X1s, X2s = X1, X2

        # --- fit base quantile regressors on I1 ---
        self._model_lo = QuantileRegressor(quantile=self._q_lo, alpha=self.cfg.qr_l2, solver=self.cfg.solver)
        self._model_hi = QuantileRegressor(quantile=self._q_hi, alpha=self.cfg.qr_l2, solver=self.cfg.solver)
        self._model_lo.fit(X1s, Y1)
        self._model_hi.fit(X1s, Y1)

        # --- conformity scores on I2 ---
        qlo = self._model_lo.predict(X2s)
        qhi = self._model_hi.predict(X2s)
        E = np.maximum(qlo - Y2, Y2 - qhi)

        # --- conformal correction ---
        self._Qhat = self._conformal_order_stat_quantile(E, self.alpha)

        self._frozen = True

    def predict(
        self,
        base_prediction: Number,
        model_uncertainty: Number = 1.0,
        x=None,
        **kwargs,
    ) -> Tuple[float, float]:
        """
        Test-time interval:
          [ q_lo(x) - Q_hat, q_hi(x) + Q_hat ].

        Before start_test() freezes the models, return a safe fallback interval.
        """
        mu = float(base_prediction)
        unc = float(model_uncertainty)

        if x is None:
            # CQR is conditional on x; without x we can't do meaningful CQR
            w = self.cfg.fallback_width * max(1e-12, unc)
            return mu - w, mu + w

        if (not self._frozen) or (self._model_lo is None) or (self._model_hi is None) or (self._Qhat is None):
            w = self.cfg.fallback_width * max(1e-12, unc)
            return mu - w, mu + w

        Xr = self._as_row(x)
        Xr = self._transform_x(Xr)

        qlo = float(self._model_lo.predict(Xr)[0])
        qhi = float(self._model_hi.predict(Xr)[0])
        lo = min(qlo, qhi) - float(self._Qhat)
        hi = max(qlo, qhi) + float(self._Qhat)
        return float(lo), float(hi)

    def update(
        self,
        y_true: Number,
        y_pred: Number,
        prediction_interval=None,
        x=None,
        **kwargs,
    ):
        """
        Calibration: collect (x, y_true).
        After start_test() (frozen): no updates.
        """
        if self._frozen:
            return
        if x is None:
            return

        yt = float(y_true)
        Xr = self._as_row(x).reshape(-1)  # store 1d feature
        self._X_calib.append(Xr)
        self._Y_calib.append(yt)
