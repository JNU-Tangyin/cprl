# src/base_conformal/enbpi_cp.py
from __future__ import annotations

import numpy as np
from collections import deque


class EnbPICP:
    """
    ENBPI: Ensemble / Nonparametric Bootstrap Prediction Interval
    (Xu & Xie / related ENBPI line; your implementation follows:
      - bootstrap ensemble
      - OOB residuals init
      - sliding residual pool during test
      - beta-search for asymmetric intervals
    )

    This file is made compatible with your ExpConformal's cp.predict/cp.update
    which passes many alias kwargs (x/X/features/base_prediction/y_pred/etc.).
    """

    def __init__(
        self,
        alpha=0.1,
        B=20,
        T=200,                 # residual pool window size (deque maxlen)
        s=1,                   # update frequency
        fit_func_factory=None,
        agg="mean",
        batch_size_s=None,     # alias for s in your builder
        random_state=0,
        **kwargs,
    ):
        self.alpha = float(alpha)
        self.initial_alpha = float(alpha)

        self.B = int(B)
        self.T = int(T)

        # IMPORTANT: respect batch_size_s if provided, else use s
        if batch_size_s is not None:
            self.s = int(batch_size_s)
        else:
            self.s = int(s)

        if self.s <= 0:
            self.s = 1

        if fit_func_factory is None:
            raise ValueError("EnbPICP requires fit_func_factory (a sklearn-like regressor factory).")
        self.fit_func_factory = fit_func_factory

        self.agg = str(agg)
        self.random_state = int(random_state)

        # internal states
        self._models = []
        self._boot_sets = []
        self._resid_pool = deque(maxlen=self.T)
        self._pending_resids = []

        # collected calibration data
        self._X_calib = []
        self._Y_calib = []

        self._in_test = False
        self._t_test = 0

        self.rng = np.random.default_rng(self.random_state)

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------

    def initialize(self, initial_data=None):
        self.alpha = float(self.initial_alpha)

        self._models = []
        self._boot_sets = []
        self._resid_pool = deque(maxlen=self.T)
        self._pending_resids = []

        self._X_calib = []
        self._Y_calib = []

        self._in_test = False
        self._t_test = 0

    def _aggregate(self, preds):
        if self.agg == "mean":
            return float(np.mean(preds))
        elif self.agg == "median":
            return float(np.median(preds))
        else:
            raise ValueError(f"Unknown agg: {self.agg}")

    def _pick_x_from_aliases(self, *, x=None, X=None, features=None):
        """Pick the covariate vector from common aliases."""
        if x is not None:
            return x
        if X is not None:
            return X
        if features is not None:
            return features
        return None

    def _get_feature(self, *, x=None, base_prediction=None) -> np.ndarray:
        """
        Prefer explicit covariate x (lags). If missing, fallback to [base_prediction].
        Keyword-only signature prevents 'multiple values' collisions.
        """
        if x is not None:
            return np.asarray(x, dtype=np.float32).reshape(-1)
        if base_prediction is None:
            raise ValueError("EnbPICP requires x (preferred) or base_prediction.")
        return np.asarray([float(base_prediction)], dtype=np.float32)

    # ------------------------------------------------------------------
    # beta-search
    # ------------------------------------------------------------------

    def _beta_search(self, resid_arr, beta_grid=50):
        """
        Find beta in [0, alpha] such that interval width is minimized:
          q_{1-alpha+beta} - q_{beta}
        """
        al = float(self.alpha)
        grid = np.linspace(0.0, al, int(beta_grid))
        best_beta = 0.0
        best_width = np.inf

        for b in grid:
            ql = np.quantile(resid_arr, b)
            qu = np.quantile(resid_arr, 1.0 - al + b)
            width = float(qu - ql)
            if width < best_width:
                best_width = width
                best_beta = float(b)
        return float(best_beta)

    # ------------------------------------------------------------------
    # calibration -> start_test
    # ------------------------------------------------------------------

    def start_test(self):
        """
        Train bootstrap models and initialize residual pool using OOB residuals.
        """
        X = np.asarray(self._X_calib, dtype=np.float32)
        Y = np.asarray(self._Y_calib, dtype=np.float32).reshape(-1)
        n = int(Y.size)

        if n <= 1:
            raise RuntimeError("Not enough calibration data for EnbPI. Need at least 2 samples.")

        # ---- 1) train bootstrap models ----
        self._models = []
        self._boot_sets = []

        for b in range(self.B):
            idx = self.rng.choice(n, size=n, replace=True)
            self._boot_sets.append(set(idx.tolist()))
            model = self.fit_func_factory()
            model.fit(X[idx], Y[idx])
            self._models.append(model)

        # ---- 2) OOB residual initialization ----
        oob_resids = []
        for t in range(n):
            Xq = X[t:t + 1]
            oob_models = [m for b, m in enumerate(self._models) if t not in self._boot_sets[b]]
            if len(oob_models) == 0:
                oob_models = self._models

            preds = np.array([m.predict(Xq)[0] for m in oob_models], dtype=np.float32)
            center = self._aggregate(preds)
            resid = abs(float(Y[t]) - float(center))
            oob_resids.append(float(resid))

        self._resid_pool = deque(oob_resids[-self.T:], maxlen=self.T)
        self._pending_resids = []
        self._t_test = 0
        self._in_test = True

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, base_prediction, model_uncertainty=None, **kwargs):
        """
        Interval at query x:
          center(x) ± quantiles of residual pool (asymmetric via beta-search).
        """
        # ---- strip aliases that would cause duplicates ----
        x_raw = self._pick_x_from_aliases(
            x=kwargs.pop("x", None),
            X=kwargs.pop("X", None),
            features=kwargs.pop("features", None),
        )

        mu = float(base_prediction)

        # during calibration or invalid state
        if (not self._in_test) or (len(self._models) == 0) or (len(self._resid_pool) == 0):
            return mu - 1e6, mu + 1e6

        x_vec = self._get_feature(x=x_raw, base_prediction=mu)
        Xq = x_vec.reshape(1, -1)

        preds = np.array([m.predict(Xq)[0] for m in self._models], dtype=np.float32)
        center = self._aggregate(preds)

        resid_arr = np.asarray(self._resid_pool, dtype=np.float32)
        if resid_arr.size == 0:
            return center - 1e6, center + 1e6

        beta_grid = int(kwargs.pop("beta_grid", 50))
        beta_hat = self._beta_search(resid_arr, beta_grid=beta_grid)

        w_lower = float(np.quantile(resid_arr, beta_hat))
        w_upper = float(np.quantile(resid_arr, 1.0 - float(self.alpha) + beta_hat))

        return float(center - w_lower), float(center + w_upper)

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    def update(self, y_true, y_pred, prediction_interval=None, **kwargs):
        """
        Calibration phase: store (x, y).
        Test phase: update sliding residual pool every s steps.
        """
        yt = float(y_true)
        mu = float(y_pred)

        # ---- strip alias collisions ----
        x_raw = self._pick_x_from_aliases(
            x=kwargs.pop("x", None),
            X=kwargs.pop("X", None),
            features=kwargs.pop("features", None),
        )
        # also remove prediction aliases if present
        kwargs.pop("base_prediction", None)
        kwargs.pop("prediction", None)
        kwargs.pop("pred", None)
        kwargs.pop("y_hat", None)

        x_vec = self._get_feature(x=x_raw, base_prediction=mu)

        # calibration phase
        if not self._in_test:
            self._X_calib.append(x_vec.astype(np.float32))
            self._Y_calib.append(yt)
            return

        # test phase residual
        Xq = x_vec.reshape(1, -1)
        preds = np.array([m.predict(Xq)[0] for m in self._models], dtype=np.float32)
        center = self._aggregate(preds)

        resid_t = abs(yt - float(center))
        self._pending_resids.append(float(resid_t))
        self._t_test += 1

        if (self._t_test % self.s) == 0:
            for r in self._pending_resids:
                self._resid_pool.append(r)
            self._pending_resids = []
