import numpy as np
from collections import deque


class EnbPICP:
    """
    Ensemble / Nonparametric Bootstrap Prediction Interval (ENBPI)

    Key properties:
    - Bootstrap ensemble of base learners
    - OOB residuals for calibration
    - Sliding residual pool during test
    - Beta-search for asymmetric intervals
    """

    def __init__(
        self,
        alpha=0.1,
        B=20,
        T=200,
        s=1,
        fit_func_factory=None,
        agg="mean",
        batch_size_s=None, **kwargs
    ):
        self.alpha = alpha
        self.B = B                  # number of bootstrap models
        self.T = T 
        if batch_size_s is not None:
            self.s = int(batch_size_s)                 # residual pool window size
        self.s = s                  # update frequency
        self.fit_func_factory = fit_func_factory
        self.agg = agg

        # internal states
        self._models = []
        self._boot_sets = []
        self._resid_pool = deque(maxlen=self.T)
        self._pending_resids = []

        self._X_calib = []
        self._Y_calib = []

        self._in_test = False
        self._t_test = 0

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------

    def _aggregate(self, preds):
        if self.agg == "mean":
            return float(np.mean(preds))
        elif self.agg == "median":
            return float(np.median(preds))
        else:
            raise ValueError(f"Unknown agg: {self.agg}")

    def _get_feature(self, base_prediction, **kwargs):
        """
        Priority:
        1. use kwargs["x"] if provided
        2. fallback to base_prediction
        """
        if "x" in kwargs and kwargs["x"] is not None:
            x = np.asarray(kwargs["x"], dtype=np.float32)
        else:
            x = np.asarray([base_prediction], dtype=np.float32)
        return x

    # ------------------------------------------------------------------
    # beta-search (kept as requested)
    # ------------------------------------------------------------------

    def _beta_search(self, resid_arr):
        """
        Find beta in [0, alpha] such that
        q_{1-alpha+beta} - q_{beta} is minimized
        """
        al = self.alpha
        grid = np.linspace(0.0, al, 50)
        best_beta = 0.0
        best_width = np.inf

        for b in grid:
            ql = np.quantile(resid_arr, b)
            qu = np.quantile(resid_arr, 1.0 - al + b)
            width = qu - ql
            if width < best_width:
                best_width = width
                best_beta = b

        return float(best_beta)

    # ------------------------------------------------------------------
    # calibration -> start_test
    # ------------------------------------------------------------------

    def start_test(self):
        """
        Train bootstrap models and initialize residual pool
        using OOB residuals (core ENBPI step)
        """
        X = np.asarray(self._X_calib, dtype=np.float32)
        Y = np.asarray(self._Y_calib, dtype=np.float32)
        n = len(Y)

        if n == 0:
            raise RuntimeError("No calibration data collected.")

        # ---- 1. train bootstrap models ----
        self._models = []
        self._boot_sets = []

        for b in range(self.B):
            idx = np.random.choice(n, size=n, replace=True)
            self._boot_sets.append(set(idx.tolist()))

            model = self.fit_func_factory()
            model.fit(X[idx], Y[idx])
            self._models.append(model)

        # ---- 2. OOB residual initialization ----
        oob_resids = []

        for t in range(n):
            Xq = X[t:t + 1]

            oob_models = [
                m for b, m in enumerate(self._models)
                if t not in self._boot_sets[b]
            ]
            if len(oob_models) == 0:
                oob_models = self._models  # fallback

            preds = np.array(
                [m.predict(Xq)[0] for m in oob_models],
                dtype=np.float32,
            )
            center = self._aggregate(preds)
            resid = abs(float(Y[t]) - center)
            oob_resids.append(resid)

        # initialize sliding residual pool
        self._resid_pool = deque(oob_resids[-self.T:], maxlen=self.T)
        self._pending_resids = []
        self._t_test = 0
        self._in_test = True

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, base_prediction, model_uncertainty=None, **kwargs):
        x = self._get_feature(base_prediction, **kwargs)

        # during calibration or invalid state
        if (not self._in_test) or (len(self._models) == 0) or (len(self._resid_pool) == 0):
            mu = float(base_prediction)
            return mu - 1e6, mu + 1e6

        Xq = x.reshape(1, -1)
        preds = np.array(
            [m.predict(Xq)[0] for m in self._models],
            dtype=np.float32,
        )
        center = self._aggregate(preds)

        resid_arr = np.asarray(self._resid_pool, dtype=np.float32)

        beta_hat = self._beta_search(resid_arr)
        w_lower = float(np.quantile(resid_arr, beta_hat))
        w_upper = float(np.quantile(resid_arr, 1.0 - self.alpha + beta_hat))

        return center - w_lower, center + w_upper

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    def update(self, y_true, y_pred, prediction_interval=None, **kwargs):
        yt = float(y_true)
        x = self._get_feature(y_pred, **kwargs)

        # calibration phase
        if not self._in_test:
            self._X_calib.append(x.astype(np.float32))
            self._Y_calib.append(yt)
            return

        # test phase
        Xq = x.reshape(1, -1)
        preds = np.array(
            [m.predict(Xq)[0] for m in self._models],
            dtype=np.float32,
        )
        center = self._aggregate(preds)

        resid_t = abs(yt - center)
        self._pending_resids.append(float(resid_t))
        self._t_test += 1

        # sliding update every s steps
        if (self._t_test % self.s) == 0:
            for r in self._pending_resids:
                self._resid_pool.append(r)   # deque(maxlen=T) auto-pops
            self._pending_resids = []
