import numpy as np
from collections import deque
from sklearn.linear_model import QuantileRegressor

class CQRCP:
    """
    CQR (Conformalized Quantile Regression) baseline, per Romano et al. (NeurIPS 2019):
      - Collect (X, y) during calibration (ExpConformal.calibrate)
      - At start_test(): randomly split collected calib data into I1/I2
          * Fit two quantile regressors on I1: q_lo(x), q_hi(x)
          * Compute conformity scores on I2:
                E_i = max(q_lo(X_i) - y_i, y_i - q_hi(X_i))
          * Compute correction Q_hat = empirical_quantile_{(1-α)(1+1/|I2|)}(E)
          * Freeze (no more updates)
      - At predict() on test: return [q_lo(x) - Q_hat, q_hi(x) + Q_hat]
    """

    def __init__(
        self,
        alpha: float,
        feature_window: int = 2,
        past_window: int = 1,
        min_calib_size: int = 60,      # needs enough to split into I1/I2
        split_ratio: float = 0.5,      # I1 proportion inside calibration set
        qr_l2: float = 0.0,
        solver: str = "highs",
        random_state: int = 1103,
    ):
        self.alpha = float(alpha)
        self.initial_alpha = float(alpha)

        self.feature_window = int(feature_window)
        if self.feature_window != 2:
            raise ValueError("CQRQuantileCP currently supports feature_window=2 only.")

        self.past_window = int(past_window)
        self.min_calib_size = int(min_calib_size)
        self.split_ratio = float(split_ratio)
        self.qr_l2 = float(qr_l2)
        self.solver = solver
        self.random_state = int(random_state)

        # internal state
        self._frozen = False
        self._prev_pred = None

        self._X_calib = []
        self._Y_calib = []

        self._model_lo = None
        self._model_hi = None
        self._Qhat = None  # conformal correction

        # for optional smoothing of features at predict time
        self._feat_hist = deque(maxlen=max(1, self.past_window))

        # quantiles for base QR models
        self._q_lo = self.alpha / 2.0
        self._q_hi = 1.0 - self.alpha / 2.0

    def initialize(self, initial_data=None):
        self.alpha = self.initial_alpha
        self._frozen = False
        self._prev_pred = None
        self._X_calib = []
        self._Y_calib = []
        self._model_lo = None
        self._model_hi = None
        self._Qhat = None
        self._feat_hist.clear()

    def _make_feature(self, curr_pred: float):
        """feature = [prev_pred, curr_pred]"""
        if self._prev_pred is None:
            return None
        return np.asarray([self._prev_pred, float(curr_pred)], dtype=np.float32)

    def _smooth_feature(self, feat: np.ndarray) -> np.ndarray:
        """Optional smoothing over past_window features (prediction-time only)."""
        if self.past_window <= 1:
            return feat
        self._feat_hist.append(feat)
        return np.mean(np.stack(list(self._feat_hist), axis=0), axis=0).astype(np.float32)

    @staticmethod
    def _empirical_quantile_conformal(values: np.ndarray, alpha: float) -> float:
        """
        Conformal quantile used in split conformal literature:
          Q = (1-α)(1+1/m)-th empirical quantile for m = |I2|.
        Implemented as the k-th order statistic with k = ceil((m+1)*(1-α)).
        """
        v = np.asarray(values, dtype=float)
        m = v.size
        if m == 0:
            raise ValueError("Empty values for conformal quantile.")
        k = int(np.ceil((m + 1) * (1.0 - alpha)))
        k = min(max(k, 1), m)  # clamp to [1, m]
        return float(np.sort(v)[k - 1])

    def start_test(self):
        """
        Fit QR on I1, compute conformal correction on I2, then freeze.
        Matches Algorithm 1 / Eq.(6)-(8) in the CQR paper. :contentReference[oaicite:3]{index=3}
        """
        n = len(self._Y_calib)
        if n < self.min_calib_size:
            raise ValueError(f"Not enough calibration samples for CQR: {n} < {self.min_calib_size}")

        X = np.asarray(self._X_calib, dtype=np.float32)
        Y = np.asarray(self._Y_calib, dtype=np.float32)

        # --- random split into I1 / I2 ---
        rng = np.random.default_rng(self.random_state)
        idx = np.arange(n)
        rng.shuffle(idx)

        n1 = int(np.floor(self.split_ratio * n))
        n1 = min(max(n1, 1), n - 1)  # ensure both non-empty
        I1, I2 = idx[:n1], idx[n1:]

        X1, Y1 = X[I1], Y[I1]
        X2, Y2 = X[I2], Y[I2]

        # --- fit base quantile regressors on I1 ---
        self._model_lo = QuantileRegressor(quantile=self._q_lo, alpha=self.qr_l2, solver=self.solver)
        self._model_hi = QuantileRegressor(quantile=self._q_hi, alpha=self.qr_l2, solver=self.solver)
        self._model_lo.fit(X1, Y1)
        self._model_hi.fit(X1, Y1)

        # --- conformity scores on I2: E_i = max(q_lo(x)-y, y-q_hi(x)) (Eq. 6) ---
        qlo = self._model_lo.predict(X2)
        qhi = self._model_hi.predict(X2)
        E = np.maximum(qlo - Y2, Y2 - qhi)  # :contentReference[oaicite:4]{index=4}

        # --- conformal correction Q_hat: (1-α)(1+1/|I2|) empirical quantile (Eq. 8) ---
        self._Qhat = self._empirical_quantile_conformal(E, self.alpha)  # :contentReference[oaicite:5]{index=5}

        self._frozen = True

    def predict(self, base_prediction, model_uncertainty=None, **kwargs):
        """
        Test-time interval:
          [ q_lo(x) - Q_hat, q_hi(x) + Q_hat ]  (Eq. 7) :contentReference[oaicite:6]{index=6}
        Here x is our feature built from predictions: [prev_pred, curr_pred].
        """
        mu = float(base_prediction)

        feat = self._make_feature(mu)
        if feat is None:
            # first point: no prev_pred yet
            return mu - 1e6, mu + 1e6

        feat = self._smooth_feature(feat)

        if (not self._frozen) or (self._model_lo is None) or (self._model_hi is None) or (self._Qhat is None):
            # before fitting, output a very wide interval
            return mu - 1e6, mu + 1e6

        qlo = float(self._model_lo.predict(feat.reshape(1, -1))[0])
        qhi = float(self._model_hi.predict(feat.reshape(1, -1))[0])

        lower = min(qlo, qhi) - float(self._Qhat)
        upper = max(qlo, qhi) + float(self._Qhat)
        return lower, upper

    def update(self, y_true, y_pred, prediction_interval=None, **kwargs):
        """
        Calibration phase: collect (feature, y_true).
        Test phase (frozen): no updates (split conformal).
        """
        yp = float(y_pred)
        yt = float(y_true)

        feat = self._make_feature(yp)

        # update prev_pred regardless, so next step has a feature
        self._prev_pred = yp

        if self._frozen:
            return

        if feat is None:
            # can't form [prev, curr] for the very first point
            return

        # Collect raw features/targets; conformal split happens inside start_test()
        self._X_calib.append(feat.tolist())
        self._Y_calib.append(yt)
