import numpy as np

class DFPI:
    """
    Distribution-Free Predictive Inference (Split Conformal Regression)
    (Lei et al., 2018)

    Workflow:
      Calibration phase (I2):
        - collect residuals r_i = |y_i - yhat_i|
      Test phase:
        - freeze q_hat = k-th order statistic of residuals where
              k = ceil((m + 1) * (1 - alpha)),  m = |I2|
        - predict interval for new x:
              [yhat - q_hat, yhat + q_hat]
    """

    def __init__(self, alpha: float, min_calib_size: int = 30):
        self.alpha = float(alpha)
        self.initial_alpha = float(alpha)
        self.min_calib_size = int(min_calib_size)

        self._frozen = False
        self.residuals = []
        self.q_hat = None

    def initialize(self, initial_data=None):
        self.alpha = self.initial_alpha
        self._frozen = False
        self.residuals = []
        self.q_hat = None

    @staticmethod
    def _conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
        """
        Conformal quantile for split conformal regression:
          m = len(residuals)
          k = ceil((m + 1) * (1 - alpha))
          q_hat = k-th smallest residual
        """
        r = np.asarray(residuals, dtype=float)
        m = r.size
        if m == 0:
            raise ValueError("Empty residuals.")
        k = int(np.ceil((m + 1) * (1.0 - alpha)))
        k = min(max(k, 1), m)  # clamp to [1, m]
        return float(np.sort(r)[k - 1])

    def start_test(self):
        """Freeze after calibration: compute q_hat from calibration residuals."""
        if len(self.residuals) < self.min_calib_size:
            raise ValueError(
                f"Not enough calibration residuals for DFPI: {len(self.residuals)} < {self.min_calib_size}."
            )
        self.q_hat = self._conformal_quantile(self.residuals, self.alpha)
        self._frozen = True

    def predict(self, base_prediction, model_uncertainty=None, **kwargs):
        """
        Return DFPI interval around base prediction.
        During calibration (not frozen), return a wide placeholder interval.
        """
        mu = float(base_prediction)

        if (not self._frozen) or (self.q_hat is None):
            return mu - 1e6, mu + 1e6

        q = float(self.q_hat)
        return mu - q, mu + q

    def update(self, y_true, y_pred, prediction_interval=None, **kwargs):
        """
        Calibration phase: collect residuals on I2.
        Test phase: do nothing (split conformal is frozen).
        """
        if self._frozen:
            return
        err = abs(float(y_true) - float(y_pred))
        self.residuals.append(err)
