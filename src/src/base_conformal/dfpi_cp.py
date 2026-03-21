import numpy as np


class DFPI:
    """
    Standard DFPI / Split Conformal Regression
    compatible with ACP-style pipeline.

    Standard workflow:
      Calibration phase:
        - collect residuals r_i = |y_i - yhat_i|
      start_test():
        - compute q_hat = k-th order statistic of calibration residuals
          where k = ceil((m + 1) * (1 - alpha))
      Test phase:
        - return [yhat - q_hat, yhat + q_hat]
        - do not update anymore

    """

    def __init__(self, alpha: float, min_calib_size: int = 1):
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

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
        Standard split conformal quantile:
            k = ceil((m + 1) * (1 - alpha))
            q_hat = k-th smallest residual
        """
        r = np.asarray(residuals, dtype=float)
        m = r.size
        if m == 0:
            raise ValueError("Calibration residuals are empty.")

        k = int(np.ceil((m + 1) * (1.0 - alpha)))
        k = min(max(k, 1), m)  # clamp to [1, m]

        # k-th smallest element, 1-based -> 0-based index k-1
        return float(np.partition(r, k - 1)[k - 1])

    def start_test(self):
        """
        Freeze after calibration and compute q_hat once.
        """
        m = len(self.residuals)
        if m < self.min_calib_size:
            raise ValueError(
                f"Not enough calibration residuals for DFPI: {m} < {self.min_calib_size}."
            )

        self.q_hat = self._conformal_quantile(
            np.asarray(self.residuals, dtype=float),
            self.alpha
        )
        self._frozen = True

    def predict(self, base_prediction, model_uncertainty=None, **kwargs):
        """
        Before start_test():
            return a dummy wide interval for pipeline compatibility.
        After start_test():
            return standard split conformal interval [yhat - q_hat, yhat + q_hat].
        """
        mu = float(base_prediction)

        if (not self._frozen) or (self.q_hat is None):
            return mu - 1e6, mu + 1e6

        q = float(self.q_hat)
        return mu - q, mu + q

    def update(self, y_true, y_pred, prediction_interval=None, **kwargs):
        """
        Calibration phase:
            collect residuals r_i = |y_i - yhat_i|
        Test phase:
            do nothing (standard DFPI is frozen)
        """
        if self._frozen:
            return

        err = abs(float(y_true) - float(y_pred))
        self.residuals.append(err)