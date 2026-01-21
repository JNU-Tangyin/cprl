import numpy as np

class StandardCP:
    """
    Standard CP baseline (online residual quantile):
    - Calibration/warm-up: collect residuals until min_calib_size
    - Online phase: at each step, form PI using empirical (1-alpha) quantile
      of past residuals, then append the new residual (no freezing, no weighting).
    """
    def __init__(self, alpha, window_size=None, min_calib_size=30):
        self.alpha = float(alpha)
        self.initial_alpha = float(alpha)
        self.min_calib_size = int(min_calib_size)

        self.residuals = []
        self.window_size = window_size  # unused for this baseline

    def initialize(self, initial_data=None):
        self.residuals = []
        self.alpha = self.initial_alpha

    def start_test(self):
        """API compatibility; online baseline does not freeze."""
        return

    def _current_q(self, model_uncertainty=None):
        if len(self.residuals) < self.min_calib_size:
            raise ValueError(
                f"Not enough calibration residuals: {len(self.residuals)} < {self.min_calib_size}."
            )
        r = np.asarray(self.residuals, dtype=float)
        return float(np.quantile(r, 1.0 - self.alpha))

    def predict(self, base_prediction, model_uncertainty=None, **kwargs):
        q = self._current_q(model_uncertainty=model_uncertainty)
        mu = float(base_prediction)
        return mu - q, mu + q

    def update(self, y_true, y_pred, prediction_interval=None, **kwargs):
        # Online CP: always update residuals (matches run_cp logic)
        self.residuals.append(abs(float(y_true) - float(y_pred)))
