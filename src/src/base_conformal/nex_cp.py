import numpy as np
from .standard_cp import StandardCP

def weighted_quantile_with_inf(values, weights, q, w_inf):
    """
    Compute Q_q( sum_i w_i δ_{v_i} + w_inf δ_{+∞} )
    where weights are already normalized (sum(weights) + w_inf = 1).

    Returns:
        finite threshold if CDF over finite values reaches q,
        else +∞.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if len(values) == 0:
        return float("inf") if q <= w_inf else float("inf")

    # sort by value
    idx = np.argsort(values)
    v = values[idx]
    w = weights[idx]

    # CDF over finite points only
    cdf = np.cumsum(w)

    # if q exceeds total finite mass, quantile is +∞
    if q > cdf[-1]:
        return float("inf")

    k = int(np.searchsorted(cdf, q, side="left"))
    k = min(max(k, 0), len(v) - 1)
    return float(v[k])

class NexCP(StandardCP):
    """
    NexCP (nonexchangeable split conformal with residual weights),
    implemented according to Eq. (10)-(11) of the NexCP paper.
    """

    def __init__(self, alpha, window_size, min_calib_size=30, gamma=0.99):
        super().__init__(alpha, window_size, min_calib_size)
        self.gamma = float(gamma)

    def _current_q(self, model_uncertainty):
        # follow your project's warmup convention (not part of the NexCP paper)
        if len(self.residuals) < self.min_calib_size:
            return 3.0 * float(model_uncertainty)

        r = np.asarray(self.residuals, dtype=float)
        n = len(r)

        # weights like w_i = 0.99^{n+1-i} (paper uses 0.99^{n+1-i} in experiments)
        # If residuals are stored in time order, "most recent" should get the largest weight.
        ages = np.arange(n - 1, -1, -1)  # recent -> 0, old -> n-1
        w_raw = (self.gamma ** ages)

        # Eq (10): normalized weights with "+1" and extra mass at +∞
        denom = float(w_raw.sum() + 1.0)
        w = w_raw / denom
        w_inf = 1.0 / denom

        # Eq (11): Q_{1-α} of weighted empirical + w_inf δ_{+∞}
        return weighted_quantile_with_inf(r, w, q=(1.0 - self.alpha), w_inf=w_inf)
