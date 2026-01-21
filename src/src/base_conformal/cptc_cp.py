import numpy as np
from collections import defaultdict

class CPTCCP:
    """
    CPTC (Sun & Yu, "Conformal Prediction for Time-series Forecasting with Change Points"):
    - Requires external state model outputs at each time t:
        z_prob_t: shape (K,), p_hat(z_t | x_{0:t})
        z_mean_t: shape (K,) or (K, d), state-conditioned forecast mean f_hat(x_t, z)
      (provided via kwargs in predict/update)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.2,          # learning rate in Algorithm 1
        min_residuals: int = 25,     # fallback threshold
        max_width: float = 1e6,      # acts like "infinity" score guard
        prob_threshold: float = 0.0, # ignore states with prob <= threshold
        seed: int = 0,
    ):
        self.alpha = float(alpha)
        self.initial_alpha = float(alpha)

        self.gamma = float(gamma)
        self.min_residuals = int(min_residuals)
        self.max_width = float(max_width)
        self.prob_threshold = float(prob_threshold)

        self.rng = np.random.default_rng(seed)

        # stateful:
        self._frozen = False
        self._K = None

        # per-state residual sets S_z and per-state alpha targets alpha_{z,t}
        self.S_z = None            # dict z -> list[score]
        self.alpha_z = None        # dict z -> float

        # global fallback residual pool
        self.S_all = None          # list[score]

        # last produced interval (for update err_t)
        self._last_interval = None

    def initialize(self, initial_data=None):
        self.alpha = self.initial_alpha
        self._frozen = False
        self._K = None
        self.S_z = None
        self.alpha_z = None
        self.S_all = []
        self._last_interval = None

    def start_test(self):
        # CPTC is an online method; no “fit then freeze” step is required.
        self._frozen = True

    @staticmethod
    def _score(y_true, y_pred_state):
        # nonconformity A: use L2 for vector, abs for scalar
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred_state, dtype=float)
        diff = yt - yp
        return float(np.linalg.norm(diff))

    @staticmethod
    def _quantile(residuals, q):
        r = np.asarray(residuals, dtype=float)
        return float(np.quantile(r, q))

    def _ensure_state_structures(self, K: int):
        if self._K is not None:
            return
        self._K = int(K)
        self.S_z = {z: [self.max_width] for z in range(self._K)}
        self.alpha_z = {z: float(self.alpha) for z in range(self._K)}
        # S_all also has a max_width guard
        self.S_all = [self.max_width]

    def predict(self, base_prediction=None, model_uncertainty=None, **kwargs):
        """
        kwargs must include:
          - z_prob_t: (K,)
          - z_mean_t: (K,) or (K,d)
        Returns:
          (lower, upper) for 1D;
          for d>1 you may want to adapt to per-dim intervals; here we keep scalar baseline.
        """
        z_prob_t = kwargs.get("z_prob_t", None)
        z_mean_t = kwargs.get("z_mean_t", None)
        if z_prob_t is None or z_mean_t is None:
            raise ValueError("CPTCCP.predict requires kwargs: z_prob_t and z_mean_t")

        z_prob_t = np.asarray(z_prob_t, dtype=float)
        z_mean_t = np.asarray(z_mean_t, dtype=float)

        if z_prob_t.ndim != 1:
            raise ValueError(f"z_prob_t must be 1D (K,), got {z_prob_t.shape}")
        K = z_prob_t.shape[0]
        self._ensure_state_structures(K)

        # Build state-specific intervals Γ_{z,t} using Q_{1-α_{z,t}}(S_z ∪ {∞})
        state_intervals = []
        for z in range(self._K):
            p = float(z_prob_t[z])
            if p <= self.prob_threshold:
                continue

            # choose residual pool
            residuals = self.S_z[z]
            if len(residuals) < self.min_residuals:
                residuals = self.S_all

            q = self._quantile(residuals, np.clip(1.0 - self.alpha_z[z], 0.0, 1.0))

            mu_z = float(np.ravel(z_mean_t[z])[0])  # scalar baseline
            lo_z, hi_z = mu_z - q, mu_z + q
            state_intervals.append((lo_z, hi_z, p))

        if len(state_intervals) == 0:
            mu = float(np.ravel(z_mean_t[0])[0])
            self._last_interval = (mu - self.max_width, mu + self.max_width)
            return self._last_interval

        state_intervals.sort(key=lambda x: x[2], reverse=True)

        target_mass = 1.0 - self.alpha
        mass = 0.0
        chosen = []
        for lo_z, hi_z, p in state_intervals:
            if mass < target_mass:
                chosen.append((lo_z, hi_z, p))
                mass += p
            else:
                break

        lower = min(iv[0] for iv in chosen)
        upper = max(iv[1] for iv in chosen)

        self._last_interval = (float(lower), float(upper))
        return self._last_interval

    def update(self, y_true, y_pred=None, prediction_interval=None, **kwargs):
        """
        kwargs must include:
          - z_prob_t: (K,)
          - z_mean_t: (K,) or (K,d)
        """
        z_prob_t = kwargs.get("z_prob_t", None)
        z_mean_t = kwargs.get("z_mean_t", None)
        if z_prob_t is None or z_mean_t is None:
            raise ValueError("CPTCCP.update requires kwargs: z_prob_t and z_mean_t")

        z_prob_t = np.asarray(z_prob_t, dtype=float)
        z_mean_t = np.asarray(z_mean_t, dtype=float)
        K = z_prob_t.shape[0]
        self._ensure_state_structures(K)

        # use interval from caller if provided, else last predicted
        interval = prediction_interval if prediction_interval is not None else self._last_interval
        if interval is None:
            raise RuntimeError("CPTCCP.update called before any predict().")

        yt = float(np.ravel(np.asarray(y_true, dtype=float))[0])
        lo, hi = float(interval[0]), float(interval[1])
        err_t = 1.0 if not (lo <= yt <= hi) else 0.0

        # sample z_hat ~ p_hat(z_t | x_{0:t})
        p = z_prob_t.clip(min=0.0)
        s = float(p.sum())
        if s <= 0:
            z_hat = int(self.rng.integers(0, self._K))
        else:
            p = p / s
            z_hat = int(self.rng.choice(self._K, p=p))

        # update alpha_{z_hat,t+1} = alpha_{z_hat,t} + gamma * (alpha - err_t)
        self.alpha_z[z_hat] = float(self.alpha_z[z_hat] + self.gamma * (self.alpha - err_t))
        self.alpha_z[z_hat] = float(np.clip(self.alpha_z[z_hat], 1e-6, 1.0 - 1e-6))


        mu_hat = z_mean_t[z_hat]
        score = self._score(yt, mu_hat)
        self.S_z[z_hat].append(score)
        self.S_all.append(score)
