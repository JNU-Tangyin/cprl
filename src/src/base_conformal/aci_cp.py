# src/base_conformal/aci_cp.py
import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize


@dataclass
class ACIConfig:
    alpha0: float = 0.1          # target miscoverage
    gamma: float = 0.01          # step size for alpha update
    split_size: float = 0.75     # fraction used for training each step
    t_init: int = 200            # start ACI after t_init samples
    max_iter: int = 500          # QR optimizer iters
    eps: float = 1e-4            # clip alpha in (eps, 1-eps)
    seed: int = 0                # rng seed


def _add_intercept(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    ones = np.ones((X.shape[0], 1), dtype=float)
    return np.concatenate([ones, X], axis=1)


def _fit_linear_quantile_regression(X: np.ndarray, y: np.ndarray, q: float, max_iter: int) -> np.ndarray:
    """
    Linear quantile regression via pinball loss + BFGS:
        min_beta sum_i rho_q(y_i - x_i^T beta)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    n, d = X.shape
    beta0 = np.zeros(d, dtype=float)

    def pinball_loss(beta):
        r = y - X.dot(beta)
        return np.sum(np.maximum(q * r, (q - 1.0) * r))

    res = minimize(pinball_loss, beta0, method="BFGS", options={"maxiter": int(max_iter)})
    return res.x  # (d,)


class ACICP:
    """
    Full ACI baseline (Gibbs & Candès, 2021 style)
    """

    def __init__(
        self,
        alpha: float,
        window_size=None,            # not used; kept for builder compatibility
        min_calib_size: int = 30,    # used as lower bound for t_init
        lr: float = 0.01,            # treat as gamma (to keep old arg name)
        split_size: float = 0.75,
        t_init: int = 200,
        max_iter: int = 500,
        seed: int = 0,
    ):
        self.initial_alpha = float(alpha)
        self.alpha = float(alpha)

        self.config = ACIConfig(
            alpha0=float(alpha),
            gamma=float(lr),
            split_size=float(split_size),
            t_init=int(max(t_init, min_calib_size)),
            max_iter=int(max_iter),
            seed=int(seed),
        )

        self.rng = np.random.default_rng(self.config.seed)
        self.X_hist = []  # list of (d,)
        self.y_hist = []  # list of float

    def initialize(self, initial_data=None):
        self.alpha = self.initial_alpha
        self.X_hist.clear()
        self.y_hist.clear()

    def _ready(self):
        return len(self.y_hist) >= self.config.t_init

    def predict(self, base_prediction=None, model_uncertainty=1.0, x=None, **kwargs):
        if x is None:
            raise ValueError("ACICP.predict requires x=... (feature vector).")

        x = np.asarray(x, dtype=float).reshape(1, -1)

        # warmup fallback
        if not self._ready():
            mu = float(base_prediction) if base_prediction is not None else 0.0
            w = 3.0 * float(model_uncertainty)
            return mu - w, mu + w

        X_all = np.asarray(self.X_hist, dtype=float)
        y_all = np.asarray(self.y_hist, dtype=float).reshape(-1)

        t = len(y_all)
        n_train = int(self.config.split_size * t)

        idx_train = self.rng.choice(t, size=n_train, replace=False)
        mask = np.zeros(t, dtype=bool)
        mask[idx_train] = True
        train_idx = np.where(mask)[0]
        calib_idx = np.where(~mask)[0]

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_cal, y_cal = X_all[calib_idx], y_all[calib_idx]

        # intercept
        X_train_i = _add_intercept(X_train)
        X_cal_i = _add_intercept(X_cal)
        x_i = _add_intercept(x)

        a = float(np.clip(self.alpha, self.config.eps, 1.0 - self.config.eps))
        q_low = a / 2.0
        q_up = 1.0 - a / 2.0

        beta_low = _fit_linear_quantile_regression(X_train_i, y_train, q_low, self.config.max_iter)
        beta_up = _fit_linear_quantile_regression(X_train_i, y_train, q_up, self.config.max_iter)

        pred_low_cal = X_cal_i.dot(beta_low)
        pred_up_cal = X_cal_i.dot(beta_up)

        # conformity scores
        scores = np.maximum(y_cal - pred_up_cal, pred_low_cal - y_cal)
        q_hat = float(np.quantile(scores, 1.0 - a))

        mu_low = float(x_i.dot(beta_low)[0])
        mu_up = float(x_i.dot(beta_up)[0])

        return mu_low - q_hat, mu_up + q_hat

    def update(self, y_true, y_pred=None, prediction_interval=None, interval=None, x=None, **kwargs):
        if prediction_interval is None:
            prediction_interval = interval
        if prediction_interval is None:
            raise ValueError("ACICP.update requires prediction_interval (or interval).")
        if x is None:
            raise ValueError("ACICP.update requires x=... (feature vector).")

        y_true = float(y_true)
        lower, upper = prediction_interval
        miss = 0.0 if (lower <= y_true <= upper) else 1.0

        # ACI update
        self.alpha = float(self.alpha + self.config.gamma * (self.initial_alpha - miss))
        self.alpha = float(np.clip(self.alpha, self.config.eps, 1.0 - self.config.eps))

        # append to history
        self.X_hist.append(np.asarray(x, dtype=float).reshape(-1))
        self.y_hist.append(y_true)
