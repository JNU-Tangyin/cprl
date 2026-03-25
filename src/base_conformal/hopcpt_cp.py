# src/base_conformal/hopcpt_cp.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

Number = Union[int, float]


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x.reshape(1, -1)
    if x.ndim == 2:
        return x
    return x.reshape(1, -1)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """
    Weighted empirical quantile over finite support.
    Assumes weights are nonnegative and sum(weights) == 1 (approximately).
    Returns the smallest v such that CDF(v) >= q.
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if v.size == 0:
        raise ValueError("weighted_quantile: empty values.")
    if v.size != w.size:
        raise ValueError(f"weighted_quantile: size mismatch {v.size} vs {w.size}")
    q = float(q)
    if q <= 0.0:
        return float(np.min(v))
    if q >= 1.0:
        return float(np.max(v))

    order = np.argsort(v)
    v_sorted = v[order]
    w_sorted = w[order]
    cdf = np.cumsum(w_sorted)

    k = int(np.searchsorted(cdf, q, side="left"))
    k = min(max(k, 0), v_sorted.size - 1)
    return float(v_sorted[k])


def weighted_quantile_with_inf(values: np.ndarray, weights: np.ndarray, q: float, w_inf: float) -> float:
    """
    Compute Q_q( sum_i w_i δ_{v_i} + w_inf δ_{+∞} ), with:
        sum(weights) + w_inf = 1.

    If q exceeds the total finite mass, return +∞.
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if v.size == 0:
        return float("inf")
    if v.size != w.size:
        raise ValueError(f"weighted_quantile_with_inf: size mismatch {v.size} vs {w.size}")

    q = float(q)
    if q <= 0.0:
        return float(np.min(v))
    if q >= 1.0:
        return float("inf") if w_inf > 0 else float(np.max(v))

    order = np.argsort(v)
    v_sorted = v[order]
    w_sorted = w[order]
    cdf = np.cumsum(w_sorted)

    finite_mass = float(cdf[-1])  # = 1 - w_inf ideally
    if q > finite_mass + 1e-15:
        return float("inf")

    k = int(np.searchsorted(cdf, q, side="left"))
    k = min(max(k, 0), v_sorted.size - 1)
    return float(v_sorted[k])


@dataclass
class HopCPTConfig:
    alpha: float = 0.1
    min_calib_size: int = 60

    emb_dim: int = 32
    hidden_dim: int = 64

    train_epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    seed: int = 0

    beta: float = 1.0
    online_update: bool = True

    # ---- IMPORTANT SWITCH ----
    # If True, include +∞ point mass like "nonexchangeable weighted quantile with inf-mass".
    # This can legitimately yield infinite intervals when w_inf > alpha.
    # Default False to keep HopCPT a usable baseline.
    use_inf_mass: bool = False

    # If use_inf_mass=True and q falls into inf mass, you have two choices:
    #   - strict: return (-inf, inf) (paper-pure, but ruins experiments)
    #   - fallback: return the max finite threshold (keeps experiments comparable)
    inf_mass_fallback_to_max: bool = True

    fallback_width: float = 3.0
    eps: float = 1e-12

    # engineering bound to avoid plotting/numerical explosion
    M_lower: float = -1e6
    M_upper: float = 1e6


class _HopCPTNet(nn.Module):
    """
    Encoder m(.) + linear projections Wq/Wk for dot-product similarity.
    """
    def __init__(self, in_dim: int, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )
        self.Wq = nn.Linear(emb_dim, emb_dim, bias=False)
        self.Wk = nn.Linear(emb_dim, emb_dim, bias=False)

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        return self.encoder(z)

    def logits(self, q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        q: (B, d) or (d,)
        K: (T, d)
        returns: (B, T)
        """
        if q.dim() == 1:
            q = q.unsqueeze(0)
        qh = self.Wq(q)          # (B,d)
        Kh = self.Wk(K)          # (T,d)
        return qh @ Kh.transpose(0, 1)  # (B,T)


class HopCPTCP:
    """
    HopCPT baseline (usable version for your ExpConformal pipeline):
      - calibration: store (z=x, e=|y-mu|)
      - start_test: train retrieval net to predict e_t from past (soft attention)
      - predict: compute attention weights over memory, derive threshold via weighted quantile of past errors
      - update: optionally append memory online
    """

    def __init__(
        self,
        alpha: float = 0.1,
        min_calib_size: int = 60,
        emb_dim: int = 32,
        hidden_dim: int = 64,
        train_epochs: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 64,
        beta: float = 1.0,
        seed: int = 0,
        online_update: bool = True,
        use_inf_mass: bool = False,
        inf_mass_fallback_to_max: bool = True,
        fallback_width: float = 3.0,
        M_lower: float = -1e6,
        M_upper: float = 1e6,
        **kwargs,
    ):
        self.cfg = HopCPTConfig(
            alpha=float(alpha),
            min_calib_size=int(min_calib_size),
            emb_dim=int(emb_dim),
            hidden_dim=int(hidden_dim),
            train_epochs=int(train_epochs),
            lr=float(lr),
            weight_decay=float(weight_decay),
            batch_size=int(batch_size),
            beta=float(beta),
            seed=int(seed),
            online_update=bool(online_update),
            use_inf_mass=bool(use_inf_mass),
            inf_mass_fallback_to_max=bool(inf_mass_fallback_to_max),
            fallback_width=float(fallback_width),
            M_lower=float(M_lower),
            M_upper=float(M_upper),
        )

        self.initial_alpha = float(alpha)
        self.alpha = float(alpha)

        self._Z: List[np.ndarray] = []
        self._E: List[float] = []

        self._frozen = False
        self._net: Optional[_HopCPTNet] = None
        self._in_dim: Optional[int] = None
        self._device = None

    def initialize(self, initial_data=None):
        self.alpha = self.initial_alpha
        self._Z = []
        self._E = []
        self._frozen = False
        self._net = None
        self._in_dim = None
        self._device = None

    def start_test(self):
        if torch is None:
            # no torch: keep a usable fallback (uniform weights)
            self._frozen = True
            return

        n = len(self._E)
        if n < self.cfg.min_calib_size:
            raise ValueError(f"HopCPT: not enough calibration samples: {n} < {self.cfg.min_calib_size}")

        Z = np.asarray(self._Z, dtype=np.float32)
        E = np.asarray(self._E, dtype=np.float32)
        self._in_dim = int(Z.shape[1])

        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        self._device = torch.device("cpu")
        self._net = _HopCPTNet(self._in_dim, self.cfg.emb_dim, self.cfg.hidden_dim).to(self._device)

        opt = torch.optim.Adam(self._net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        Zt = torch.from_numpy(Z).to(self._device)  # (n,din)
        Et = torch.from_numpy(E).to(self._device)  # (n,)

        idx_all = np.arange(1, n, dtype=int)
        bs = max(1, int(self.cfg.batch_size))

        for _ in range(int(self.cfg.train_epochs)):
            np.random.shuffle(idx_all)
            # encode once per epoch (big speedup)
            X = self._net.encode(Zt)  # (n, emb)

            for start in range(0, len(idx_all), bs):
                batch_idx = idx_all[start:start + bs]
                if batch_idx.size == 0:
                    continue

                loss = 0.0
                cnt = 0

                for j in batch_idx:
                    if j <= 0:
                        continue
                    q = X[j]        # (emb,)
                    K = X[:j]       # (j, emb)
                    v = Et[:j]      # (j,)

                    logits = self._net.logits(q, K).squeeze(0)  # (j,)
                    a = torch.softmax(self.cfg.beta * logits, dim=0)
                    e_hat = torch.sum(a * v)
                    loss = loss + (e_hat - Et[j]) ** 2
                    cnt += 1

                if cnt == 0:
                    continue

                loss = loss / cnt
                opt.zero_grad()
                loss.backward()
                opt.step()

        self._frozen = True

    def _compute_attention_weights(self, z_query: np.ndarray) -> np.ndarray:
        """
        Return normalized attention weights over memory, sum to 1.
        """
        n = len(self._E)
        if n == 0:
            return np.zeros((0,), dtype=float)

        if torch is None or self._net is None or self._in_dim is None:
            # uniform weights
            w = np.full(n, 1.0 / n, dtype=float)
            return w

        zq = np.asarray(z_query, dtype=np.float32).reshape(1, -1)
        Zm = np.asarray(self._Z, dtype=np.float32)

        with torch.no_grad():
            Zt = torch.from_numpy(Zm).to(self._device)  # (n, din)
            q = torch.from_numpy(zq).to(self._device)   # (1, din)

            X = self._net.encode(Zt)         # (n, emb)
            qx = self._net.encode(q)         # (1, emb)
            logits = self._net.logits(qx.squeeze(0), X).squeeze(0)  # (n,)
            logits = (self.cfg.beta * logits).cpu().numpy().astype(float)

        # stable softmax
        logits = logits - np.max(logits)
        omega = np.exp(logits)
        s = float(np.sum(omega))
        if (not np.isfinite(s)) or s <= self.cfg.eps:
            return np.full(n, 1.0 / n, dtype=float)

        w = omega / s
        return w.astype(float)

    def predict(
        self,
        base_prediction: Number,
        model_uncertainty: Number = 1.0,
        x=None,
        **kwargs,
    ) -> Tuple[float, float]:
        mu = float(base_prediction)
        unc = float(model_uncertainty)

        if x is None:
            w = self.cfg.fallback_width * max(1e-12, unc)
            return mu - w, mu + w

        if (not self._frozen) or (len(self._E) < self.cfg.min_calib_size):
            w = self.cfg.fallback_width * max(1e-12, unc)
            return mu - w, mu + w

        z = np.asarray(x, dtype=np.float32).reshape(-1)
        errs = np.asarray(self._E, dtype=float)

        w_attn = self._compute_attention_weights(z_query=z)  # sum=1

        q_level = 1.0 - float(self.alpha)

        if not self.cfg.use_inf_mass:
            # ---- usable HopCPT baseline (no +∞ mass) ----
            qhat = weighted_quantile(errs, w_attn, q=q_level)
        else:
            # ---- optional nonexchangeable-style +∞ mass version ----
            # construct (w_i, w_inf) using denom=sum(omega)+1 logic,
            # but we only have normalized weights; convert back:
            # simplest: set w_inf = 1/(n+1), and shrink finite mass accordingly
            # (this preserves "some" inf-mass behavior without exploding too easily)
            n = len(errs)
            w_inf = 1.0 / float(n + 1)
            w_fin = w_attn * (1.0 - w_inf)

            qhat = weighted_quantile_with_inf(errs, w_fin, q=q_level, w_inf=w_inf)
            if not np.isfinite(qhat):
                if self.cfg.inf_mass_fallback_to_max:
                    # engineering fallback: keep experiments comparable
                    qhat = float(np.max(errs))
                else:
                    return float("-inf"), float("inf")

        if (not np.isfinite(qhat)) or qhat < 0:
            qhat = float(np.max(errs)) if errs.size else self.cfg.fallback_width * max(1e-12, unc)

        lo = mu - float(qhat)
        hi = mu + float(qhat)

        # engineering bounds (so you不会再出现 avg_width=inf / 图炸掉)
        lo = self.cfg.M_lower if (not np.isfinite(lo)) else float(np.clip(lo, self.cfg.M_lower, self.cfg.M_upper))
        hi = self.cfg.M_upper if (not np.isfinite(hi)) else float(np.clip(hi, self.cfg.M_lower, self.cfg.M_upper))
        if hi < lo:
            lo, hi = hi, lo
        return lo, hi

    def update(
        self,
        y_true: Number,
        y_pred: Number,
        prediction_interval=None,
        interval=None,
        x=None,
        **kwargs,
    ):
        if x is None:
            return

        yt = float(y_true)
        mu = float(y_pred)
        err = abs(yt - mu)

        z = np.asarray(x, dtype=np.float32).reshape(-1)

        # collect during calibration
        if not self._frozen:
            self._Z.append(z)
            self._E.append(float(err))
            return

        # after start_test, optionally update memory online
        if self.cfg.online_update:
            self._Z.append(z)
            self._E.append(float(err))
