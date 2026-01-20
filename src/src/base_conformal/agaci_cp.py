# src/base_conformal/agaci_cp.py
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

from .aci_cp import ACICP  


def pinball_loss(y: float, q: float, beta: float) -> float:
    """
    论文里用于 AgACI 的 pinball loss ρ_β（quantile regression 常用）:
      ρ_β(u) = β*u - min(u, 0), 其中 u = y - q
    """
    u = float(y - q)
    return float(beta * u - min(u, 0.0))


@dataclass
class AgACIConfig:
    alpha: float = 0.1
    gammas: Tuple[float, ...] = (0.001, 0.01, 0.05, 0.1)
    warmup_steps: int = 50

    # ACI(每个专家)内部参数（沿用你的 ACICP）
    split_size: float = 0.75
    t_init: int = 200
    max_iter: int = 500
    seed: int = 0

    M_lower: float = -1e9
    M_upper: float =  1e9

    # 聚合器（BOA-style）数值稳定参数
    eps: float = 1e-12
    clip_eta_max: float = 50.0


class _BOAStyleAggregator:
    """
    BOA-style 在线聚合器（用于 lower 或 upper 一边）：BOA-like（二阶自适应）实现，核心目标：
      - 用 pinball loss 更新专家权重
      - 权重依赖历史损失（类似 Bernstein/second-order）
    """
    def __init__(self, n_experts: int, eps: float = 1e-12, clip_eta_max: float = 50.0):
        self.K = int(n_experts)
        self.eps = float(eps)
        self.clip_eta_max = float(clip_eta_max)

        self.w = np.full(self.K, 1.0 / self.K, dtype=float)  # weights (probabilities)
        self.cum_sq = np.zeros(self.K, dtype=float)           # sum of squared centered losses
        self.cum_l = np.zeros(self.K, dtype=float)            # cumulated "l-values" (BOA-like)
        self.max_abs = np.zeros(self.K, dtype=float)          # track max abs loss for bounding
        self.eta = np.zeros(self.K, dtype=float)              # adaptive learning rates

    def reset(self):
        self.w[:] = 1.0 / self.K
        self.cum_sq[:] = 0.0
        self.cum_l[:] = 0.0
        self.max_abs[:] = 0.0
        self.eta[:] = 0.0

    def aggregate(self, preds: np.ndarray) -> float:
        # preds: (K,)
        return float(np.sum(self.w * preds))

    def update(self, losses: np.ndarray):
        """
        losses: (K,) 每个专家在当前时刻的 loss（pinball）
        我们用“相对损失”来更新，避免所有专家同向漂移导致数值不稳定。
        """
        losses = np.asarray(losses, dtype=float).reshape(-1)
        assert losses.shape[0] == self.K

        # centered loss
        loss_bar = float(np.sum(self.w * losses))
        ell = losses - loss_bar  # (K,)

        # second-order stats
        self.cum_sq += ell ** 2
        self.max_abs = np.maximum(self.max_abs, np.abs(ell))

        # BOA-like bounding term e_k (binary scale), 对应论文里需要 bounded experts/loss
        e_vals = 2.0 ** (np.ceil(np.log2(self.max_abs + self.eps)) + 1.0)

        # adaptive eta
        logK = np.log(self.K + 1.0)
        eta = np.minimum(1.0 / (e_vals + self.eps),
                         np.sqrt(logK / (self.cum_sq + self.eps)))
        eta = np.clip(eta, 0.0, self.clip_eta_max)
        self.eta = eta

        # l-values update（BOA-like）
        self.cum_l += 0.5 * (ell * (1.0 + eta * ell) + e_vals * (eta * ell > 0.5))

        # weights update: w_k ∝ eta_k * exp(- eta_k * L_k)
        z = -eta * self.cum_l
        z -= np.max(z)  # stabilize
        w_unnorm = eta * np.exp(z)
        s = float(np.sum(w_unnorm))
        if s <= 0 or not np.isfinite(s):
            self.w[:] = 1.0 / self.K
        else:
            self.w[:] = w_unnorm / (s + self.eps)


class AgACICP:
    """
    AgACI：K 个 ACI(γ_k) 专家 + 对 lower/upper 两套独立在线聚合
    - 每一步：先用每个专家给出 [l_k, u_k]
    - 进行 bounded thresholding（Algorithm 1 line 5） :contentReference[oaicite:4]{index=4}
    - 用 pinball loss 更新 lower/upper 各自聚合器（Algorithm 1 line 9-11） :contentReference[oaicite:5]{index=5}
    """
    def __init__(
        self,
        alpha: float = 0.1,
        gammas: Optional[List[float]] = None,
        split_size: float = 0.75,
        t_init: int = 200,
        max_iter: int = 500,
        warmup_steps: int = 50,
        M_lower: float = -1e9,
        M_upper: float = 1e9,
        seed: int = 0,
    ):
        self.cfg = AgACIConfig(
            alpha=float(alpha),
            gammas=tuple(gammas) if gammas is not None else AgACIConfig().gammas,
            warmup_steps=int(warmup_steps),
            split_size=float(split_size),
            t_init=int(t_init),
            max_iter=int(max_iter),
            M_lower=float(M_lower),
            M_upper=float(M_upper),
            seed=int(seed),
        )

        self.initial_alpha = float(alpha)
        self.alpha = float(alpha)  # AgACI 不更新 alpha；这里只是保持接口一致

        self.gammas = list(self.cfg.gammas)
        self.K = len(self.gammas)

        # K 个 ACI 专家（各自维护自己的 alpha_t 和历史）
        self.experts: List[ACICP] = []
        for k, g in enumerate(self.gammas):
            self.experts.append(
                ACICP(
                    alpha=self.cfg.alpha,
                    lr=float(g),                 # 在你的 ACICP 里 lr 被当成 gamma
                    split_size=self.cfg.split_size,
                    t_init=self.cfg.t_init,
                    max_iter=self.cfg.max_iter,
                    seed=self.cfg.seed + 97 * (k + 1),
                )
            )

        # 两个独立聚合器：lower / upper（分别聚合）
        self.agg_low = _BOAStyleAggregator(self.K)
        self.agg_up  = _BOAStyleAggregator(self.K)

        # pinball 参数：β^(l)=α/2, β^(u)=1-α/2
        self.beta_low = self.cfg.alpha / 2.0
        self.beta_up  = 1.0 - self.cfg.alpha / 2.0

        self.t = 0  # step counter

    def initialize(self, initial_data=None):
        self.alpha = self.initial_alpha
        self.t = 0
        self.agg_low.reset()
        self.agg_up.reset()
        for e in self.experts:
            e.initialize(initial_data=initial_data)

    def predict(self, base_prediction=None, model_uncertainty=1.0, x=None, **kwargs):
        if x is None:
            raise ValueError("AgACICP.predict requires x=... (feature vector).")

        # warmup：前 warmup_steps 用“均匀权重”（或你也可以固定选 γ=0 的专家）
        use_uniform = (self.t < self.cfg.warmup_steps)
        if use_uniform:
            w_low = np.full(self.K, 1.0 / self.K, dtype=float)
            w_up  = np.full(self.K, 1.0 / self.K, dtype=float)
        else:
            w_low = self.agg_low.w
            w_up  = self.agg_up.w

        lows = np.zeros(self.K, dtype=float)
        ups  = np.zeros(self.K, dtype=float)

        for k, e in enumerate(self.experts):
            l_k, u_k = e.predict(base_prediction=base_prediction, model_uncertainty=model_uncertainty, x=x)

            # bounded thresholding
            if not np.isfinite(l_k):
                l_k = self.cfg.M_lower
            if not np.isfinite(u_k):
                u_k = self.cfg.M_upper

            l_k = float(np.clip(l_k, self.cfg.M_lower, self.cfg.M_upper))
            u_k = float(np.clip(u_k, self.cfg.M_lower, self.cfg.M_upper))

            lows[k] = l_k
            ups[k]  = u_k

        # weighted mean aggregation
        low_agg = float(np.sum(w_low * lows) / (np.sum(w_low) + 1e-12))
        up_agg  = float(np.sum(w_up  * ups)  / (np.sum(w_up)  + 1e-12))

        return low_agg, up_agg

    def update(self, y_true, y_pred=None, prediction_interval=None, interval=None, x=None, **kwargs):
        if prediction_interval is None:
            prediction_interval = interval
        if prediction_interval is None:
            raise ValueError("AgACICP.update requires prediction_interval (or interval).")
        if x is None:
            raise ValueError("AgACICP.update requires x=... (feature vector).")

        y_true = float(y_true)
        low_agg, up_agg = prediction_interval

        # 1) 先让每个 ACI 专家用自己的区间更新自己的 alpha / 历史（内部alpha 更新）
        expert_lows = np.zeros(self.K, dtype=float)
        expert_ups  = np.zeros(self.K, dtype=float)

        for k, e in enumerate(self.experts):
            l_k, u_k = e.predict(x=x)  # 用专家自己的当前状态再算一次该步区间
            if not np.isfinite(l_k):
                l_k = self.cfg.M_lower
            if not np.isfinite(u_k):
                u_k = self.cfg.M_upper
            l_k = float(np.clip(l_k, self.cfg.M_lower, self.cfg.M_upper))
            u_k = float(np.clip(u_k, self.cfg.M_lower, self.cfg.M_upper))

            expert_lows[k] = l_k
            expert_ups[k]  = u_k

            # 更新专家（会推进它的历史，并更新它自己的 alpha）
            e.update(y_true=y_true, prediction_interval=(l_k, u_k), x=x)

        # 2) 计算 pinball losses（lower/upper 独立） 
        loss_low = np.array([pinball_loss(y_true, q=expert_lows[k], beta=self.beta_low) for k in range(self.K)], dtype=float)
        loss_up  = np.array([pinball_loss(y_true, q=expert_ups[k],  beta=self.beta_up)  for k in range(self.K)], dtype=float)

        # 3) 更新聚合权重（BOA-style）
        if self.t >= self.cfg.warmup_steps:
            self.agg_low.update(loss_low)
            self.agg_up.update(loss_up)

        self.t += 1
