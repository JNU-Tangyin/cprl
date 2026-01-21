# base_conformal/cem.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np


@dataclass
class CEMConfig:
    alpha_min: float = 0.01
    alpha_max: float = 0.5
    pop_size: int = 20
    elite_frac: float = 0.2
    init_std: float = 0.05
    min_std: float = 0.01
    smooth: float = 0.7
    # 每次 step 内部做多少次 CEM 迭代（>1 更“严格”，但更耗时）
    n_iters: int = 1


class CEMAlpha:
    """严格 CEM：每次 step 做 sample->eval->elite->update。"""

    def __init__(self, alpha_init: float, cfg: CEMConfig) -> None:
        self.cfg = cfg
        self.mu = float(np.clip(alpha_init, cfg.alpha_min, cfg.alpha_max))
        self.sigma = float(cfg.init_std)

    def sample_population(self) -> np.ndarray:
        sigma = max(float(self.sigma), float(self.cfg.min_std))
        pop = np.random.normal(float(self.mu), sigma, size=int(self.cfg.pop_size))
        pop = np.clip(pop, float(self.cfg.alpha_min), float(self.cfg.alpha_max))
        return pop.astype(float)

    def step(self, eval_fn: Callable[[float], float]) -> float:
        """
        严格 CEM 更新：采样一批候选 alpha，离线评估 reward，选 elite 更新 mu/sigma。
        eval_fn(alpha) -> reward（越大越好）
        返回：更新后的 mu（“建议点”）
        """
        n_iters = max(1, int(self.cfg.n_iters))

        for _ in range(n_iters):
            pop = self.sample_population()
            rewards = np.asarray([float(eval_fn(float(a))) for a in pop], dtype=float)
            # 数值稳健：NaN/Inf 一律视为极差
            rewards = np.nan_to_num(rewards, nan=-1e18, posinf=-1e18, neginf=-1e18)

            k = max(1, int(np.ceil(len(pop) * float(self.cfg.elite_frac))))
            elite = pop[np.argsort(rewards)[-k:]]

            new_mu = float(np.mean(elite))
            new_sigma = float(max(np.std(elite), float(self.cfg.min_std)))

            self.mu = float(float(self.cfg.smooth) * self.mu + (1.0 - float(self.cfg.smooth)) * new_mu)
            self.sigma = float(float(self.cfg.smooth) * self.sigma + (1.0 - float(self.cfg.smooth)) * new_sigma)

            self.mu = float(np.clip(self.mu, float(self.cfg.alpha_min), float(self.cfg.alpha_max)))
            self.sigma = float(max(self.sigma, float(self.cfg.min_std)))

        return float(self.mu)

    def choose_alpha(self) -> float:
        """
        选项1：执行阶段从当前分布采样一个 alpha（探索）。
        """
        sigma = max(float(self.sigma), float(self.cfg.min_std))
        a = float(np.random.normal(float(self.mu), sigma))
        return float(np.clip(a, float(self.cfg.alpha_min), float(self.cfg.alpha_max)))


class RegimeCEMAlphaController:
    """
    多分布 / 条件化 CEM：
    - 每个 regime_id 一个 CEMAlpha
    - 可选 global fallback：每次 step 也更新 global，choose_alpha(regime=None) 走 global
    """

    def __init__(
        self,
        n_regimes: int,
        alpha_init: float,
        cfg: CEMConfig,
        use_global_fallback: bool = True,
    ) -> None:
        self.cfg = cfg
        self.n_regimes = int(n_regimes)
        self.models: Dict[int, CEMAlpha] = {k: CEMAlpha(alpha_init, cfg) for k in range(self.n_regimes)}
        self.global_model: Optional[CEMAlpha] = CEMAlpha(alpha_init, cfg) if use_global_fallback else None

    def _clip_rid(self, rid: int) -> int:
        return max(0, min(self.n_regimes - 1, int(rid)))

    def choose_alpha(self, regime_id: Optional[int]) -> float:
        if regime_id is None:
            if self.global_model is not None:
                return self.global_model.choose_alpha()
            return self.models[0].choose_alpha()

        rid = self._clip_rid(int(regime_id))
        return self.models[rid].choose_alpha()

    def step(self, regime_id: Optional[int], eval_fn: Callable[[float], float]) -> float:
        """
        严格 CEM 更新：对 global（若启用）+ 对应 regime 各做一次 step。
        返回：该 regime 更新后的 mu（建议点）
        """
        if self.global_model is not None:
            _ = self.global_model.step(eval_fn)

        if regime_id is None:
            if self.global_model is not None:
                return float(self.global_model.mu)
            return float(self.models[0].mu)

        rid = self._clip_rid(int(regime_id))
        return float(self.models[rid].step(eval_fn))

    def get_mu(self, regime_id: Optional[int]) -> float:
        if regime_id is None:
            if self.global_model is not None:
                return float(self.global_model.mu)
            return float(self.models[0].mu)

        rid = self._clip_rid(int(regime_id))
        return float(self.models[rid].mu)
