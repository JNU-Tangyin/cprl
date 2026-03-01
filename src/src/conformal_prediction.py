# conformal_prediction.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque

import numpy as np

@dataclass
class ConformalPredictionConfig:
    initial_alpha: float = 0.1
    target_coverage: float = 0.9
    window_size: int = 64

    # --- adaptive regime discovery ---
    max_regimes: int = 8                
    new_regime_threshold: float = 2.2
    new_regime_patience: int = 3      

    sticky_bonus: float = 0.5             
    min_state_duration: int = 5         

    ewma_beta: float = 0.94         
    jump_q: float = 0.95            
    feature_ema: float = 0.05         

    # sliding windows
    calib_window_size: int = 150
    min_calib_size: int = 20
    min_regime_calib_size: int = 20

    # spectral term weight
    lambda_spectral: float = 0.5
    min_spectral_size: int = 30

    # per-regime fallback thresholds
    min_regime_eval_size: int = 20
    min_regime_cov_size: int = 20

    # CEM replay buffer
    eval_window_size: int = 150

    # coverage penalty window
    coverage_window: int = 50
    lambda_cov: float = 5.0

    # refresh k in steps
    k_update_every: int = 20
    k_min: float = 1e-3
    k_max: float = 100.0
    k_fallback: float = 1.0

    # --- CEM hyper ---
    alpha_min: float = 0.01
    alpha_max: float = 0.3
    cem_pop: int = 16
    cem_elite_frac: float = 0.25
    cem_noise: float = 0.02
    cem_lr: float = 0.3
    cem_n_iters: int = 3

    use_spectral: bool = True
    use_regime: bool = True
    use_cem: bool = True
    width_weight: float = 1.0


class _SpectralDrift:
    def __init__(self, window_size: int):
        self.window_size = int(window_size)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) == 0 or len(y) == 0:
            return 0.0

        # FFT → power spectrum
        fx = np.abs(np.fft.rfft(x)) ** 2
        fy = np.abs(np.fft.rfft(y)) ** 2

        sx = fx.sum()
        sy = fy.sum()
        if sx <= 1e-12 or sy <= 1e-12:
            return 0.0

        px = fx / sx
        py = fy / sy

        # Wasserstein-1 (1D closed form)
        cdf_x = np.cumsum(px)
        cdf_y = np.cumsum(py)

        return float(np.mean(np.abs(cdf_x - cdf_y)))


class _AlphaController:
    """
    对齐 RegimeCEMAlphaController(use_global_fallback=True)：

    - choose(rid, use_regime): 若use_regime=False则用global分布；否则用rid分布
    - step(rid, objective_fn, update_regime): global总是更新；rid仅当update_regime=True才更新
    - 每次更新做 cem_n_iters 轮 CEM 迭代（不是只更新一轮）
    """

    def __init__(self, n_regimes: int, alpha_init: float, cfg: ConformalPredictionConfig):
        self.cfg = cfg
        self.n_regimes = int(n_regimes)

        # per-regime
        self.mu = np.full(self.n_regimes, float(alpha_init), dtype=float)
        self.sigma = np.full(self.n_regimes, float(cfg.cem_noise), dtype=float)

        # global fallback
        self.mu_global = float(alpha_init)
        self.sigma_global = float(cfg.cem_noise)

    def _sample(self, mu: float, sigma: float) -> np.ndarray:
        sigma = max(float(sigma), float(self.cfg.cem_noise), 1e-6)
        pop = np.random.normal(float(mu), sigma, size=int(self.cfg.cem_pop))
        pop = np.clip(pop, float(self.cfg.alpha_min), float(self.cfg.alpha_max))
        return pop.astype(float)

    def choose(self, rid: int, use_regime: bool) -> float:
        """
        执行动作：若use_regime=False -> global；否则 -> rid
        """
        rid = int(rid)
        if not use_regime:
            mu, sigma = float(self.mu_global), float(self.sigma_global)
        else:
            mu, sigma = float(self.mu[rid]), float(self.sigma[rid])

        sigma = max(sigma, float(self.cfg.cem_noise), 1e-6)
        a = np.random.normal(mu, sigma)
        return float(np.clip(a, float(self.cfg.alpha_min), float(self.cfg.alpha_max)))

    def step(self, rid: int, objective_g, objective_r, update_regime: bool) -> Tuple[float, float]:
        """
        更新：global总更新；regime仅当update_regime=True更新
        返回：(mu_global, mu_rid_used)
        """
        rid = int(rid)

        # 1) global always updated
        self.mu_global, self.sigma_global = self._cem_update(
            self.mu_global, self.sigma_global, objective_g
        )

        # 2) regime optionally updated
        if update_regime:
            self.mu[rid], self.sigma[rid] = self._cem_update(
                self.mu[rid], self.sigma[rid], objective_r
            )

        mu_r = float(self.mu[rid]) if update_regime else float(self.mu_global)
        return float(self.mu_global), float(mu_r)

    def _cem_update(self, mu: float, sigma: float, objective_fn) -> Tuple[float, float]:
        mu_new, sigma_new = float(mu), float(sigma)
        n_iters = max(1, int(getattr(self.cfg, "cem_n_iters", 1)))

        for _ in range(n_iters):
            pop = self._sample(mu_new, sigma_new)

            rewards = np.asarray([float(objective_fn(float(a))) for a in pop], dtype=float)
            rewards = np.nan_to_num(rewards, nan=-1e18, posinf=-1e18, neginf=-1e18)

            k = max(1, int(np.ceil(len(pop) * float(self.cfg.cem_elite_frac))))
            elite = pop[np.argsort(rewards)[-k:]]

            elite_mu = float(np.mean(elite))
            elite_sigma = float(max(np.std(elite), 1e-6))

            # 平滑更新（类似 cem.py 的 smooth）
            smooth = float(self.cfg.cem_lr)  # 你这里 cem_lr 实际扮演 smooth
            mu_new = smooth * mu_new + (1.0 - smooth) * elite_mu
            sigma_new = smooth * sigma_new + (1.0 - smooth) * elite_sigma

            mu_new = float(np.clip(mu_new, float(self.cfg.alpha_min), float(self.cfg.alpha_max)))
            sigma_new = float(max(sigma_new, float(self.cfg.cem_noise), 1e-6))

        return mu_new, sigma_new


class _AdaptiveRegimeKernel:
    """
    Online regime discovery for financial series:
    robust volatility + jump rate + dependence + sticky/hysteresis.
    Returns rid in [0, K-1], where K grows up to cfg.max_regimes.
    """
    def __init__(self, cfg: ConformalPredictionConfig, random_seed: int = 42):
        self.cfg = cfg

        self.K = 0
        self.centers: List[np.ndarray] = []
        self.counts: List[int] = []

        self.prev_rid: Optional[int] = None
        self.dwell: int = 0
        self._new_candidate_hits = 0

        self._ewma_var: Optional[float] = None

    def _mad(self, x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return 0.0
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        return float(1.4826 * mad + 1e-6)

    def _ewma_vol(self, x: np.ndarray) -> float:
        beta = float(self.cfg.ewma_beta)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return float(np.sqrt(self._ewma_var)) if self._ewma_var else 0.0
        if self._ewma_var is None:
            self._ewma_var = float(np.mean(x**2) + 1e-6)
        for v in x[-min(10, len(x)):]:
            self._ewma_var = beta * self._ewma_var + (1.0 - beta) * float(v**2)
        return float(np.sqrt(self._ewma_var + 1e-12))

    def _jump_rate(self, x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        if len(x) < 10:
            return 0.0
        thr = float(np.quantile(np.abs(x), float(self.cfg.jump_q)))
        if thr <= 1e-12:
            return 0.0
        return float(np.mean(np.abs(x) > thr))

    def _acf1(self, x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        if len(x) < 6:
            return 0.0
        a = x[:-1] - np.mean(x[:-1])
        b = x[1:]  - np.mean(x[1:])
        denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
        return float(np.dot(a, b) / denom)

    def _features(self, price_window: np.ndarray) -> np.ndarray:
        """
        Market-state features from past price window.

        price_window: shape (L,) = lagged observed values (e.g., exchange rate level)
        We use log-return / volatility / jump / autocorr / trend slope.
        """

        p = np.asarray(price_window, dtype=float).reshape(-1)
        p = p[np.isfinite(p)]
        if p.size < 12:
            return np.zeros(5, dtype=float)

        # log returns (robust to scale)
        eps = 1e-8
        r = np.diff(np.log(np.clip(p, eps, None)))
        r = r[np.isfinite(r)]
        if r.size < 10:
            return np.zeros(5, dtype=float)

        # 1) robust vol (MAD on returns)
        med = np.median(r)
        mad = np.median(np.abs(r - med))
        vol_rob = 1.4826 * mad + 1e-6

        # 2) EWMA vol (on returns)
        vol_ewma = self._ewma_vol(r)

        # 3) jump rate (on returns)
        jump = self._jump_rate(r)

        # 4) ACF1 of returns
        ac1 = self._acf1(r)

        # 5) trend slope of price level (normalized)
        t = np.arange(p.size, dtype=float)
        t = t - t.mean()
        denom = (np.dot(t, t) + 1e-6)
        slope = float(np.dot(t, (p - p.mean())) / denom)
        slope = slope / (np.std(p) + 1e-6)

        return np.array([
            np.log1p(vol_rob),
            np.log1p(vol_ewma),
            jump,
            ac1,
            slope,
        ], dtype=float)
        

    def _dist2(self, a: np.ndarray, b: np.ndarray) -> float:
        d = a - b
        return float(np.dot(d, d))

    def _assign(self, f: np.ndarray) -> Tuple[int, float]:
        if self.K == 0:
            self.K = 1
            self.centers = [f.copy()]
            self.counts = [0]
            return 0, 0.0

        d2 = np.array([self._dist2(f, c) for c in self.centers], dtype=float)
        j = int(np.argmin(d2))
        dmin = float(np.sqrt(max(d2[j], 0.0)))
        return j, dmin

    def _update_and_get_regime(self, price_window: np.ndarray) -> int:
        f = self._features(price_window)
        j, dmin = self._assign(f)

        # sticky preference
        if self.prev_rid is not None and self.prev_rid < self.K:
            stay = int(self.prev_rid)
            d_stay = float(np.sqrt(max(self._dist2(f, self.centers[stay]), 0.0)))
            d_stay_eff = max(0.0, d_stay - float(self.cfg.sticky_bonus))
            if d_stay_eff <= dmin:
                j, dmin = stay, d_stay_eff

        # minimum dwell time
        if self.prev_rid is not None and self.dwell < int(self.cfg.min_state_duration):
            j = int(self.prev_rid)

        # new regime creation with patience (anti-spike)
        if (dmin > float(self.cfg.new_regime_threshold)) and (self.K < int(self.cfg.max_regimes)):
            self._new_candidate_hits += 1
            if self._new_candidate_hits >= int(self.cfg.new_regime_patience):
                j = self.K
                self.K += 1
                self.centers.append(f.copy())
                self.counts.append(0)
                self._new_candidate_hits = 0
        else:
            self._new_candidate_hits = 0

        # update center
        eta = float(self.cfg.feature_ema)
        self.centers[j] = (1 - eta) * self.centers[j] + eta * f
        self.counts[j] += 1

        # dwell update
        if self.prev_rid is None or int(j) != int(self.prev_rid):
            self.dwell = 1
        else:
            self.dwell += 1

        self.prev_rid = int(j)
        return int(j)

class AdaptiveConformalPredictor:
    def __init__(self, config: Optional[ConformalPredictionConfig] = None) -> None:
        self.config = config or ConformalPredictionConfig()

        R = int(self.config.max_regimes)

        # internal "flow" components (merged)
        self._drift = _SpectralDrift(window_size=int(self.config.window_size))
        self._regime = _AdaptiveRegimeKernel(cfg=self.config)
        self._alpha = _AlphaController(n_regimes=R, alpha_init=float(self.config.initial_alpha), cfg=self.config)

        # state
        self.current_state: Optional[int] = None
        self.state_history: List[int] = []

        # rolling errors
        self.prediction_errors: Deque[float] = deque(maxlen=int(self.config.window_size)+1)

        # buffers: global + per regime
        self._init_buffers()

        self.alpha_history: List[float] = []
        self.k_history: List[float] = []
        self.spectral_q_history: List[float] = []
        self.use_regime_history: List[bool] = []

    def _init_buffers(self) -> None:
        R = int(self.config.max_regimes)
        Wc = int(self.config.calib_window_size)

        self.calib_e_lo_global: Deque[float] = deque(maxlen=Wc)
        self.calib_e_hi_global: Deque[float] = deque(maxlen=Wc)
        self.calib_s_global: Deque[float] = deque(maxlen=Wc)

        self.calib_e_lo_by_regime: Dict[int, Deque[float]] = {r: deque(maxlen=Wc) for r in range(R)}
        self.calib_e_hi_by_regime: Dict[int, Deque[float]] = {r: deque(maxlen=Wc) for r in range(R)}
        self.calib_s_by_regime: Dict[int, Deque[float]] = {r: deque(maxlen=Wc) for r in range(R)}

        self.eval_buffer_global: Deque[Tuple[float, float]] = deque(maxlen=int(self.config.eval_window_size))
        self.eval_buffer_by_regime: Dict[int, Deque[Tuple[float, float]]] = {
            r: deque(maxlen=int(self.config.eval_window_size)) for r in range(R)
        }

        self.cover_hist_global: Deque[float] = deque(maxlen=int(self.config.coverage_window))
        self.cover_hist_by_regime: Dict[int, Deque[float]] = {r: deque(maxlen=int(self.config.coverage_window)) for r in range(R)}

        # k scales
        self._k_scale_global: float = float(self.config.k_fallback)
        self._k_scale_by_regime: Dict[int, float] = {r: float(self.config.k_fallback) for r in range(R)}
        self._k_t_global: int = 0
        self._k_t_by_regime: Dict[int, int] = {r: 0 for r in range(R)}

    @property
    def alpha(self) -> float:
        if hasattr(self, "alpha_history") and len(self.alpha_history) > 0:
            return float(self.alpha_history[-1])
        return float(self.config.initial_alpha)

    def _use_regime(self, rid: int) -> bool:
        """Whether regime-specific stats are trusted (sample size gates)."""
        if not self.config.use_regime:
            return False
        rid = int(rid)
        
        ok = (
            len(self.calib_e_lo_by_regime[rid]) >= int(self.config.min_regime_calib_size)
            and len(self.calib_e_hi_by_regime[rid]) >= int(self.config.min_regime_calib_size)
            and len(self.eval_buffer_by_regime[rid]) >= int(self.config.min_regime_eval_size)
            and len(self.cover_hist_by_regime[rid]) >= int(self.config.min_regime_cov_size)
        )
        if not ok:
            return False

        if bool(self.config.use_spectral):
            return len(self.calib_s_by_regime[rid]) >= int(self.config.min_regime_calib_size)

        return True

    def _buffers_global(self):
        """Return global buffers (no gating)."""
        return (
            self.calib_e_lo_global,
            self.calib_e_hi_global,
            self.calib_s_global,
            self.eval_buffer_global,
            self.cover_hist_global,
            self._k_scale_global,
            )

    def _buffers_regime(self, rid: int):
        """Return regime buffers for rid (no gating)."""
        rid = int(rid)
        return (
            self.calib_e_lo_by_regime[rid],
            self.calib_e_hi_by_regime[rid],
            self.calib_s_by_regime[rid],
            self.eval_buffer_by_regime[rid],
            self.cover_hist_by_regime[rid],
            self._k_scale_by_regime[rid],
        )
        
    # ---------- k refresh ----------
    def _maybe_refresh_k(self, rid: int) -> None:
        def compute_k(e_lo_buf, e_hi_buf, s_buf) -> float:
            if len(e_lo_buf) < 10 or len(e_hi_buf) < 10 or len(s_buf) < 10:
                return float(self.config.k_fallback)

            e_lo = np.asarray(list(e_lo_buf), float)
            e_hi = np.asarray(list(e_hi_buf), float)
            s = np.asarray(list(s_buf), float)

            e = (e_lo + e_hi)
            e = e[np.isfinite(e)]
            s = s[np.isfinite(s)]
            if len(e) < 10 or len(s) < 10:
                return float(self.config.k_fallback)

            med_e = float(np.median(e))
            med_s = float(np.median(s))
            if (not np.isfinite(med_e)) or (not np.isfinite(med_s)) or (med_s <= 1e-12):
                k = float(self.config.k_fallback)
            else:
                k = med_e / med_s

            k = float(np.clip(k, float(self.config.k_min), float(self.config.k_max)))
            return float(k) if np.isfinite(k) else float(self.config.k_fallback)

        rid = int(rid)
        self._k_t_global += 1
        if self._k_t_global % int(self.config.k_update_every) == 0:
            self._k_scale_global = compute_k(self.calib_e_lo_global, self.calib_e_hi_global, self.calib_s_global)

        self._k_t_by_regime[rid] += 1
        if self._k_t_by_regime[rid] % int(self.config.k_update_every) == 0:
            self._k_scale_by_regime[rid] = compute_k(
                self.calib_e_lo_by_regime[rid], self.calib_e_hi_by_regime[rid], self.calib_s_by_regime[rid]
            )

    # ---------- margins ----------
    def _margins_from_buffers(
        self,
        e_lo_buf,
        e_hi_buf,
        s_buf,
        k_scale: float,
        alpha: float,
        model_uncertainty: float,
    ) -> Tuple[float, float, float]:
        a = float(np.clip(alpha, float(self.config.alpha_min), float(self.config.alpha_max)))

        # warm start
        if len(e_lo_buf) < int(self.config.min_calib_size) or len(e_hi_buf) < int(self.config.min_calib_size):
            scale = 1.0 / max(1e-6, 1.0 - a)
            m = float(model_uncertainty) * scale
            return float(m), float(m), 0.0

        e_lo = np.asarray(list(e_lo_buf), float)
        e_hi = np.asarray(list(e_hi_buf), float)
        e_lo = e_lo[np.isfinite(e_lo)]
        e_hi = e_hi[np.isfinite(e_hi)]
        if len(e_lo) == 0 or len(e_hi) == 0:
            m = float(model_uncertainty)
            return float(m), float(m), 0.0

        q_lo = float(np.quantile(e_lo, 1.0 - a))
        q_hi = float(np.quantile(e_hi, 1.0 - a))

        q_s = 0.0
        if len(s_buf) >= int(getattr(self.config, "min_spectral_size", self.config.min_calib_size)):
            s = np.asarray(list(s_buf), float)
            s = s[np.isfinite(s)]
            if len(s) > 0:
                q_s = float(np.quantile(s, 1.0 - a))

        extra = float(self.config.lambda_spectral) * float(k_scale) * float(q_s)
        if not self.config.use_spectral:
            extra = 0.0

        m_lo = max(0.0, q_lo + extra)
        m_hi = max(0.0, q_hi + extra)
        return float(m_lo), float(m_hi), float(q_s)

    def _margins_global(self, alpha: float, model_uncertainty: float) -> Tuple[float, float, float]:
        e_lo, e_hi, s, _, _, k = self._buffers_global()
        return self._margins_from_buffers(e_lo, e_hi, s, k, alpha, model_uncertainty)

    def _margins_regime(self, rid: int, alpha: float, model_uncertainty: float) -> Tuple[float, float, float]:
        e_lo, e_hi, s, _, _, k = self._buffers_regime(rid)
        return self._margins_from_buffers(e_lo, e_hi, s, k, alpha, model_uncertainty)

    # ---------- unified objective for alpha ----------
    def _objective_global(self):
        e_lo_buf, e_hi_buf, _, eval_buf, _, _ = self._buffers_global()

        def mu_unc() -> float:
            if len(e_lo_buf) >= 2 and len(e_hi_buf) >= 2:
                e = np.asarray(list(e_lo_buf), float) + np.asarray(list(e_hi_buf), float)
                e = e[np.isfinite(e)]
                if len(e) >= 5:
                    tail = e[-min(50, len(e)):]
                    return float(np.std(tail) + 1e-6)
            return 1.0
        
        unc = mu_unc()

        def obj(a: float) -> float:
            a = float(np.clip(a, float(self.config.alpha_min), float(self.config.alpha_max)))
            m_lo, m_hi, _ = self._margins_global(alpha=a, model_uncertainty=unc)
            width_const = float(m_lo + m_hi)
            if len(eval_buf) == 0:
                return -1e9

            winklers = []
            covered = []
            for yp, yt in eval_buf:
                lo = float(yp) - float(m_lo)
                hi = float(yp) + float(m_hi)
                cov = 1.0 if (lo <= float(yt) <= hi) else 0.0
                covered.append(cov)

                if float(yt) < lo:
                    miss = float(lo - float(yt))
                elif float(yt) > hi:
                    miss = float(float(yt) - hi)
                else:
                    miss = 0.0

                winklers.append(width_const + (2.0 / max(a, 1e-6)) * miss)

            c_hat = float(np.mean(covered))
            target = float(self.config.target_coverage)
            gap = max(0.0, target - c_hat)
            penalty = float(self.config.lambda_cov) * (gap ** 2)
            w = float(getattr(self.config, "width_weight", 1.0))
            return -(w * float(np.mean(winklers)) + penalty)

        return obj

    def _objective_regime(self, rid: int):
        rid = int(rid)
        e_lo_buf, e_hi_buf, _, eval_buf, _, _ = self._buffers_regime(rid)

        def mu_unc() -> float:
            if len(e_lo_buf) >= 2 and len(e_hi_buf) >= 2:
                e = np.asarray(list(e_lo_buf), float) + np.asarray(list(e_hi_buf), float)
                e = e[np.isfinite(e)]
                if len(e) >= 5:
                    tail = e[-min(50, len(e)):]
                    return float(np.std(tail) + 1e-6)
            return 1.0

        unc = mu_unc()

        def obj(a: float) -> float:
            a = float(np.clip(a, float(self.config.alpha_min), float(self.config.alpha_max)))
            m_lo, m_hi, _ = self._margins_regime(rid=rid, alpha=a, model_uncertainty=unc)

            width_const = float(m_lo + m_hi)
            if len(eval_buf) == 0:
                return -1e9

            winklers = []
            covered = []
            for yp, yt in eval_buf:
                lo = float(yp) - float(m_lo)
                hi = float(yp) + float(m_hi)
                cov = 1.0 if (lo <= float(yt) <= hi) else 0.0
                covered.append(cov)

                if float(yt) < lo:
                    miss = float(lo - float(yt))
                elif float(yt) > hi:
                    miss = float(float(yt) - hi)
                else:
                    miss = 0.0

                winklers.append(width_const + (2.0 / max(a, 1e-6)) * miss)

            c_hat = float(np.mean(covered))
            target = float(self.config.target_coverage)
            gap = max(0.0, target - c_hat)
            penalty = float(self.config.lambda_cov) * (gap ** 2)
            w = float(getattr(self.config, "width_weight", 1.0))
            return -(w * float(np.mean(winklers)) + penalty)

        return obj

    def _extract_price_window(self, x) -> Optional[np.ndarray]:
        if x is None:
            return None
        a = np.asarray(x, dtype=float)
        if a.ndim == 1:
            return a
        if a.ndim == 2:
            return a[:, 0]
        if a.ndim == 3:
            return a[0, :, 0]
        return a.reshape(-1)

    # ============================================================
    # Public API: one-step closed loop
    # =========================================================


    def reset(self) -> None:
        self.current_state = None
        self.state_history.clear()
        self.prediction_errors.clear()
        self._init_buffers()

    def predict(
        self,
        base_prediction: float = None,
        y_pred: float = None,
        model_uncertainty: float = None,
        uncertainty: float = None,
        **kwargs,
        ):

        if getattr(self, "_pending", None) is not None:
            self._dropped_pending = getattr(self, "_dropped_pending", 0) + 1
            self._pending = None

        yp = base_prediction if base_prediction is not None else y_pred
        unc = model_uncertainty if model_uncertainty is not None else uncertainty

        yp = float(yp)
        unc = float(unc) if unc is not None else 1.0

        x = kwargs.get("x", None)
        if x is None:
            x = kwargs.get("X", None)
        if x is None:
            x = kwargs.get("features", None)

        if x is not None:
            pw = self._extract_price_window(x)
            if pw is None or pw.size < 12:
                rid = 0
            else:
                rid = int(self._regime._update_and_get_regime(pw))
        else:
            rid = 0

        rid = int(max(0, min(self.config.max_regimes - 1, rid)))
        self.current_state = rid
        self.state_history.append(rid)

        use_r = bool(self._use_regime(rid))

        if not self.config.use_cem:
            alpha = float(self.config.initial_alpha)
        else:
            alpha = float(self._alpha.choose(rid, use_regime=use_r))

        if use_r:
            m_lo, m_hi, q_s = self._margins_regime(rid, alpha=alpha, model_uncertainty=unc)
            k_used = float(self._k_scale_by_regime[rid])
        else:
            m_lo, m_hi, q_s = self._margins_global(alpha=alpha, model_uncertainty=unc)
            k_used = float(self._k_scale_global)

        lower = float(yp - m_lo)
        upper = float(yp + m_hi)

        self._pending = {
            "rid": rid,
            "use_r": use_r,
            "yp": yp,
            "unc": unc,
            "alpha": alpha,
            "m_lo": float(m_lo),
            "m_hi": float(m_hi),
            "q_s": float(q_s),
            "k_used": float(k_used),
            "lower": float(lower),
            "upper": float(upper),
        }

        self._last_alpha_sampled = float(alpha)

        return lower, upper
    
    def update(
        self,
        y_true: float = None,
        y: float = None,
        y_obs: float = None,
        **kwargs,
    ):

        yt = y_true if y_true is not None else (y if y is not None else y_obs)
        yt = float(yt)

        p = getattr(self, "_pending", None)

        if p is None:
            return None

        rid = int(p["rid"])
        use_r_pred = bool(p["use_r"])
        yp = float(p["yp"])
        alpha = float(p["alpha"])
        lower = float(p["lower"])
        upper = float(p["upper"])
        q_s = float(p["q_s"])

        e_lo = max(0.0, yp - yt)
        e_hi = max(0.0, yt - yp)
        err = float(e_lo + e_hi)

        self.calib_e_lo_global.append(float(e_lo))
        self.calib_e_hi_global.append(float(e_hi))
        self.calib_e_lo_by_regime[rid].append(float(e_lo))
        self.calib_e_hi_by_regime[rid].append(float(e_hi))

        self.eval_buffer_global.append((float(yp), float(yt)))
        self.eval_buffer_by_regime[rid].append((float(yp), float(yt)))

        covered = 1.0 if (lower <= yt <= upper) else 0.0
        self.cover_hist_global.append(float(covered))
        self.cover_hist_by_regime[rid].append(float(covered))

        self.prediction_errors.append(float(err))
        if self.config.use_spectral:
            s = 0.0
            if len(self.prediction_errors) >= int(self.config.window_size):
                window = np.asarray(list(self.prediction_errors)[-int(self.config.window_size):], float)
                s = float(self._drift.score(window[:-1], window[1:]))
            self.calib_s_global.append(s)
            self.calib_s_by_regime[rid].append(float(s))

            self._maybe_refresh_k(rid)

        use_r_now = bool(self._use_regime(rid))

        if self.config.use_cem:
            obj_g = self._objective_global()
            obj_r = self._objective_regime(rid)
            mu_g, mu_used = self._alpha.step(rid, obj_g, obj_r, update_regime=use_r_now)
            alpha_state = float(mu_used)
        else:
            alpha_state = float(alpha)

        self.alpha_history.append(alpha_state)

        self.k_history.append(float(p["k_used"]))
        self.spectral_q_history.append(float(q_s))
        self.use_regime_history.append(bool(use_r_pred))

        self._pending = None

        return float(lower), float(upper)

    def start_test(self):
        return


