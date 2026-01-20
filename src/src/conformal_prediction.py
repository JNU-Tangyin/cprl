# conformal_prediction.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Deque, Dict
from collections import deque

import numpy as np

from .latent_state_generator import LatentStateGenerator
from .spectral_analysis import SpectralAnalyzer
from .cem import CEMConfig, RegimeCEMAlphaController


@dataclass
class ConformalPredictionConfig:
    initial_alpha: float = 0.1
    window_size: int = 64
    n_latent_states: int = 3

    cem_config: Optional[CEMConfig] = None

    # sliding windows
    calib_window_size: int = 200
    min_calib_size: int = 30
    min_regime_calib_size: int = 50

    # spectral term weight (调参)
    lambda_spectral: float = 0.5

    # 每个 regime 至少多少样本才用 regime buffer；否则 fallback 到 global
    min_regime_eval_size: int = 30
    min_regime_cov_size: int = 30

    # CEM replay buffer
    eval_window_size: int = 100

    # coverage penalty window
    coverage_window: int = 50
    lambda_cov: float = 5.0

    # refresh k in steps
    k_update_every: int = 20
    k_min: float = 1e-3
    k_max: float = 100.0 # (调参)
    k_fallback: float = 1.0


class AdaptiveConformalPredictor:
    """
    HSMM state + spectral Wasserstein + CEM(alpha).
    """

    def __init__(self, config: Optional[ConformalPredictionConfig] = None) -> None:
        self.config: ConformalPredictionConfig = config if config is not None else ConformalPredictionConfig()

        self.alpha: float = float(self.config.initial_alpha)

        self.latent_model: LatentStateGenerator = LatentStateGenerator(n_states=self.config.n_latent_states)
        self.spectral_analyzer: SpectralAnalyzer = SpectralAnalyzer(window_size=self.config.window_size)

        cem_cfg = self.config.cem_config or CEMConfig()
        self.alpha_controller = RegimeCEMAlphaController(
            n_regimes=self.config.n_latent_states,
            alpha_init=self.alpha,
            cfg=cem_cfg,
            use_global_fallback=True,
        )

        self._last_alpha_used: Optional[float] = None
        self._last_regime_used: Optional[int] = None

        self.current_state: Optional[int] = None
        self.state_history: List[int] = []

        self.prediction_errors: List[float] = []
        self.spectral_errors: List[float] = []

        # global calib buffers
        self.calib_s_global: Deque[float] = deque(maxlen=int(self.config.calib_window_size))  # spectral error
        self.calib_e_lo_global: Deque[float] = deque(maxlen=int(self.config.calib_window_size))  # max(0, y_pred - y_true)
        self.calib_e_hi_global: Deque[float] = deque(maxlen=int(self.config.calib_window_size))  # max(0, y_true - y_pred)

        # regime-wise calib buffers
        self.calib_s_by_regime: Dict[int, Deque[float]] = {
            r: deque(maxlen=int(self.config.calib_window_size)) for r in range(int(self.config.n_latent_states))
        }
        self.calib_e_lo_by_regime: Dict[int, Deque[float]] = {
            r: deque(maxlen=int(self.config.calib_window_size)) for r in range(int(self.config.n_latent_states))
        }
        self.calib_e_hi_by_regime: Dict[int, Deque[float]] = {
            r: deque(maxlen=int(self.config.calib_window_size)) for r in range(int(self.config.n_latent_states))
        }

        # margin k scale: global + per regime
        self._k_scale_global: float = float(self.config.k_fallback)
        self._k_scale_by_regime: Dict[int, float] = {r: float(self.config.k_fallback) for r in range(int(self.config.n_latent_states))}

        self._k_t_global: int = 0
        self._k_t_by_regime: Dict[int, int] = {r: 0 for r in range(int(self.config.n_latent_states))}

        # --- global CEM buffers ---
        self.cover_hist_global: Deque[float] = deque(maxlen=int(self.config.coverage_window))
        self.eval_buffer_global: Deque[Tuple[float, float]] = deque(maxlen=int(self.config.eval_window_size))

        # --- regime-wise CEM buffers ---
        self.cover_hist_by_regime: Dict[int, Deque[float]] = {
            r: deque(maxlen=int(self.config.coverage_window)) for r in range(int(self.config.n_latent_states))
        }
        self.eval_buffer_by_regime: Dict[int, Deque[Tuple[float, float]]] = {
            r: deque(maxlen=int(self.config.eval_window_size)) for r in range(int(self.config.n_latent_states))
        }

    def _clip_regime(self, rid: Optional[int]) -> Optional[int]:
        if rid is None:
            return None
        rid = int(rid)
        return max(0, min(int(self.config.n_latent_states) - 1, rid))

    def _get_eval_buffer(self, rid: Optional[int]):
        rid = self._clip_regime(rid)
        if rid is None:
            return self.eval_buffer_global
        buf = self.eval_buffer_by_regime[rid]
        if len(buf) >= int(self.config.min_regime_eval_size):
            return buf
        return self.eval_buffer_global

    def _get_calib_buffers_asym(self, rid: Optional[int]):
        rid = self._clip_regime(rid)

        # ---- e_lo: independent fallback ----
        if rid is not None and len(self.calib_e_lo_by_regime[rid]) >= int(self.config.min_regime_calib_size):
            e_lo_buf = self.calib_e_lo_by_regime[rid]
        else:
            e_lo_buf = self.calib_e_lo_global

        # ---- e_hi: independent fallback ----
        if rid is not None and len(self.calib_e_hi_by_regime[rid]) >= int(self.config.min_regime_calib_size):
            e_hi_buf = self.calib_e_hi_by_regime[rid]
        else:
            e_hi_buf = self.calib_e_hi_global

        # ---- spectral + k: keep original fallback logic ----
        if rid is not None and len(self.calib_s_by_regime[rid]) >= int(self.config.min_regime_calib_size):
            s_buf = self.calib_s_by_regime[rid]
            k_scale = self._k_scale_by_regime[rid]
        else:
            s_buf = self.calib_s_global
            k_scale = self._k_scale_global

        return e_lo_buf, e_hi_buf, s_buf, k_scale

    def _update_calib_e_asym(self, rid: Optional[int], y_true: float, y_pred: float) -> Tuple[float, float]:
        rid = self._clip_regime(rid)

        e_lo = max(0.0, float(y_pred) - float(y_true))
        e_hi = max(0.0, float(y_true) - float(y_pred))

        self.calib_e_lo_global.append(float(e_lo))
        self.calib_e_hi_global.append(float(e_hi))

        if rid is not None:
            self.calib_e_lo_by_regime[rid].append(float(e_lo))
            self.calib_e_hi_by_regime[rid].append(float(e_hi))

        return float(e_lo), float(e_hi)


    def _update_calib_s(self, rid: Optional[int], s: float) -> None:
        rid = self._clip_regime(rid)
        self.calib_s_global.append(float(s))
        if rid is not None:
            self.calib_s_by_regime[rid].append(float(s))


    def _get_cov_hist(self, rid: Optional[int]):
        rid = self._clip_regime(rid)
        if rid is None:
            return self.cover_hist_global
        buf = self.cover_hist_by_regime[rid]
        if len(buf) >= int(self.config.min_regime_cov_size):
            return buf
        return self.cover_hist_global

    def initialize(self, initial_data: np.ndarray) -> None:
        try:
            self.latent_model.initialize_params(initial_data)
        except TypeError:
            self.latent_model.initialize_params()
        self._update_latent_state(initial_data)

    def _update_latent_state(self, data: np.ndarray) -> None:
        if self.current_state is None:
            self.current_state = np.random.randint(0, self.config.n_latent_states)
        else:
            transition_probs = self.latent_model.get_next_state_probs(self.current_state)
            transition_probs = np.asarray(transition_probs, dtype=float)
            s = float(transition_probs.sum())
            if s <= 1e-12:
                transition_probs = np.ones_like(transition_probs) / len(transition_probs)
            else:
                transition_probs = transition_probs / s
            self.current_state = int(np.random.choice(len(transition_probs), p=transition_probs))

        self.state_history.append(int(self.current_state))

    def _compute_spectral_error(self, y_true_window: np.ndarray, y_pred_window: np.ndarray) -> float:
        return float(self.spectral_analyzer.spectral_wasserstein_distance(y_true_window, y_pred_window))

    def _refresh_k_scale_if_needed(self, rid: Optional[int]) -> None:
        """
        Robustly align spectral scale to error scale using median ratio.
        """

        def _compute_k_from_asym(e_lo_buf, e_hi_buf, s_buf) -> float:
            if len(e_lo_buf) < 10 or len(e_hi_buf) < 10 or len(s_buf) < 10:
                return float(self.config.k_fallback)

            e_lo = np.asarray(list(e_lo_buf), dtype=float)
            e_hi = np.asarray(list(e_hi_buf), dtype=float)
            s = np.asarray(list(s_buf), dtype=float)

            e_lo = e_lo[np.isfinite(e_lo)]
            e_hi = e_hi[np.isfinite(e_hi)]
            s = s[np.isfinite(s)]

            if len(e_lo) < 10 or len(e_hi) < 10 or len(s) < 10:
                return float(self.config.k_fallback)

            n = min(len(e_lo), len(e_hi))
            if n <= 1:
                return float(self.config.k_fallback)

            e_scale = e_lo[-n:] + e_hi[-n:]

            med_e = float(np.median(e_scale))
            med_s = float(np.median(s))

            if (not np.isfinite(med_e)) or (not np.isfinite(med_s)) or (med_s <= 1e-12):
                k = float(self.config.k_fallback)
            else:
                k = med_e / med_s

            k = float(np.clip(k, float(self.config.k_min), float(self.config.k_max)))
            if not np.isfinite(k):
                k = float(self.config.k_fallback)
            return float(k)

        self._k_t_global += 1
        if self._k_t_global % int(self.config.k_update_every) == 0:
            self._k_scale_global = _compute_k_from_asym(
                self.calib_e_lo_global,
                self.calib_e_hi_global,
                self.calib_s_global,
            )

        rid = self._clip_regime(rid)
        if rid is None:
            return

        self._k_t_by_regime[rid] += 1
        if self._k_t_by_regime[rid] % int(self.config.k_update_every) == 0:
            self._k_scale_by_regime[rid] = _compute_k_from_asym(
                self.calib_e_lo_by_regime[rid],
                self.calib_e_hi_by_regime[rid],
                self.calib_s_by_regime[rid],
            )

    def _compute_margins(self, rid: Optional[int], alpha: float, model_uncertainty: float) -> Tuple[float, float]:
        """
        Asymmetric:
            margin_lo = q_lo(1-a) + lambda * k * q_s(1-a)
            margin_hi = q_hi(1-a) + lambda * k * q_s(1-a)
        where:
            q_lo from e_lo = max(0, y_pred - y_true)
            q_hi from e_hi = max(0, y_true - y_pred)

        Warm-start:
            return (model_uncertainty/(1-a), model_uncertainty/(1-a))
        """
        a = float(alpha)

        e_lo_buf, e_hi_buf, s_buf, k_scale = self._get_calib_buffers_asym(rid)

        if (len(e_lo_buf) < int(self.config.min_calib_size)) or (len(e_hi_buf) < int(self.config.min_calib_size)):
            scale = 1.0 / max(1e-6, 1.0 - a)
            m = float(model_uncertainty) * float(scale)
            return float(m), float(m)

        e_lo = np.asarray(list(e_lo_buf), dtype=float)
        e_hi = np.asarray(list(e_hi_buf), dtype=float)
        e_lo = e_lo[np.isfinite(e_lo)]
        e_hi = e_hi[np.isfinite(e_hi)]

        if len(e_lo) == 0 or len(e_hi) == 0:
            m = float(model_uncertainty)
            return float(m), float(m)

        q_lo = float(np.quantile(e_lo, 1.0 - a))
        q_hi = float(np.quantile(e_hi, 1.0 - a))

        if len(s_buf) >= int(self.config.min_calib_size):
            s = np.asarray(list(s_buf), dtype=float)
            s = s[np.isfinite(s)]
            q_s = float(np.quantile(s, 1.0 - a)) if len(s) > 0 else 0.0
        else:
            q_s = 0.0

        extra = float(self.config.lambda_spectral) * float(k_scale) * float(q_s)

        m_lo = q_lo + extra
        m_hi = q_hi + extra

        # safety
        if not np.isfinite(m_lo):
            m_lo = q_lo
        if not np.isfinite(m_hi):
            m_hi = q_hi

        return float(max(0.0, m_lo)), float(max(0.0, m_hi))

    def update(self, y_true: float, y_pred: float, prediction_interval: Tuple[float, float]) -> None:
        if self._last_alpha_used is None:
            return

        alpha_used = float(self._last_alpha_used)
        regime_used = self._last_regime_used

        # consume
        self._last_alpha_used = None
        self._last_regime_used = None

        rid = self._clip_regime(regime_used)

        self.eval_buffer_global.append((float(y_pred), float(y_true)))

        if rid is not None:
            self.eval_buffer_by_regime[rid].append((float(y_pred), float(y_true)))

        e_lo, e_hi = self._update_calib_e_asym(rid, y_true, y_pred)
        err = float(e_lo + e_hi)
        self.prediction_errors.append(err)

        spectral_error: Optional[float] = None

        if len(self.prediction_errors) >= int(self.config.window_size):
            window_errors = np.asarray(self.prediction_errors[-int(self.config.window_size):], dtype=float)

            self._update_latent_state(window_errors)

            spectral_error = self._compute_spectral_error(window_errors[:-1], window_errors[1:])
            self.spectral_errors.append(float(spectral_error))

            self._update_calib_s(rid, float(spectral_error))

        self._refresh_k_scale_if_needed(rid)


        lower, upper = float(prediction_interval[0]), float(prediction_interval[1])
        width = float(upper - lower)
        covered = 1.0 if (lower <= float(y_true) <= upper) else 0.0

        if float(y_true) < lower:
            miss_dist = float(lower - float(y_true))
        elif float(y_true) > upper:
            miss_dist = float(float(y_true) - upper)
        else:
            miss_dist = 0.0

        winkler = width + (2.0 / max(alpha_used, 1e-6)) * miss_dist

        self.cover_hist_global.append(float(covered))

        if rid is not None:
            self.cover_hist_by_regime[rid].append(float(covered))

        cov_hist = self._get_cov_hist(rid)
        c_hat = float(np.mean(cov_hist)) if len(cov_hist) > 0 else 0.0

        target_cov = 1.0 - float(self.config.initial_alpha)
        gap = max(0.0, target_cov - c_hat)
        pareto_penalty = float(self.config.lambda_cov) * (gap ** 2)

        reward = -(winkler + pareto_penalty)

        def eval_fn(a: float) -> float:
            cfg = self.alpha_controller.cfg
            a = float(np.clip(a, float(cfg.alpha_min), float(cfg.alpha_max)))


            e_lo_buf, e_hi_buf, _, _ = self._get_calib_buffers_asym(rid)
            if len(e_lo_buf) >= 2 and len(e_hi_buf) >= 2:
                e_lo_arr = np.asarray(list(e_lo_buf), dtype=float)
                e_hi_arr = np.asarray(list(e_hi_buf), dtype=float)
                e_lo_arr = e_lo_arr[np.isfinite(e_lo_arr)]
                e_hi_arr = e_hi_arr[np.isfinite(e_hi_arr)]

                n = min(len(e_lo_arr), len(e_hi_arr))
                if n >= 2:
                    tail = min(50, n)
                    e_sum = e_lo_arr[-tail:] + e_hi_arr[-tail:]
                    mu_unc = float(e_sum.std() + 1e-6)
                else:
                    mu_unc = 1.0
            else:
                mu_unc = 1.0

            m_lo, m_hi = self._compute_margins(rid=rid, alpha=a, model_uncertainty=mu_unc)
            width_const = float(m_lo) + float(m_hi)

            buf = self._get_eval_buffer(rid)
            if len(buf) == 0:
                return -1e9

            winklers = []
            covered_list = []

            for yp, yt in buf:
                lower_c = float(yp) - float(m_lo)
                upper_c = float(yp) + float(m_hi)

                cov = 1.0 if (lower_c <= float(yt) <= upper_c) else 0.0
                covered_list.append(cov)

                if float(yt) < lower_c:
                    miss = float(lower_c - float(yt))
                elif float(yt) > upper_c:
                    miss = float(float(yt) - upper_c)
                else:
                    miss = 0.0

                wink = width_const + (2.0 / max(a, 1e-6)) * miss
                winklers.append(wink)

            winkler_mean = float(np.mean(winklers))
            c_hat_local = float(np.mean(covered_list))

            target_cov_local = 1.0 - float(self.config.initial_alpha)
            gap_local = max(0.0, target_cov_local - c_hat_local)
            pareto_penalty_local = float(self.config.lambda_cov) * (gap_local ** 2)

            return -(winkler_mean + pareto_penalty_local)

        _ = self.alpha_controller.step(regime_used, eval_fn)

        self.alpha = float(alpha_used)
        _ = reward

    def predict(self, base_prediction: float, model_uncertainty: float) -> Tuple[float, float]:
        if self.current_state is None:
            self.current_state = int(np.random.randint(0, self.config.n_latent_states))

        regime_id = int(self.current_state)

        alpha_used = float(self.alpha_controller.choose_alpha(regime_id))

        self._last_alpha_used = float(alpha_used)
        self._last_regime_used = int(regime_id)
        self.alpha = float(alpha_used)

        m_lo, m_hi = self._compute_margins(rid=regime_id, alpha=self.alpha, model_uncertainty=float(model_uncertainty))

        lower = float(base_prediction) - float(m_lo)
        upper = float(base_prediction) + float(m_hi)
        return lower, upper

    def reset(self) -> None:
        self.alpha = float(self.config.initial_alpha)
        self.current_state = None
        self.state_history.clear()

        self.prediction_errors.clear()
        self.spectral_errors.clear()

        self.calib_s_global.clear()
        for r in range(int(self.config.n_latent_states)):
            self.calib_s_by_regime[r].clear()

        self._k_scale_global = float(self.config.k_fallback)
        self._k_t_global = 0
        for r in range(int(self.config.n_latent_states)):
            self._k_scale_by_regime[r] = float(self.config.k_fallback)
            self._k_t_by_regime[r] = 0

        self.eval_buffer_global.clear()
        self.cover_hist_global.clear()
        for r in range(int(self.config.n_latent_states)):
            self.eval_buffer_by_regime[r].clear()
            self.cover_hist_by_regime[r].clear()

        self.calib_e_lo_global.clear()
        self.calib_e_hi_global.clear()
        for r in range(int(self.config.n_latent_states)):
            self.calib_e_lo_by_regime[r].clear()
            self.calib_e_hi_by_regime[r].clear()

        self._last_alpha_used = None
        self._last_regime_used = None
