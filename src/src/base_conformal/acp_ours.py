# src/base_conformal/acp_ours.py

from src.conformal_prediction_ import ConformalPredictionConfig, AdaptiveConformalPredictor

def build_acp_ours(args) -> AdaptiveConformalPredictor:
    cfg = ConformalPredictionConfig(
        initial_alpha=float(getattr(args, "alpha", 0.1)),
        target_coverage=float(getattr(args, "target_coverage", 1.0 - float(getattr(args, "alpha", 0.1)))),
        window_size=int(getattr(args, "spectral_window", getattr(args, "window_size", 64))),

        max_regimes=int(getattr(args, "max_regimes", 8)),
        new_regime_threshold=float(getattr(args, "new_regime_threshold", 2.2)),
        new_regime_patience=int(getattr(args, "new_1regime_patience", 3)),
        sticky_bonus=float(getattr(args, "sticky_bonus", 0.5)),
        min_state_duration=int(getattr(args, "min_state_duration", 5)),
        ewma_beta=float(getattr(args, "ewma_beta", 0.94)),
        jump_q=float(getattr(args, "jump_q", 0.95)),
        feature_ema=float(getattr(args, "feature_ema", 0.05)),

        calib_window_size=int(getattr(args, "calib_window_size", 200)),
        min_calib_size=int(getattr(args, "min_calib_size", 30)),
        min_regime_calib_size=int(getattr(args, "min_regime_calib_size", 50)),
        min_regime_eval_size=int(getattr(args, "min_regime_eval_size", 30)),
        min_regime_cov_size=int(getattr(args, "min_regime_cov_size", 30)),

        eval_window_size=int(getattr(args, "eval_window_size", 100)),
        coverage_window=int(getattr(args, "coverage_window", 50)),
        lambda_cov=float(getattr(args, "lambda_cov", 5.0)),

        lambda_spectral=float(getattr(args, "lambda_spectral", 0.5)),

        k_update_every=int(getattr(args, "k_update_every", 20)),
        k_min=float(getattr(args, "k_min", 1e-3)),
        k_max=float(getattr(args, "k_max", 100.0)),
        k_fallback=float(getattr(args, "k_fallback", 1.0)),

        alpha_min=float(getattr(args, "cem_alpha_min", getattr(args, "alpha_min", 0.01))),
        alpha_max=float(getattr(args, "cem_alpha_max", getattr(args, "alpha_max", 0.3))),
        cem_pop=int(getattr(args, "cem_pop", getattr(args, "cem_pop_size", 16))),
        cem_elite_frac=float(getattr(args, "cem_elite_frac", 0.25)),
        cem_noise=float(getattr(args, "cem_noise", getattr(args, "cem_init_std", 0.02))),
        cem_lr=float(getattr(args, "cem_lr", getattr(args, "cem_smooth", 0.3))),
        cem_n_iters=int(getattr(args, "cem_n_iters", 3)),
    )

    mode = str(getattr(args, "ablation_mode", "M0")).upper()

    cfg.use_spectral = True
    cfg.use_regime   = True
    cfg.use_cem      = True
    cfg.width_weight = 1.0

    if mode == "M0":
        pass
    elif mode == "M1":
        cfg.use_spectral = False
    elif mode == "M2":
        cfg.use_regime = False
    elif mode == "M3":
        cfg.use_cem = False
    elif mode == "M4":
        cfg.width_weight = 0.0
    else:
        raise ValueError(f"Unknown ablation_mode: {mode}")

    return AdaptiveConformalPredictor(config=cfg)