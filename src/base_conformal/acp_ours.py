# src/base_conformal/acp_ours.py

from src.conformal_prediction import ConformalPredictionConfig, AdaptiveConformalPredictor

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

        coverage_window=int(getattr(args, "coverage_window", 50)),

        lambda_spectral=float(getattr(args, "lambda_spectral", 0.5)),

        # ACI + spectral modulation
        aci_gamma_base=float(getattr(args, "aci_gamma_base", 0.05)),
        aci_spectral_beta=float(getattr(args, "aci_spectral_beta", 1.0)),
        spectral_score_cap=float(getattr(args, "spectral_score_cap", 2.0)),

        # Wasserstein reweighting
        wass_reweight=bool(getattr(args, "wass_reweight", True)),
        wass_temperature=float(getattr(args, "wass_temperature", 0.1)),

        # CQR-inspired single-score conformal with online QR
        use_cqr_score=bool(getattr(args, "use_cqr_score", True)),
        cqr_refit_every=int(getattr(args, "cqr_refit_every", 50)),
        cqr_l2=float(getattr(args, "cqr_l2", 0.1)),
        cqr_split_ratio=float(getattr(args, "cqr_split_ratio", 0.6)),
        use_legacy_buffer_cqr=bool(getattr(args, "use_legacy_buffer_cqr", False)),
        cqr_r_clip=float(getattr(args, "cqr_r_clip", 8.0)),
        cqr_x_clip_quantile=float(getattr(args, "cqr_x_clip_quantile", 0.01)),
        cqr_x_std_clip=float(getattr(args, "cqr_x_std_clip", 6.0)),
        unc_floor_min=float(getattr(args, "unc_floor_min", 1e-3)),
        unc_floor_window=int(getattr(args, "unc_floor_window", 128)),
        unc_floor_quantile=float(getattr(args, "unc_floor_quantile", 0.1)),
        unc_floor_scale=float(getattr(args, "unc_floor_scale", 0.5)),

        # residual-space regime + warm-start
        regime_on_residuals=bool(getattr(args, "regime_on_residuals", True)),
        warmstart_blend=float(getattr(args, "warmstart_blend", 0.3)),
        regime_method=str(getattr(args, "regime_method", "feature")).lower(),

        # ODE-based regime discovery
        ode_window_size=int(getattr(args, "ode_window_size", 48)),
        ode_smooth_window=int(getattr(args, "ode_smooth_window", 3)),
        ode_use_residuals=bool(getattr(args, "ode_use_residuals", True)),
        ode_ic=str(getattr(args, "ode_ic", "bic")).lower(),
        ode_cond_max=float(getattr(args, "ode_cond_max", 1e6)),
        ode_stable_only=bool(getattr(args, "ode_stable_only", True)),
        ode_min_samples=int(getattr(args, "ode_min_samples", 16)),
        ode_cluster_eps_order0=float(getattr(args, "ode_cluster_eps_order0", 0.55)),
        ode_cluster_eps_order1=float(getattr(args, "ode_cluster_eps_order1", 0.9)),
        ode_cluster_eps_order2=float(getattr(args, "ode_cluster_eps_order2", 1.1)),
        ode_cluster_min_samples=int(getattr(args, "ode_cluster_min_samples", 6)),
        ode_refit_every=int(getattr(args, "ode_refit_every", 25)),
        ode_bootstrap_size=int(getattr(args, "ode_bootstrap_size", 60)),
        ode_assignment_threshold=float(getattr(args, "ode_assignment_threshold", 2.0)),

        k_update_every=int(getattr(args, "k_update_every", 20)),
        k_min=float(getattr(args, "k_min", 1e-3)),
        k_max=float(getattr(args, "k_max", 100.0)),
        k_fallback=float(getattr(args, "k_fallback", 1.0)),

        alpha_min=float(getattr(args, "alpha_min", getattr(args, "cem_alpha_min", 0.01))),
        alpha_max=float(getattr(args, "alpha_max", getattr(args, "cem_alpha_max", 0.3))),
    )

    mode = str(getattr(args, "ablation_mode", "M0")).upper()

    cfg.use_spectral = True
    cfg.use_regime   = True
    cfg.use_adaptive_alpha = bool(getattr(args, "adaptive_alpha", True))

    if mode == "M0":
        pass
    elif mode == "M1":  # no spectral
        cfg.use_spectral = False
        cfg.wass_reweight = False
    elif mode == "M2":  # no regime
        cfg.use_regime = False
        cfg.regime_on_residuals = False
    elif mode == "M3":  # no adaptive alpha control (fixed alpha)
        cfg.use_adaptive_alpha = False
    elif mode == "M4":  # no Wasserstein reweighting
        cfg.wass_reweight = False
    elif mode == "M5":  # no score normalization (raw residuals)
        pass  # controlled in update() by unc; needs flag if desired
    elif mode == "M6":  # no CQR score (legacy separate quantiles)
        cfg.use_cqr_score = False
    else:
        raise ValueError(f"Unknown ablation_mode: {mode}")

    return AdaptiveConformalPredictor(config=cfg)
