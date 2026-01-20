from src.conformal_prediction import ConformalPredictionConfig, AdaptiveConformalPredictor
from src.cem import CEMConfig  # 按你实际路径改（你 cem.py 在哪就从哪 import）


def build_acp_ours(args) -> AdaptiveConformalPredictor:
    # 允许命令行覆写 CEM 超参数（如果你还没加 args，就用默认）
    cem_cfg = CEMConfig(
        alpha_min=float(getattr(args, "cem_alpha_min", 0.01)),
        alpha_max=float(getattr(args, "cem_alpha_max", 0.5)),
        pop_size=int(getattr(args, "cem_pop_size", 20)),
        elite_frac=float(getattr(args, "cem_elite_frac", 0.2)),
        init_std=float(getattr(args, "cem_init_std", 0.05)),
        min_std=float(getattr(args, "cem_min_std", 0.01)),
        smooth=float(getattr(args, "cem_smooth", 0.7)),
        n_iters=int(getattr(args, "cem_n_iters", 1)),
    )

    config = ConformalPredictionConfig(
        initial_alpha=float(getattr(args, "alpha", 0.1)),
        window_size=int(getattr(args, "spectral_window", 64)),
        n_latent_states=int(getattr(args, "n_latent_states", 3)),

        cem_config=cem_cfg,

        calib_window_size=int(getattr(args, "calib_window_size", 200)),
        lambda_spectral=float(getattr(args, "lambda_spectral", 0.5)),
        min_calib_size=int(getattr(args, "min_calib_size", 30)),

        # 你 strict CEM 新加的参数：建议也透出
        eval_window_size=int(getattr(args, "eval_window_size", 100)),
        coverage_window=int(getattr(args, "coverage_window", 50)),
        lambda_cov=float(getattr(args, "lambda_cov", 5.0)),
    )

    return AdaptiveConformalPredictor(config=config)
