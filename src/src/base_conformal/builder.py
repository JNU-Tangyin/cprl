from .standard_cp import StandardCP
from .aci_cp import ACICP
from .cqr_cp import CQRCP
from .nex_cp import NexCP
from .acp_ours import build_acp_ours
from .agaci_cp import AgACICP
from .dfpi_cp import DFPI
from .enbpi_cp import EnbPICP
from sklearn.ensemble import RandomForestRegressor
from .cptc_cp import CPTCCP


def build_conformal_predictor(args):
    mode = getattr(args, "cp_mode", "acp")

    if mode == "acp":
        return build_acp_ours(args)

    elif mode == "standard":
        return StandardCP(
            alpha=args.alpha,
            window_size=getattr(args, "calib_window_size", 200),
            min_calib_size=args.min_calib_size,
        )

    elif mode == "aci":
        return ACICP(
            alpha=args.alpha,
            window_size=getattr(args, "calib_window_size", 200),
            min_calib_size=args.min_calib_size,
            lr=getattr(args, "cp_lr", 0.01),
        )

    elif mode == "cqr":
        return CQRCP(
            alpha=args.alpha,
            feature_window=getattr(args, "spci_feature_window", 2),
            past_window=getattr(args, "spci_past_window", 1),
            min_calib_size=args.min_calib_size,
            split_ratio=getattr(args, "spci_split_ratio", 0.5),
            qr_l2=getattr(args, "spci_qr_l2", 0.0),
            solver=getattr(args, "spci_solver", "highs"),
            random_state=getattr(args, "seed", 1103),
        )

    elif mode == "nex":
        return NexCP(
            alpha=args.alpha,
            window_size=getattr(args, "calib_window_size", 200),
            min_calib_size=args.min_calib_size,
            gamma=getattr(args, "nex_gamma", 0.99),
        )

    elif mode == "agaci":
        gammas = getattr(args, "agaci_gammas", [0.001, 0.01, 0.05, 0.1])
        return AgACICP(
            alpha=getattr(args, "alpha", 0.1),
            gammas=gammas,
            split_size=getattr(args, "aci_split_size", 0.75),
            t_init=getattr(args, "aci_t_init", 200),
            max_iter=getattr(args, "aci_max_iter", 500),
            warmup_steps=getattr(args, "agaci_warmup_steps", 50),
            seed=getattr(args, "seed", 0),
        )

    elif mode == "dfpi":
        return DFPI(
            alpha=args.alpha,
            min_calib_size=args.min_calib_size,
        )

    elif mode == "enbpi":
        return EnbPICP(
            alpha=args.alpha,
            B=getattr(args, "enbpi_B", 25),
            batch_size_s=getattr(args, "enbpi_batch_size", 30),
            min_calib_size=args.min_calib_size,
            agg=getattr(args, "enbpi_agg", "mean"),
            beta_grid=getattr(args, "enbpi_beta_grid", 101),
            random_state=getattr(args, "seed", 1103),
            fit_func_factory=lambda: RandomForestRegressor(
            n_estimators=getattr(args, "enbpi_n_estimators", 50),
            max_depth=getattr(args, "enbpi_max_depth", 3),
            bootstrap=False,
            n_jobs=-1,
            random_state=getattr(args, "seed", 1103),
        ),
    )

    elif mode == "cptc":
        return CPTCCP(
            alpha=args.alpha,
            gamma=getattr(args, "cptc_gamma", 0.2),
            min_residuals=getattr(args, "cptc_min_residuals", 25),
            max_width=getattr(args, "cptc_max_width", 1e6),
            prob_threshold=getattr(args, "cptc_prob_threshold", 0.0),
            seed=getattr(args, "seed", 0),
        )

    else:
        raise ValueError(f"Unknown cp_mode: {mode}")
