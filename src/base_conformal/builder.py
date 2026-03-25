# src/base_conformal/builder.py

from .aci_cp import ACICP
from .cqr_cp import CQRCP
from .nex_cp import NexCP
from .acp_ours import build_acp_ours
from .agaci_cp import AgACICP
from .dfpi_cp import DFPI
from .enbpi_cp import EnbPICP
from .cptc_cp import CPTCCP
from .hopcpt_cp import HopCPTCP

from sklearn.ensemble import RandomForestRegressor


def build_conformal_predictor(args):
    mode = getattr(args, "cp_mode", "acp")

    # ---- global safe defaults (avoid AttributeError) ----
    alpha = float(getattr(args, "alpha", 0.1))
    min_calib_size = int(getattr(args, "min_calib_size", 30))
    calib_window_size = int(getattr(args, "calib_window_size", 200))
    seed = int(getattr(args, "seed", 1103))

    if mode == "acp":
        return build_acp_ours(args)

    elif mode == "aci":
        return ACICP(
            alpha=alpha,
            T0=int(getattr(args, "aci_T0", calib_window_size)),
            min_calib_size=min_calib_size,
            lr=float(getattr(args, "cp_lr", 0.01)),
            warm_start=int(getattr(args, "aci_warm_start", min_calib_size)),
            fallback_width=float(getattr(args, "aci_fallback_width", 3.0)),
            clip_alpha=bool(int(getattr(args, "aci_clip_alpha", 0))),
            eps=float(getattr(args, "aci_eps", 1e-6)),
            seed=seed,
        )


    elif mode == "cqr":
        return CQRCP(
            alpha=alpha,
            min_calib_size=int(getattr(args, "min_calib_size", 60)),
            split_ratio=float(getattr(args, "cqr_split_ratio", 0.5)),
            qr_l2=float(getattr(args, "cqr_qr_l2", 0.0)),
            solver=str(getattr(args, "cqr_solver", "highs")),
            standardize_x=bool(int(getattr(args, "cqr_standardize_x", 1))),
            sequential_split=bool(int(getattr(args, "cqr_sequential_split", 0))),
            fallback_width=float(getattr(args, "cqr_fallback_width", 3.0)),
            random_state=seed,
        )

    elif mode == "nex":
        return NexCP(
            alpha=alpha,
            window_size=calib_window_size,
            min_calib_size=min_calib_size,
            gamma=float(getattr(args, "nex_gamma", 0.99)),
        )

    elif mode == "agaci":
        gammas = getattr(args, "agaci_gammas", [0.001, 0.01, 0.05, 0.1])
        return AgACICP(
            alpha=alpha,
            gammas=gammas,
            warmup_steps=int(getattr(args, "agaci_warmup_steps", 50)),

            # --- align expert ACI settings with ACI baseline ---
            T0=int(getattr(args, "aci_T0", calib_window_size)),
            min_calib_size=min_calib_size,
            warm_start=int(getattr(args, "aci_warm_start", min_calib_size)),
            fallback_width=float(getattr(args, "aci_fallback_width", 3.0)),
            clip_alpha=bool(int(getattr(args, "aci_clip_alpha", 0))),
            eps=float(getattr(args, "aci_eps", 1e-6)),

            seed=seed,
        )

    elif mode == "dfpi":
        return DFPI(
            alpha=alpha,
            min_calib_size=min_calib_size,
        )

    elif mode == "enbpi":
        # 关键：这里把括号闭合写清楚，避免语法错误
        enbpi_n_estimators = int(getattr(args, "enbpi_n_estimators", 50))
        enbpi_max_depth = int(getattr(args, "enbpi_max_depth", 3))

        def _fit_func_factory():
            return RandomForestRegressor(
                n_estimators=enbpi_n_estimators,
                max_depth=enbpi_max_depth,
                bootstrap=False,
                n_jobs=-1,
                random_state=seed,
            )

        return EnbPICP(
            alpha=alpha,
            B=int(getattr(args, "enbpi_B", 25)),
            batch_size_s=int(getattr(args, "enbpi_batch_size", 30)),
            min_calib_size=min_calib_size,
            agg=str(getattr(args, "enbpi_agg", "mean")),
            beta_grid=int(getattr(args, "enbpi_beta_grid", 101)),
            random_state=seed,
            fit_func_factory=_fit_func_factory,
        )

    elif mode == "cptc":
        return CPTCCP(
            alpha=alpha,
            gamma=float(getattr(args, "cptc_gamma", 0.2)),
            warm_start=int(getattr(args, "cptc_warm_start", 100)),
            min_residuals=int(getattr(args, "cptc_min_residuals", 25)),
            max_width=float(getattr(args, "cptc_max_width", 3.0)),
            prob_threshold=float(getattr(args, "cptc_prob_threshold", 0.3)),
            agg=str(getattr(args, "cptc_agg", "mass")),  # "mass" | "union_all"
            use_argmax_state=bool(int(getattr(args, "cptc_use_argmax_state", 1))),
            seed=int(getattr(args, "seed", 0)),
        )

    elif mode == "hopcpt":
        return HopCPTCP(
            alpha=alpha,
            min_calib_size=min_calib_size,
            emb_dim=int(getattr(args, "hopcpt_emb_dim", 32)),
            hidden_dim=int(getattr(args, "hopcpt_hidden_dim", 64)),
            train_epochs=int(getattr(args, "hopcpt_train_epochs", 200)),
            lr=float(getattr(args, "hopcpt_lr", 1e-3)),
            beta=float(getattr(args, "hopcpt_beta", 1.0)),
            online_update=bool(int(getattr(args, "hopcpt_online_update", 1))),
            seed=seed,
        )
        
    else:
        raise ValueError(f"Unknown cp_mode: {mode}")
