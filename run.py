import argparse
import random
import numpy as np
import os
import sys
from pathlib import Path

import torch
import torch.backends

project_root = str(Path(__file__).parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from exp.exp_conformal import ExpConformal
from exp.exp_forecast_cache import ExpForecastCache

try:
    from src.utils import print_args
except Exception:
    print_args = None


def parse_args():
    parser = argparse.ArgumentParser(description="Forecast Cache + Conformal Runner")

    # ======================================================
    # task selection
    # ======================================================
    parser.add_argument(
        "--task",
        type=str,
        default="conformal",
        choices=["conformal", "forecast_cache"],
        help="Run conformal prediction or export forecast cache",
    )

    # ======================================================
    # protocol
    # ======================================================
    parser.add_argument(
        "--run_mode",
        type=str,
        default="online",
        choices=["online", "eval"],
        help="online: update CP on test stream; eval: do NOT update CP",
    )
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    # ======================================================
    # naming / output
    # ======================================================
    parser.add_argument("--exp_name", type=str, default="cachecp")
    parser.add_argument("--des", type=str, default="debug")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--comment", type=str, default="")

    # ======================================================
    # data (used for cache generation)
    # ======================================================
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--target_col", type=str, default=None)

    parser.add_argument("--lags", type=int, default=96)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--calib_ratio", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)

    # ======================================================
    # base forecasting model
    # ======================================================
    parser.add_argument("--base_model", type=str, default="linear")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_epochs", type=int, default=20)

    parser.add_argument(
        "--cache_save_path",
        type=str,
        default=None,
        help="Optional manual path to save forecast cache",
    )

    # ======================================================
    # cache input (for conformal)
    # ======================================================
    parser.add_argument("--cache_path", type=str, default=None)

    parser.add_argument(
        "--x_lag",
        type=int,
        default=24,
        help="Lag length used to build CP features from past y_true history",
    )

    # ======================================================
    # conformal
    # ======================================================
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--cp_mode",
        type=str,
        default="acp",
        choices=[
            "acp",
            "aci",
            "agaci",
            "nex",
            "cqr",
            "dfpi",
            "enbpi",
            "cptc",
            "hopcpt",
        ],
    )

    parser.add_argument("--spectral_window", type=int, default=64)
    parser.add_argument("--n_latent_states", type=int, default=3)
    parser.add_argument("--calib_window_size", type=int, default=200)
    parser.add_argument("--lambda_spectral", type=float, default=0.5)
    parser.add_argument("--min_calib_size", type=int, default=30)

    # ======================================================
    # diagnostics
    # ======================================================
    parser.add_argument("--unc_window", type=int, default=256)
    parser.add_argument("--calib_window", type=int, default=200)
    parser.add_argument("--calib_print_every", type=int, default=200)
    parser.add_argument("--test_window", type=int, default=100)
    parser.add_argument("--test_print_every", type=int, default=100)
    parser.add_argument("--dynamics_stride", type=int, default=1)
    parser.add_argument("--worst_window", type=int, default=100)

    # ======================================================
    # device
    # ======================================================
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--gpu_type",
        type=str,
        default="cuda",
        choices=["cuda", "mps", "cpu"],
    )

    # ======================================================
    # CPTC
    # ======================================================
    parser.add_argument("--cptc_gamma", type=float, default=0.2)
    parser.add_argument("--cptc_warm_start", type=int, default=100)
    parser.add_argument("--cptc_max_width", type=float, default=3.0)
    parser.add_argument("--cptc_min_residuals", type=int, default=25)
    parser.add_argument("--cptc_warm_threshold", type=float, default=0.3)
    parser.add_argument("--cptc_aggregate_sort_asc", type=int, default=1)

    # ======================================================
    # ACI
    # ======================================================

    parser.add_argument("--aci_gamma", type=float, default=5e-4)

    # ======================================================
    # HopCPT
    # ======================================================
    parser.add_argument("--hopcpt_emb_dim", type=int, default=32)
    parser.add_argument("--hopcpt_hidden_dim", type=int, default=64)
    parser.add_argument("--hopcpt_train_epochs", type=int, default=200)
    parser.add_argument("--hopcpt_lr", type=float, default=1e-3)
    parser.add_argument("--hopcpt_beta", type=float, default=1.0)
    parser.add_argument("--hopcpt_online_update", type=int, default=1)

    # ======================================================
    # ablation
    # ======================================================
    parser.add_argument("--ablation", type=int, default=0)
    parser.add_argument("--ablation_mode", type=str, default="M0")

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(args):
    if args.gpu_type == "cpu":
        args.device = torch.device("cpu")
        print("Using CPU")
        return

    if args.gpu_type == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = torch.device("mps")
            print("Using MPS")
        else:
            args.device = torch.device("cpu")
            print("MPS not available, fallback to CPU")
        return

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device(f"cuda:{args.gpu}")
        print("Using CUDA:", args.device)
    else:
        args.device = torch.device("cpu")
        print("CUDA not available or --use_gpu not set, fallback to CPU")


def infer_dataset_name_from_cache(cache_path: str):
    parent = Path(cache_path).resolve().parent.name
    parts = parent.split("_")

    if "cache" in parts:
        idx = parts.index("cache")
        if idx - 1 >= 0:
            return parts[idx - 1]

    return parent

def infer_base_model_from_cache(cache_path: str):
    parent = Path(cache_path).resolve().parent.name
    parts = parent.split("_")

    if "cache" in parts:
        idx = parts.index("cache")
        if idx + 1 < len(parts):
            return parts[idx + 1]

    return "cache"


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    set_seed(int(args.seed))
    select_device(args)

    print("Args in experiment:")
    if print_args is not None:
        print_args(args)
    else:
        for k, v in sorted(vars(args).items()):
            print(f"  {k}: {v}")

    # ======================================================
    # TASK 1: forecast cache
    # ======================================================
    if args.task == "forecast_cache":
        if args.data_path is None:
            raise ValueError("forecast_cache task requires --data_path")

        exp = ExpForecastCache(args)

        print(">>>> exporting forecast cache >>>>")
        save_path = exp.run()
        print(">>>> cache exported:", save_path)

        sys.exit(0)

    # ======================================================
    # TASK 2: conformal prediction
    # ======================================================
    if args.cache_path is None:
        raise ValueError("conformal task requires --cache_path")

    if not os.path.exists(args.cache_path):
        raise FileNotFoundError(f"Cache file not found: {args.cache_path}")

    if int(getattr(args, "ablation", 0)) == 1:
        from exp.exp_ablation import run_ablation
        run_ablation(args)
        sys.exit(0)

    dataset_name = infer_dataset_name_from_cache(args.cache_path)
    base_model_name = infer_base_model_from_cache(args.cache_path)

    args.dataset_name = dataset_name
    args.base_model = base_model_name

    for ii in range(args.itr):
        exp = ExpConformal(args)

        setting = "{}_{}_model{}_cp{}_a{}_cw{}_xlag{}_seed{}_mode{}_{}_{}".format(
            dataset_name,
            args.exp_name,
            args.base_model,
            args.cp_mode,
            args.alpha,
            args.calib_window_size,
            args.x_lag,
            args.seed,
            args.run_mode,
            f"{args.des}-{ii}",
            getattr(args, "ablation_mode", "M0"),
        )

        print(f">>>>>>> start : {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>")

        if hasattr(exp, "run"):
            exp.run(setting)
        else:
            raise AttributeError("ExpConformal must implement run(setting).")

        if args.gpu_type == "mps" and hasattr(torch.backends, "mps"):
            try:
                torch.backends.mps.empty_cache()
            except Exception:
                pass
        elif args.gpu_type == "cuda":
            torch.cuda.empty_cache()

        print(f">>>>>>> done  : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<")