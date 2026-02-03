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

try:
    from src.utils import print_args
except Exception:
    print_args = None


def parse_args():
    parser = argparse.ArgumentParser(description="ACP / Adaptive Conformal Prediction Runner")

    # ---- protocol ----
    parser.add_argument(
        "--run_mode",
        type=str,
        default="online",
        choices=["online", "eval"],
        help="online: update CP on test stream; eval: do NOT update CP on test stream",
    )
    parser.add_argument("--itr", type=int, default=1, help="repeat runs")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    # ---- naming / output ----
    parser.add_argument("--task_name", type=str, default="conformal")
    parser.add_argument("--exp_name", type=str, default="acp")
    parser.add_argument("--des", type=str, default="debug")
    parser.add_argument("--results_dir", type=str, default="./results")

    # ---- data ----
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--target_col", type=str, default=None)
    parser.add_argument("--lags", type=int, default=96)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--calib_ratio", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)

    # ---- point forecasting model ----
    parser.add_argument("--base_model", type=str, default="linear")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_epochs", type=int, default=20)

    # ---- conformal ----
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--cp_mode",
        type=str,
        default="acp",
        choices=["acp", "standard", "aci", "agaci", "nex", "cqr", "dfpi", "enbpi", "cptc", "hopcpt"],
    )
    parser.add_argument("--spectral_window", type=int, default=64)
    parser.add_argument("--n_latent_states", type=int, default=3)
    parser.add_argument("--calib_window_size", type=int, default=200)
    parser.add_argument("--lambda_spectral", type=float, default=0.5)
    parser.add_argument("--min_calib_size", type=int, default=30)

    # ---- device ----
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu_type", type=str, default="cuda", choices=["cuda", "mps", "cpu"])

    # CPTC
    parser.add_argument("--cptc_gamma", type=float, default=0.2)
    parser.add_argument("--cptc_warm_start", type=int, default=100)
    parser.add_argument("--cptc_max_width", type=float, default=3.0)
    parser.add_argument("--cptc_min_residuals", type=int, default=25)
    parser.add_argument("--cptc_warm_threshold", type=float, default=0.3)
    parser.add_argument("--cptc_aggregate_sort_asc", type=int, default=1)

    # HopCPT
    parser.add_argument("--hopcpt_emb_dim", type=int, default=32)
    parser.add_argument("--hopcpt_hidden_dim", type=int, default=64)
    parser.add_argument("--hopcpt_train_epochs", type=int, default=200)
    parser.add_argument("--hopcpt_lr", type=float, default=1e-3)
    parser.add_argument("--hopcpt_beta", type=float, default=1.0)
    parser.add_argument("--hopcpt_online_update", type=int, default=1)

    # ---- sensitive analysis ----
    parser.add_argument("--worst_window", type=int, default=100)

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

    # cuda
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device(f"cuda:{args.gpu}")
        print("Using CUDA:", args.device)
    else:
        args.device = torch.device("cpu")
        print("CUDA not available or --use_gpu not set, fallback to CPU")


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

    for ii in range(args.itr):
        exp = ExpConformal(args)

        setting = "{}_{}_lags{}_model{}_cp{}_a{}_cw{}_seed{}_mode{}_{}".format(
            args.task_name,
            args.exp_name,
            args.lags,
            args.base_model,
            args.cp_mode,
            args.alpha,
            args.calib_window_size,
            args.seed,
            args.run_mode,
            f"{args.des}-{ii}",
        )

        print(f">>>>>>> start : {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>")

        if hasattr(exp, "run"):
            exp.run(setting)
        else:
            raise AttributeError("ExpConformal must implement run(setting).")

        # cache cleanup
        if args.gpu_type == "mps" and hasattr(torch.backends, "mps"):
            try:
                torch.backends.mps.empty_cache()
            except Exception:
                pass
        elif args.gpu_type == "cuda":
            torch.cuda.empty_cache()

        print(f">>>>>>> done  : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<")
