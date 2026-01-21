import sys
import os
import argparse
from typing import List, Tuple
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt

from .exp_basic import ExpBasic
from src.base_conformal.builder import build_conformal_predictor
from src.utils import compute_coverage, compute_average_width
from src.result_logger import ResultLogger

from datetime import datetime
import inspect


def _call_with_accepted_kwargs(fn, **kwargs):
    """
    Call fn with only the kwargs it accepts.
    Works for different CP baselines with different signatures.
    """
    sig = inspect.signature(fn)
    params = sig.parameters

    # if fn accepts **kwargs, pass all
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**kwargs)

    filtered = {k: v for k, v in kwargs.items() if k in params}
    return fn(**filtered)


def _normalize_interval(interval):
    """Normalize different interval formats to a (lower, upper) float tuple."""
    if interval is None:
        raise ValueError("CP returned None interval.")
    if isinstance(interval, dict):
        lo = interval.get("lower", interval.get("lo"))
        hi = interval.get("upper", interval.get("hi"))
        if lo is None or hi is None:
            raise ValueError(f"Unrecognized interval dict keys: {list(interval.keys())}")
        return float(lo), float(hi)
    if isinstance(interval, (list, tuple)) and len(interval) == 2:
        return float(interval[0]), float(interval[1])
    arr = np.asarray(interval)
    if arr.shape == (2,):
        return float(arr[0]), float(arr[1])
    if arr.ndim >= 1 and arr.shape[-1] == 2:
        # if batched, return as-is (caller should handle); here we take first element
        return float(arr.reshape(-1, 2)[0, 0]), float(arr.reshape(-1, 2)[0, 1])
    raise ValueError(f"Unrecognized interval format: type={type(interval)} shape={getattr(arr, 'shape', None)}")


def _cp_predict(cp, *, y_pred, uncertainty=None, x=None, step=None, horizon=None):
    """
    Robustly call cp.predict with a superset of kwargs, filtering unsupported ones.
    We pass multiple common aliases so different baselines can work unchanged.
    """
    kwargs = {
        # prediction aliases
        "base_prediction": y_pred,
        "y_pred": y_pred,
        "y_hat": y_pred,
        "pred": y_pred,
        "prediction": y_pred,

        # uncertainty aliases
        "model_uncertainty": uncertainty,
        "uncertainty": uncertainty,
        "sigma": uncertainty,

        # features / conditioning
        "x": x,
        "X": x,
        "features": x,

        # time info
        "step": step,
        "t": step,
        "horizon": horizon,
        "h": horizon,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    interval = _call_with_accepted_kwargs(cp.predict, **kwargs)
    return _normalize_interval(interval)


def _cp_update(cp, *, y_true, y_pred, interval, x=None, step=None):
    """Robustly call cp.update/observe (whatever exists) with filtered kwargs."""
    fn = None
    if hasattr(cp, "update"):
        fn = cp.update
    elif hasattr(cp, "observe"):
        fn = cp.observe
    else:
        return  # some CP variants may be stateless

    kwargs = {
        # truth aliases
        "y_true": y_true,
        "y": y_true,
        "y_obs": y_true,

        # prediction aliases
        "y_pred": y_pred,
        "y_hat": y_pred,
        "y_pred_mean": y_pred,
        "base_prediction": y_pred,

        # interval aliases
        "prediction_interval": interval,
        "interval": interval,
        "pi": interval,

        # conditioning
        "x": x,
        "X": x,
        "features": x,

        # time info
        "step": step,
        "t": step,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    _call_with_accepted_kwargs(fn, **kwargs)


# ====== 让 Python 能看到项目根目录和 time_series_library 包 ====== #
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 尝试导入 time_series_library
try:
    import time_series_library
    import time_series_library.layers as tsl_layers
    import time_series_library.utils as tsl_utils
    import sys as _sys

    # 关键：让 "import layers" 实际等价于 "import time_series_library.layers"
    _sys.modules.setdefault("layers", tsl_layers)
    _sys.modules.setdefault("utils", tsl_utils)

    from time_series_library.models import MODEL_REGISTRY
    print("[DEBUG] time_series_library 加载自：", time_series_library.__file__)
    print("[DEBUG] 可用模型：", list(MODEL_REGISTRY.keys()))
except Exception as e:
    MODEL_REGISTRY = {}
    print("[ERROR] 导入 time_series_library.models 失败：", repr(e))
    print("[ERROR] 当前 sys.path 前几项：", sys.path[:5])
    print("[Warning] time_series_library.models 未找到，暂时只能使用线性模型 LinearForecastModel.")


# ======== 一个简单的线性基础预测模型（baseline） ======== #
class LinearForecastModel(nn.Module):
    """
    使用滞后特征的线性回归模型：
    输入: (batch_size, lags)
    输出: (batch_size, 1)
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)  # (B, 1)


# ======== 包装 Autoformer / FEDformer / iTransformer / Mamba 等模型的统一接口 ======== #
class TSModelWrapper(nn.Module):
    """
    把 time_series_library 里的复杂模型统一包装成：
        输入: (batch, lags)
        输出: (batch, 1)
    方便直接接入你现在的共形预测流水线。
    """
    def __init__(self, model_class, lags: int, device: torch.device, pred_len: int = 1):
        super().__init__()
        self.lags = lags
        self.pred_len = pred_len
        self.device = device

        label_len = max(1, lags // 2)

        self.configs = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=lags,
            label_len=label_len,
            pred_len=pred_len,
            enc_in=1,
            dec_in=1,
            c_out=1,
            d_model=64,
            d_ff=128,
            n_heads=4,
            e_layers=2,
            d_layers=1,
            dropout=0.1,
            embed="timeF",
            freq="h",
            moving_avg=25,
            factor=3,
            activation="gelu",
            num_class=1,
            individual=False,
            top_k=5,
            num_kernels=6,
            d_conv=4,
            expand=2,
        )

        self.inner_model = model_class(self.configs).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x_enc = x.unsqueeze(-1)  # (B, L, 1)

        x_mark_enc = torch.zeros((b, self.lags, 4), device=self.device)

        dec_len = self.configs.label_len + self.pred_len
        x_dec = torch.zeros((b, dec_len, 1), device=self.device)
        x_mark_dec = torch.zeros((b, dec_len, 4), device=self.device)

        out = self.inner_model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        if out.ndim == 3:
            return out[:, -1, :]  # (B, 1)
        elif out.ndim == 2:
            return out[:, -1].unsqueeze(-1)
        else:
            raise RuntimeError(f"Unexpected output shape from base model: {out.shape}")


def plot_prediction_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    intervals: List[Tuple[float, float]],
    save_path: str,
    max_points: int = 200,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n = min(len(y_true), max_points)
    if n == 0:
        print("[Plot] y_true 为空，跳过画图。")
        return

    x = np.arange(n)
    y_true_plot = y_true[:n]
    y_pred_plot = y_pred[:n]
    lowers = np.array([iv[0] for iv in intervals[:n]])
    uppers = np.array([iv[1] for iv in intervals[:n]])

    plt.figure(figsize=(12, 4))
    plt.plot(x, y_true_plot, label="True", linewidth=1.5)
    plt.plot(x, y_pred_plot, label="Prediction", linewidth=1.2, linestyle="--")
    plt.fill_between(x, lowers, uppers, alpha=0.2, label="Prediction Interval")

    plt.xlabel("Time step (index on test set)")
    plt.ylabel("Value")
    plt.title("Prediction Intervals on Test Set")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[Plot] Prediction interval figure saved to: {save_path}")


def plot_series(
    values: List[float],
    save_path: str,
    title: str,
    ylabel: str,
    max_points: int = 1000,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if len(values) == 0:
        print(f"[Plot] 空序列，跳过画图: {title}")
        return

    n = min(len(values), max_points)
    x = np.arange(n)
    y = np.array(values[:n])

    plt.figure(figsize=(10, 3))
    plt.plot(x, y, linewidth=1.2)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[Plot] Series figure saved to: {save_path}")


# ======== 共形预测实验类 ======== #
class ExpConformal(ExpBasic):
    def __init__(self, args):
        super().__init__(args)

        # 用于可视化的历史记录
        self.calib_alpha_history: List[float] = []
        self.test_alpha_history: List[float] = []                 # alpha used to FORM intervals
        self.test_alpha_after_update_history: List[float] = []    # alpha after update (optional)
        self.test_interval_widths: List[float] = []

        base_model_name = getattr(args, "base_model", "linear")

        if base_model_name.lower() == "linear":
            print("[BaseModel] 使用 LinearForecastModel 作为基础预测模型.")
            self.model = LinearForecastModel(input_dim=self.data_cfg.lags).to(self.device)
        else:
            if base_model_name not in MODEL_REGISTRY:
                raise ValueError(
                    f"指定的 base_model='{base_model_name}' 不在 MODEL_REGISTRY 中"
                    f"或只用 'linear'。当前可选: {list(MODEL_REGISTRY.keys())}"
                )
            model_class = MODEL_REGISTRY[base_model_name]
            print(f"[BaseModel] 使用 time_series_library 模型 '{base_model_name}' 作为基础预测模型.")
            self.model = TSModelWrapper(
                model_class=model_class,
                lags=self.data_cfg.lags,
                device=self.device,
                pred_len=1,
            ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=getattr(args, "lr", 1e-3))

        self.cp = build_conformal_predictor(args)

    def train_model(self, train_loader):
        self.model.train()
        n_epochs = getattr(self.args, "train_epochs", 20)

        for epoch in range(n_epochs):
            total_loss = 0.0
            for X, y in train_loader:
                X = X.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.criterion(y_hat, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * len(X)

            avg_loss = total_loss / len(train_loader.dataset)
            print(f"[Train] Epoch {epoch+1}/{n_epochs} | MSE: {avg_loss:.6f}")

    def calibrate(self, calib_loader) -> float:
        self.model.eval()
        init_len = getattr(self.args, "spectral_window", 64)
        if hasattr(self.cp, "initialize"):
            self.cp.initialize(initial_data=np.zeros(init_len))

        all_errors: List[float] = []
        self.calib_alpha_history = []

        with torch.no_grad():
            for X, y in calib_loader:
                X = X.to(self.device)
                y = y.to(self.device)

                y_hat = self.model(X)
                y_np = y.squeeze(-1).cpu().numpy()
                y_hat_np = y_hat.squeeze(-1).cpu().numpy()

                batch_errors = np.abs(y_np - y_hat_np)
                batch_uncertainty = float(np.std(batch_errors) + 1e-6)

                X_np = X.detach().cpu().numpy()

                for i, (yt, yp) in enumerate(zip(y_np, y_hat_np)):
                    x_i = X_np[i]
                    interval = _cp_predict(self.cp, y_pred=float(yp), uncertainty=batch_uncertainty, x=x_i)
                    _cp_update(self.cp, y_true=float(yt), y_pred=float(yp), interval=interval, x=x_i)

                    # ✅ change: record alpha per SAMPLE (not per batch)
                    self.calib_alpha_history.append(float(self.cp.alpha))

                all_errors.extend(batch_errors.tolist())

        calib_mse = float(np.mean(np.square(all_errors)))
        print(f"[Calib] MSE on calibration set: {calib_mse:.6f}")
        print(f"[Calib] Final adaptive alpha: {self.cp.alpha:.4f}")
        return calib_mse

    def evaluate(self, test_loader) -> Tuple[float, float, float, float]:
        self.model.eval()

        intervals: List[Tuple[float, float]] = []
        y_true_all: List[float] = []
        y_pred_all: List[float] = []

        self.test_interval_widths = []
        self.test_alpha_history = []
        self.test_alpha_after_update_history = []

        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(self.device)
                y = y.to(self.device)

                y_hat = self.model(X)
                y_np = y.squeeze(-1).cpu().numpy()
                y_hat_np = y_hat.squeeze(-1).cpu().numpy()

                batch_errors = np.abs(y_np - y_hat_np)
                batch_uncertainty = float(np.std(batch_errors) + 1e-6)

                X_np = X.detach().cpu().numpy()

                for i, (yt, yp) in enumerate(zip(y_np, y_hat_np)):
                    x_i = X_np[i]

                    interval = _cp_predict(self.cp, y_pred=float(yp), uncertainty=batch_uncertainty, x=x_i)
                    intervals.append(interval)
                    y_true_all.append(float(yt))
                    y_pred_all.append(float(yp))

                    lower, upper = interval
                    self.test_interval_widths.append(float(upper - lower))

                    # ✅ change: record alpha USED TO FORM interval (after predict, before update)
                    self.test_alpha_history.append(float(self.cp.alpha))

                    _cp_update(self.cp, y_true=float(yt), y_pred=float(yp), interval=interval, x=x_i)

                    # optional: record alpha AFTER update (distribution update)
                    self.test_alpha_after_update_history.append(float(self.cp.alpha))

        y_true_arr = np.array(y_true_all, dtype=float)
        y_pred_arr = np.array(y_pred_all, dtype=float)

        coverage = compute_coverage(y_true_arr, intervals)
        avg_width = compute_average_width(intervals)
        mse = float(np.mean((y_true_arr - y_pred_arr) ** 2))
        mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))

        print(f"[Test] Coverage: {coverage:.4f}")
        print(f"[Test] Avg. Interval Width: {avg_width:.4f}")
        print(f"[Test] MSE: {mse:.6f}, MAE: {mae:.6f}")

        base_model_name = getattr(self.args, "base_model", "linear")
        dataset_name = os.path.basename(self.args.data_path).replace(".csv", "")
        cp_mode = getattr(self.args, "cp_mode", "acp")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{dataset_name}_{base_model_name}_{cp_mode}_{timestamp}"

        pi_path = os.path.join("v_results", "prediction_intervals", f"{prefix}_prediction_intervals.png")
        plot_prediction_intervals(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            intervals=intervals,
            save_path=pi_path,
            max_points=200,
        )

        alpha_path = os.path.join("v_results", "alpha_curves", f"{prefix}_alpha_test.png")
        plot_series(
            values=self.test_alpha_history,
            save_path=alpha_path,
            title=f"Alpha used to form intervals on test set ({base_model_name}, {cp_mode})",
            ylabel="alpha",
            max_points=2000,
        )

        alpha_after_path = os.path.join("v_results", "alpha_curves", f"{prefix}_alpha_after_update_test.png")
        plot_series(
            values=self.test_alpha_after_update_history,
            save_path=alpha_after_path,
            title=f"Alpha after update on test set ({base_model_name}, {cp_mode})",
            ylabel="alpha",
            max_points=2000,
        )

        width_path = os.path.join("v_results", "interval_widths", f"{prefix}_interval_widths.png")
        plot_series(
            values=self.test_interval_widths,
            save_path=width_path,
            title=f"Prediction interval widths on test set ({base_model_name}, {cp_mode})",
            ylabel="width",
            max_points=2000,
        )

        return coverage, avg_width, mse, mae

    def run(self):
        if not os.path.exists("results"):
            os.makedirs("results")

        if not os.path.exists("v_results"):
            os.makedirs("v_results")
        for sub in ["prediction_intervals", "alpha_curves", "interval_widths"]:
            subdir = os.path.join("v_results", sub)
            if not os.path.exists(subdir):
                os.makedirs(subdir)

        logger = ResultLogger("results/conformal_results.csv")

        (
            train_loader,
            calib_loader,
            test_loader,
            _,
            _,
            _,
        ) = self.get_data()

        self.train_model(train_loader)

        calib_mse = self.calibrate(calib_loader)

        if hasattr(self.cp, "start_test"):
            self.cp.start_test()

        coverage, avg_width, mse, mae = self.evaluate(test_loader)

        dataset_name = os.path.basename(self.args.data_path)
        setting_name = (
            f"{dataset_name}_lags{self.args.lags}"
            f"_model{self.args.base_model}"
            f"_cp{self.args.cp_mode}"
        )

        logger.log(
            setting_name,
            metrics={
                "calib_mse": calib_mse,
                "mse": mse,
                "mae": mae,
                "coverage": coverage,
                "avg_width": avg_width,
                "final_alpha": self.cp.alpha,
            },
            extra_info={
                "data_path": self.args.data_path,
                "base_model": self.args.base_model,
                "cp_mode": self.args.cp_mode,
            },
        )

        excel_path = logger.to_excel()

        print("✔ Results saved to:")
        print("   CSV : results/conformal_results.csv")
        print("   XLSX:", excel_path)


def get_args():
    parser = argparse.ArgumentParser(description="Adaptive Conformal Prediction Experiment")

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--target_col", type=str, default=None)

    parser.add_argument("--lags", type=int, default=24)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--calib_ratio", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--shuffle_train", type=bool, default=True)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--use_gpu", action="store_true")

    parser.add_argument("--base_model", type=str, default="linear")

    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--spectral_window", type=int, default=64)
    parser.add_argument("--n_latent_states", type=int, default=3)
    parser.add_argument("--calib_window_size", type=int, default=200)
    parser.add_argument("--lambda_spectral", type=float, default=0.5)
    parser.add_argument("--min_calib_size", type=int, default=30)

    parser.add_argument(
        "--cp_mode",
        type=str,
        default="acp",
        choices=["acp", "standard", "aci", "agaci", "nex", "cqr", "dfpi", "enbpi"],
    )

    parser.add_argument("--agaci_gammas", type=float, nargs="+", default=[0.001, 0.01, 0.05, 0.1])
    parser.add_argument("--agaci_warmup_steps", type=int, default=50)

    parser.add_argument("--aci_split_size", type=float, default=0.75)
    parser.add_argument("--aci_t_init", type=int, default=200)
    parser.add_argument("--aci_max_iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--cqr_feature_window", type=int, default=2)
    parser.add_argument("--cqr_past_window", type=int, default=1)
    parser.add_argument("--cqr_min_calib_size", type=int, default=60)
    parser.add_argument("--cqr_split_ratio", type=float, default=0.5)
    parser.add_argument("--cqr_qr_l2", type=float, default=0.0)
    parser.add_argument("--cqr_solver", type=str, default="highs")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    exp = ExpConformal(args)
    exp.run()
    print("Experiment finished successfully.")
