# exp_conformal.py
from __future__ import annotations

import os
import sys
import copy
import inspect
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque
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

# ============================================================
# Robust CP call helpers (compat ACP.step or ACI/EnbPI predict+update)
# ============================================================

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
        # batched; take first
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
        return  # stateless CP variants

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


def _try_get_alpha(cp) -> float:
    """
    Best-effort alpha snapshot. Works for:
    - ACI/EnbPI-like baselines: cp.alpha exists
    - Your ACP: may also have cp.alpha or alpha_history
    """
    if hasattr(cp, "alpha_history") and isinstance(getattr(cp, "alpha_history"), list) and len(cp.alpha_history) > 0:
        a = cp.alpha_history[-1]
        try:
            return float(a)
        except Exception:
            pass
    if hasattr(cp, "alpha"):
        try:
            return float(cp.alpha)
        except Exception:
            return float("nan")
    return float("nan")


def _unified_cp_step(
    cp,
    *,
    y_pred: float,
    y_true: float,
    uncertainty: float,
    x=None,
    step=None,
    horizon=None,
    update: bool = True,
) -> Tuple[Tuple[float, float], float, float]:
    """
    Return: (interval(lo,hi), alpha_used, alpha_after)

    Priority:
      1) predict + (optional) update/observe  [best for most CPs, and for your ACP predict/update semantics]
      2) closed-loop step(...)                [fallback for CPs that only implement step]
    """

    # --------- 1) prefer predict/update if predict exists ----------
    if hasattr(cp, "predict") and callable(getattr(cp, "predict")):
        a_before = _try_get_alpha(cp)

        interval = _cp_predict(cp, y_pred=y_pred, uncertainty=uncertainty, x=x, step=step, horizon=horizon)
        a_used = _try_get_alpha(cp)
        if not np.isfinite(a_used):
            a_used = a_before

        if update:
            _cp_update(cp, y_true=y_true, y_pred=y_pred, interval=interval, x=x, step=step)

        a_after = _try_get_alpha(cp)

        # for your ACP: after update, mu_global is often the more meaningful "post-update alpha center"
        if hasattr(cp, "_alpha") and hasattr(cp._alpha, "mu_global"):
            try:
                a_after = float(cp._alpha.mu_global)
            except Exception:
                pass

        return interval, float(a_used), float(a_after)

    # --------- 2) fallback: cp.step if no predict ----------
    if hasattr(cp, "step") and callable(getattr(cp, "step")):
        a_before = _try_get_alpha(cp)
        interval = _call_with_accepted_kwargs(
            cp.step,
            base_prediction=y_pred,
            y_true=y_true,
            model_uncertainty=uncertainty,
            y_pred=y_pred,
            uncertainty=uncertainty,
            x=x,
            step=step,
            horizon=horizon,
        )
        lo, hi = _normalize_interval(interval)

        a_used = _try_get_alpha(cp)
        if not np.isfinite(a_used):
            a_used = a_before

        a_after = a_used
        if hasattr(cp, "_alpha") and hasattr(cp._alpha, "mu_global"):
            try:
                a_after = float(cp._alpha.mu_global)
            except Exception:
                pass

        return (float(lo), float(hi)), float(a_used), float(a_after)

    raise AttributeError("CP object has neither predict nor step.")


# ============================================================
# Make time_series_library imports robust (and fix "import layers")
# ============================================================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import time_series_library
    import time_series_library.layers as tsl_layers
    import time_series_library.utils as tsl_utils
    import sys as _sys

    _sys.modules.setdefault("layers", tsl_layers)
    _sys.modules.setdefault("utils", tsl_utils)

    from time_series_library.models import MODEL_REGISTRY
    print("[DEBUG] time_series_library loaded from:", time_series_library.__file__)
    print("[DEBUG] available models:", list(MODEL_REGISTRY.keys()))
except Exception as e:
    MODEL_REGISTRY = {}
    print("[ERROR] importing time_series_library.models failed:", repr(e))
    print("[ERROR] sys.path head:", sys.path[:5])
    print("[Warning] MODEL_REGISTRY unavailable; only 'linear' base model is usable.")


# ============================================================
# Base forecasting models (point prediction)
# ============================================================

class LinearForecastModel(nn.Module):
    """
    Lag-feature linear model:
    input : (B, lags)
    output: (B, 1)
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


class TSModelWrapper(nn.Module):
    """
    Wrap a time_series_library model into:
      input : (B, lags)
      output: (B, 1) (one-step)
    """
    def __init__(self, model_class, lags: int, device: torch.device, pred_len: int = 1):
        super().__init__()
        self.lags = int(lags)
        self.pred_len = int(pred_len)
        self.device = device

        label_len = max(1, self.lags // 2)

        # NOTE: these are generic defaults; your ExpBasic/data_cfg should still decide lags etc.
        self.configs = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=self.lags,
            label_len=label_len,
            pred_len=self.pred_len,
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

        # If your dataset provides time marks, you can pass them in instead of zeros.
        x_mark_enc = torch.zeros((b, self.lags, 4), device=self.device)

        dec_len = self.configs.label_len + self.pred_len
        x_dec = torch.zeros((b, dec_len, 1), device=self.device)
        x_mark_dec = torch.zeros((b, dec_len, 4), device=self.device)

        out = self.inner_model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        if out.ndim == 3:
            # (B, pred_len, c_out) -> take last step, last channel
            return out[:, -1, :].reshape(b, 1)
        if out.ndim == 2:
            return out[:, -1].unsqueeze(-1)
        raise RuntimeError(f"Unexpected output shape from base model: {out.shape}")


# ============================================================
# Plot utils
# ============================================================

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
        print("[Plot] empty y_true, skip plotting.")
        return

    x = np.arange(n)
    lowers = np.array([iv[0] for iv in intervals[:n]], dtype=float)
    uppers = np.array([iv[1] for iv in intervals[:n]], dtype=float)

    plt.figure(figsize=(12, 4))
    plt.plot(x, y_true[:n], label="True", linewidth=1.5)
    plt.plot(x, y_pred[:n], label="Prediction", linewidth=1.2, linestyle="--")
    plt.fill_between(x, lowers, uppers, alpha=0.2, label="Prediction Interval")
    plt.xlabel("Time step (index on test set)")
    plt.ylabel("Value")
    plt.title("Prediction Intervals on Test Set")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Plot] saved: {save_path}")


def plot_series(values: List[float], save_path: str, title: str, ylabel: str, max_points: int = 1000):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if len(values) == 0:
        print(f"[Plot] empty series, skip: {title}")
        return

    n = min(len(values), max_points)
    x = np.arange(n)
    y = np.array(values[:n], dtype=float)

    plt.figure(figsize=(10, 3))
    plt.plot(x, y, linewidth=1.2)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Plot] saved: {save_path}")


# ============================================================
# Experiment
# ============================================================

class ExpConformal(ExpBasic):
    def __init__(self, args):
        super().__init__(args)

        # visualization histories
        self.calib_alpha_history: List[float] = []
        self.test_alpha_history: List[float] = []               # alpha used to FORM intervals
        self.test_alpha_after_update_history: List[float] = []  # alpha after update
        self.test_interval_widths: List[float] = []

        base_model_name = getattr(args, "base_model", "linear")

        if base_model_name.lower() == "linear":
            print("[BaseModel] Using LinearForecastModel.")
            self.model = LinearForecastModel(input_dim=self.data_cfg.lags).to(self.device)
        else:
            if base_model_name not in MODEL_REGISTRY:
                raise ValueError(
                    f"base_model='{base_model_name}' not found in MODEL_REGISTRY. "
                    f"Available: {list(MODEL_REGISTRY.keys())} (or use 'linear')."
                )
            model_class = MODEL_REGISTRY[base_model_name]
            print(f"[BaseModel] Using time_series_library model '{base_model_name}'.")
            self.model = TSModelWrapper(
                model_class=model_class,
                lags=self.data_cfg.lags,
                device=self.device,
                pred_len=1,
            ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=getattr(args, "lr", 1e-3))

        # CP predictor (ACP/ACI/EnbPI/etc)
        self.cp = build_conformal_predictor(args)

        # rolling uncertainty from past residuals (prevents y_true leakage in test)
        self._err_hist = deque(maxlen=int(getattr(args, "unc_window", 256)))

    def _rolling_uncertainty(self) -> float:
        if len(self._err_hist) >= 8:
            return float(np.std(np.asarray(self._err_hist, dtype=float)) + 1e-6)
        return 1.0

    def train_model(self, train_loader):
        self.model.train()
        n_epochs = int(getattr(self.args, "train_epochs", 20))

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

            avg_loss = total_loss / max(1, len(train_loader.dataset))
            print(f"[Train] Epoch {epoch+1}/{n_epochs} | MSE: {avg_loss:.6f}")

    def calibrate(self, calib_loader) -> float:
        """
        Calibration: always update CP state.
        Compatible with:
          - ACP: cp.step(...) (closed-loop)
          - ACI/EnbPI: cp.predict + cp.update/observe
        """
        self.model.eval()

        # Optional init for CPs that need it (keep for compatibility)
        init_len = int(getattr(self.args, "spectral_window", 64))
        if hasattr(self.cp, "initialize") and callable(getattr(self.cp, "initialize")):
            try:
                self.cp.initialize(initial_data=np.zeros(init_len, dtype=float))
            except TypeError:
                # different signature; ignore
                pass

        all_abs_errors: List[float] = []
        self.calib_alpha_history = []

        with torch.no_grad():
            for X, y in calib_loader:
                X = X.to(self.device)
                y = y.to(self.device)

                y_hat = self.model(X)
                y_np = y.squeeze(-1).detach().cpu().numpy()
                y_hat_np = y_hat.squeeze(-1).detach().cpu().numpy()
                X_np = X.detach().cpu().numpy()

                for i, (yt, yp) in enumerate(zip(y_np, y_hat_np)):
                    x_i = X_np[i]
                    unc = self._rolling_uncertainty()

                    interval, a_used, _a_after = _unified_cp_step(
                        self.cp,
                        y_pred=float(yp),
                        y_true=float(yt),
                        uncertainty=float(unc),
                        x=x_i,
                        step=None,
                        horizon=None,
                        update=True,  # calibration updates
                    )

                    self.calib_alpha_history.append(float(a_used))

                    err = float(abs(float(yt) - float(yp)))
                    self._err_hist.append(err)
                    all_abs_errors.append(err)

        calib_mse = float(np.mean(np.square(all_abs_errors))) if len(all_abs_errors) > 0 else float("nan")
        print(f"[Calib] MSE on calibration set: {calib_mse:.6f}")
        print(f"[Calib] Final adaptive alpha (if exists): {_try_get_alpha(self.cp):.4f}")
        return calib_mse

    def evaluate(self, test_loader, update: bool = True) -> Tuple[float, float, float, float]:
        """
        Test:
          - online mode: update=True => CP updates on test stream
          - eval mode: update=False => do NOT mutate CP; implemented via deepcopy per step
        """
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
                y_np = y.squeeze(-1).detach().cpu().numpy()
                y_hat_np = y_hat.squeeze(-1).detach().cpu().numpy()
                X_np = X.detach().cpu().numpy()

                for i, (yt, yp) in enumerate(zip(y_np, y_hat_np)):
                    x_i = X_np[i]
                    unc = self._rolling_uncertainty()

                    # if not updating on test, run on a copy so original cp is unchanged
                    cp_used = self.cp if update else copy.deepcopy(self.cp)

                    interval, a_used, a_after = _unified_cp_step(
                        cp_used,
                        y_pred=float(yp),
                        y_true=float(yt),
                        uncertainty=float(unc),
                        x=x_i,
                        step=None,
                        horizon=None,
                        update=True,  # always update the cp_used; deepcopy handles eval-mode safety
                    )

                    intervals.append(interval)
                    y_true_all.append(float(yt))
                    y_pred_all.append(float(yp))

                    lo, hi = interval
                    self.test_interval_widths.append(float(hi - lo))
                    self.test_alpha_history.append(float(a_used))
                    self.test_alpha_after_update_history.append(float(a_after))

                    err = float(abs(float(yt) - float(yp)))
                    self._err_hist.append(err)

        y_true_arr = np.array(y_true_all, dtype=float)
        y_pred_arr = np.array(y_pred_all, dtype=float)

        coverage = compute_coverage(y_true_arr, intervals)
        avg_width = compute_average_width(intervals)
        mse = float(np.mean((y_true_arr - y_pred_arr) ** 2)) if len(y_true_arr) else float("nan")
        mae = float(np.mean(np.abs(y_true_arr - y_pred_arr))) if len(y_true_arr) else float("nan")

        print(f"[Test] Coverage: {coverage:.4f}")
        print(f"[Test] Avg. Interval Width: {avg_width:.4f}")
        print(f"[Test] MSE: {mse:.6f}, MAE: {mae:.6f}")

        # ---------- plots ----------
        base_model_name = getattr(self.args, "base_model", "linear")
        dataset_name = os.path.basename(self.args.data_path).replace(".csv", "")
        cp_mode = getattr(self.args, "cp_mode", "cp")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_mode = getattr(self.args, "run_mode", "online")
        seed = getattr(self.args, "seed", "NA")
        prefix = f"{dataset_name}_{base_model_name}_{cp_mode}_{run_mode}_seed{seed}_{timestamp}"

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

    def run(self, setting: str = None):
        os.makedirs("results", exist_ok=True)
        os.makedirs("v_results", exist_ok=True)
        for sub in ["prediction_intervals", "alpha_curves", "interval_widths"]:
            os.makedirs(os.path.join("v_results", sub), exist_ok=True)

        logger = ResultLogger("results/conformal_results.csv")

        train_loader, calib_loader, test_loader, _, _, _ = self.get_data()

        self.train_model(train_loader)

        calib_mse = self.calibrate(calib_loader)

        if hasattr(self.cp, "start_test") and callable(getattr(self.cp, "start_test")):
            try:
                self.cp.start_test()
            except Exception:
                pass

        run_mode = getattr(self.args, "run_mode", "online")
        update_on_test = (run_mode == "online")  # online: update; eval: fixed

        coverage, avg_width, mse, mae = self.evaluate(test_loader, update=update_on_test)

        if setting is None:
            dataset_name = os.path.basename(self.args.data_path)
            base_model = getattr(self.args, "base_model", "linear")
            cp_mode = getattr(self.args, "cp_mode", "cp")
            lags = getattr(self.data_cfg, "lags", getattr(self.args, "lags", "NA"))
            seed = getattr(self.args, "seed", "NA")
            setting = f"{dataset_name}_lags{lags}_model{base_model}_cp{cp_mode}_mode{run_mode}_seed{seed}"

        logger.log(
            setting,
            metrics={
                "calib_mse": calib_mse,
                "mse": mse,
                "mae": mae,
                "coverage": coverage,
                "avg_width": avg_width,
                "final_alpha": float(_try_get_alpha(self.cp)),
                "run_mode": run_mode,
                "update_on_test": int(update_on_test),
            },
            extra_info={
                "data_path": self.args.data_path,
                "base_model": getattr(self.args, "base_model", "linear"),
                "cp_mode": getattr(self.args, "cp_mode", "cp"),
            },
        )

        excel_path = logger.to_excel()

        print("✔ Results saved to:")
        print("   CSV : results/conformal_results.csv")
        print("   XLSX:", excel_path)
