# exp/exp_forecast_cache.py
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from .exp_basic import ExpBasic


# ============================================================
# Make time_series_library imports robust
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
    print("[Warning] MODEL_REGISTRY unavailable; only 'linear' base model is usable.")


# ============================================================
# Base forecasting models
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

        x_mark_enc = torch.zeros((b, self.lags, 4), device=self.device)

        dec_len = self.configs.label_len + self.pred_len
        x_dec = torch.zeros((b, dec_len, 1), device=self.device)
        x_mark_dec = torch.zeros((b, dec_len, 4), device=self.device)

        out = self.inner_model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        if out.ndim == 3:
            return out[:, -1, :].reshape(b, 1)
        if out.ndim == 2:
            return out[:, -1].unsqueeze(-1)

        raise RuntimeError(f"Unexpected output shape from base model: {out.shape}")


# ============================================================
# Experiment
# ============================================================

class ExpForecastCache(ExpBasic):
    """
    Train/load a point forecaster and export aligned forecast cache for CP.
    """

    def __init__(self, args):
        super().__init__(args)

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
        self.optimizer = Adam(self.model.parameters(), lr=float(getattr(args, "lr", 1e-3)))

    def train_model(self, train_loader):
        self.model.train()
        n_epochs = int(getattr(self.args, "train_epochs", 20))

        last_mse = float("nan")
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

            last_mse = total_loss / max(1, len(train_loader.dataset))

        print(f"[Train] epochs={n_epochs} | final_mse={last_mse:.6f}")

    def _predict_loader(self, loader):
        self.model.eval()

        y_true_all = []
        y_pred_all = []

        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                y = y.to(self.device)

                y_hat = self.model(X)

                y_np = y.squeeze(-1).detach().cpu().numpy()
                y_hat_np = y_hat.squeeze(-1).detach().cpu().numpy()

                y_true_all.append(y_np)
                y_pred_all.append(y_hat_np)

        y_true = np.concatenate(y_true_all, axis=0).astype(float)
        y_pred = np.concatenate(y_pred_all, axis=0).astype(float)

        return y_true, y_pred

    def _default_cache_dir(self) -> str:
        data_name = os.path.splitext(os.path.basename(self.args.data_path))[0]
        base_model = getattr(self.args, "base_model", "linear")
        task_name = getattr(self.args, "task_name", "long_term_forecast")

        return os.path.join(
            "forecast_cache",
            f"{task_name}_{data_name}_cache_{base_model}"
        )

    def export_forecast_cache(self, save_path: Optional[str] = None):
        train_loader, calib_loader, test_loader, _, _, _ = self.get_data()

        self.train_model(train_loader)

        val_y_true, val_y_pred = self._predict_loader(calib_loader)
        test_y_true, test_y_pred = self._predict_loader(test_loader)

        # Since current ExpBasic produces already-ordered supervised samples,
        # aligned indices are just sequential indices inside each split.
        val_time_idx = np.arange(len(val_y_true), dtype=int)
        test_time_idx = np.arange(len(test_y_true), dtype=int)

        # Keep count for compatibility with your current cache schema.
        # In this cache version, each point prediction corresponds to one sample,
        # so count is all ones.
        val_count = np.ones(len(val_y_true), dtype=int)
        test_count = np.ones(len(test_y_true), dtype=int)

        if save_path is None:
            cache_dir = self._default_cache_dir()
            os.makedirs(cache_dir, exist_ok=True)
            save_path = os.path.join(cache_dir, "forecast_full.npz")
        else:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        np.savez(
            save_path,
            val_y_true_full=val_y_true,
            val_y_pred_full=val_y_pred,
            val_time_idx=val_time_idx,
            val_count=val_count,
            test_y_true_full=test_y_true,
            test_y_pred_full=test_y_pred,
            test_time_idx=test_time_idx,
            test_count=test_count,
        )

        print(f"[Cache] saved to: {save_path}")
        print(f"[Cache] val length : {len(val_y_true)}")
        print(f"[Cache] test length: {len(test_y_true)}")

        return save_path

    def run(self):
        save_path = getattr(self.args, "cache_save_path", None)
        return self.export_forecast_cache(save_path=save_path)