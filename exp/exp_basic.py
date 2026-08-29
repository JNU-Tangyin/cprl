# exp/exp_basic.py
import os
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.utils import normalize_series, create_lagged_features


@dataclass
class DataSplitConfig:
    train_ratio: float = 0.6
    calib_ratio: float = 0.2
    lags: int = 96
    batch_size: int = 32
    num_workers: int = 0
    shuffle_train: bool = True


class ExpBasic:

    def __init__(self, args):
        self.args = args

        # device is still centrally determined here (ExpConformal also has its own self.device; keep consistent)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and getattr(args, "use_gpu", False) else "cpu"
        )

        self.data_cfg = DataSplitConfig(
            train_ratio=float(getattr(args, "train_ratio", 0.6)),
            calib_ratio=float(getattr(args, "calib_ratio", 0.2)),
            lags=int(getattr(args, "lags", 96)),
            batch_size=int(getattr(args, "batch_size", 32)),
            num_workers=int(getattr(args, "num_workers", 0)),
            shuffle_train=bool(getattr(args, "shuffle_train", True)),
        )

        self._validate_data_cfg()

    # -------------------- validation -------------------- #
    def _validate_data_cfg(self) -> None:
        if not (0.0 < self.data_cfg.train_ratio < 1.0):
            raise ValueError(f"train_ratio must be in (0,1), got {self.data_cfg.train_ratio}")
        if not (0.0 <= self.data_cfg.calib_ratio < 1.0):
            raise ValueError(f"calib_ratio must be in [0,1), got {self.data_cfg.calib_ratio}")
        if self.data_cfg.train_ratio + self.data_cfg.calib_ratio >= 1.0:
            raise ValueError(
                f"train_ratio + calib_ratio must be < 1, got {self.data_cfg.train_ratio + self.data_cfg.calib_ratio}"
            )
        if self.data_cfg.lags <= 0:
            raise ValueError(f"lags must be positive, got {self.data_cfg.lags}")
        if self.data_cfg.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.data_cfg.batch_size}")
        if self.data_cfg.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.data_cfg.num_workers}")

    # -------------------- data loading -------------------- #
    def _load_series_from_csv(self) -> np.ndarray:
        """
        Load a 1D numeric series from CSV.

        Args expected:
        - args.data_path: path to csv
        - args.target_col: column name to use; if None, use last column

        Returns:
        - series: 1D numpy array (float), normalized by normalize_series
        """
        data_path = getattr(self.args, "data_path", None)
        target_col = getattr(self.args, "target_col", None)

        if not data_path:
            raise ValueError("args.data_path is required but empty/None.")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"CSV file not found: {data_path}")

        df = pd.read_csv(data_path)

        if df.shape[1] == 0:
            raise ValueError(f"CSV has no columns: {data_path}")

        # default: last column
        if target_col is None:
            target_col = df.columns[-1]

        if target_col not in df.columns:
            raise ValueError(
                f"Column '{target_col}' not found in CSV. Available columns: {list(df.columns)}"
            )

        series = df[target_col].astype(float).to_numpy()

        if len(series) <= self.data_cfg.lags:
            raise ValueError(
                f"Series too short for lags={self.data_cfg.lags}: len(series)={len(series)}"
            )

        series = normalize_series(series)

        print(f">>> Loading CSV: {data_path}")
        print(f">>> Using column: {target_col}")
        print(f">>> Series length: {len(series)} | lags: {self.data_cfg.lags}")

        return series

    # -------------------- splitting -------------------- #
    def _split_samples_indices(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split SAMPLE indices into train/calib/test chronologically.

        Important:
        - n_samples is the number of supervised samples after lag transformation,
          NOT the raw series length.

        Returns:
        - idx_train, idx_calib, idx_test
        """
        train_end = int(n_samples * self.data_cfg.train_ratio)
        calib_end = int(n_samples * (self.data_cfg.train_ratio + self.data_cfg.calib_ratio))

        # ensure non-empty splits if possible (best-effort)
        if train_end <= 0:
            raise ValueError(f"Train split is empty: n_samples={n_samples}, train_ratio={self.data_cfg.train_ratio}")
        if calib_end <= train_end:
            raise ValueError(
                f"Calib split is empty: n_samples={n_samples}, train_ratio={self.data_cfg.train_ratio}, calib_ratio={self.data_cfg.calib_ratio}"
            )
        if calib_end >= n_samples:
            raise ValueError(
                f"Test split is empty: n_samples={n_samples}, train+calib={self.data_cfg.train_ratio + self.data_cfg.calib_ratio}"
            )

        idx_train = np.arange(0, train_end)
        idx_calib = np.arange(train_end, calib_end)
        idx_test = np.arange(calib_end, n_samples)

        return idx_train, idx_calib, idx_test

    # -------------------- dataset / loader -------------------- #
    def _build_datasets_and_loaders(
        self, series: np.ndarray
    ) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert series into supervised samples (X,y), then split and build loaders.

        Returns:
            train_loader, calib_loader, test_loader,
            y_train, y_calib, y_test (numpy arrays for analysis/logging)
        """
        lags = self.data_cfg.lags

        # Build supervised samples from the FULL series first
        X_all, y_all = create_lagged_features(series, lags=lags)
        # X_all: (n_samples, lags), y_all: (n_samples,)
        n_samples = len(y_all)

        idx_train, idx_calib, idx_test = self._split_samples_indices(n_samples)

        X_train, y_train = X_all[idx_train], y_all[idx_train]
        X_calib, y_calib = X_all[idx_calib], y_all[idx_calib]
        X_test, y_test = X_all[idx_test], y_all[idx_test]

        # Torch tensors
        X_train_t = torch.from_numpy(X_train).float()
        y_train_t = torch.from_numpy(y_train).float().unsqueeze(-1)

        X_calib_t = torch.from_numpy(X_calib).float()
        y_calib_t = torch.from_numpy(y_calib).float().unsqueeze(-1)

        X_test_t = torch.from_numpy(X_test).float()
        y_test_t = torch.from_numpy(y_test).float().unsqueeze(-1)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        calib_dataset = TensorDataset(X_calib_t, y_calib_t)
        test_dataset = TensorDataset(X_test_t, y_test_t)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.data_cfg.batch_size,
            shuffle=self.data_cfg.shuffle_train,  # only training can shuffle
            num_workers=self.data_cfg.num_workers,
            drop_last=False,
        )
        calib_loader = DataLoader(
            calib_dataset,
            batch_size=self.data_cfg.batch_size,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.data_cfg.batch_size,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
            drop_last=False,
        )

        print(
            f">>> Samples (after lag): n={n_samples} | "
            f"train={len(train_dataset)} calib={len(calib_dataset)} test={len(test_dataset)}"
        )

        return train_loader, calib_loader, test_loader, y_train, y_calib, y_test

    # -------------------- public API -------------------- #
    def get_data(
        self,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray, np.ndarray]:
        """
        Public data API:
        - load series
        - build loaders + return split targets
        """
        series = self._load_series_from_csv()
        return self._build_datasets_and_loaders(series)
