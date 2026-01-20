# exp/exp_basic.py
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.utils import normalize_series, create_lagged_features


@dataclass
class DataSplitConfig:
    train_ratio: float = 0.6
    calib_ratio: float = 0.2  # 剩下自动归到 test
    lags: int = 24
    batch_size: int = 64
    num_workers: int = 0
    shuffle_train: bool = True


class ExpBasic:
    """
    共形预测项目的基础实验类：
    - 加载 CSV 中的一列时间序列
    - 归一化
    - 按比例切分为 train / calib / test
    - 将序列转换为 (X, y) 滞后特征
    - 构造 PyTorch DataLoader
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and getattr(args, "use_gpu", False) else "cpu"
        )

        self.data_cfg = DataSplitConfig(
            train_ratio=getattr(args, "train_ratio", 0.6),
            calib_ratio=getattr(args, "calib_ratio", 0.2),
            lags=getattr(args, "lags", 24),
            batch_size=getattr(args, "batch_size", 64),
            num_workers=getattr(args, "num_workers", 0),
            shuffle_train=getattr(args, "shuffle_train", True),
        )

    # -------------------- 数据相关 -------------------- #
    def _load_series_from_csv(self) -> np.ndarray:
        """
        从 CSV 文件中读取一列时间序列。
        假设 args.data_path 指向 CSV 文件路径，
        args.target_col 为要使用的列名（例如 '0T'、'OT' 或 'close'）。
        """
        data_path = self.args.data_path
        target_col = getattr(self.args, "target_col", None)

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"CSV file not found: {data_path}")

        df = pd.read_csv(data_path)

        # 如果用户没有指定 target_col，默认用最后一列
        if target_col is None:
            target_col = df.columns[-1]

        if target_col not in df.columns:
            raise ValueError(
                f"Column '{target_col}' not found in CSV. Available columns: {list(df.columns)}"
            )

        series = df[target_col].astype(float).values
        series = normalize_series(series)
        
        print(">>> Loading CSV from:", data_path)

        return series

    def _split_series_indices(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        根据比例切分索引为 train / calib / test。
        注意：真正做成 (X, y) 的时候还要考虑 lags，因此这里只是粗分。
        """
        train_end = int(n * self.data_cfg.train_ratio)
        calib_end = int(n * (self.data_cfg.train_ratio + self.data_cfg.calib_ratio))

        idx_train = np.arange(0, train_end)
        idx_calib = np.arange(train_end, calib_end)
        idx_test = np.arange(calib_end, n)

        return idx_train, idx_calib, idx_test

    def _build_datasets_and_loaders(
        self, series: np.ndarray
    ) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray, np.ndarray]:
        """
        将原始序列转换为 (X, y) 并建立 DataLoader。
        返回：
            train_loader, calib_loader, test_loader,
            y_train, y_calib, y_test（numpy，用于后续分析）
        """
        lags = self.data_cfg.lags

        # 用整条序列先生成 (X, y)
        X_all, y_all = create_lagged_features(series, lags=lags)
        n_samples = len(y_all)

        # 按样本数切分索引
        idx_train, idx_calib, idx_test = self._split_series_indices(n_samples)

        X_train, y_train = X_all[idx_train], y_all[idx_train]
        X_calib, y_calib = X_all[idx_calib], y_all[idx_calib]
        X_test, y_test = X_all[idx_test], y_all[idx_test]

        # 转成 Tensor
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
            shuffle=self.data_cfg.shuffle_train,
            num_workers=self.data_cfg.num_workers,
        )
        calib_loader = DataLoader(
            calib_dataset,
            batch_size=self.data_cfg.batch_size,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.data_cfg.batch_size,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
        )

        return (
            train_loader,
            calib_loader,
            test_loader,
            y_train,
            y_calib,
            y_test,
        )

    # -------------------- 对外接口 -------------------- #
    def get_data(
        self,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray, np.ndarray]:
        """
        顶层数据接口：从 CSV 加载序列，切分，构造 DataLoader。
        """
        series = self._load_series_from_csv()
        return self._build_datasets_and_loaders(series)
