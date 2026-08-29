import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from utils.timefeatures import time_features


class CSVTimeSeriesDataset(Dataset):
    """
    通用 CSV 时间序列数据集（不做 train/val/test 划分）。

    适用于：
        - dataset/cryptocurrency/*.csv
        - dataset/stock/*.csv
        - dataset/exchange_rate/*.csv

    要求 CSV 列格式：
        date, <feat_1>, <feat_2>, ..., OT

    说明：
        - 'date'：可以是 '2020-10-05' 或 '2020-10-05 23:59:59' 等任意可解析时间
        - 'OT'  ：预测目标（你已经把 close 改成 OT）
        - 其它列全部当作特征（features='M' 时）
    """

    def __init__(
        self,
        args,
        root_path: str,
        data_path: str,
        size: Optional[List[int]] = None,
        features: str = "M",
        target: str = "OT",
        scale: bool = True,
        timeenc: int = 0,
        freq: str = "d",
    ):
        """
        Args:
            args      : 你的全局参数（这里只存起来，不强依赖）
            root_path : 数据所在目录，例如 'dataset/cryptocurrency'
            data_path : 文件名，例如 'AAVE.csv'
            size      : [seq_len, label_len, pred_len]
            features  : 'M' 多变量；'S' 单变量只用 target
            target    : 目标列名（默认 'OT'）
            scale     : 是否对特征做 StandardScaler
            timeenc   : 0 = 手工时间特征；1 = 用 time_features
            freq      : 频率，用于 time_features（'d'、'h' 等）
        """
        self.args = args
        self.root_path = root_path
        self.data_path = data_path

        # 窗口长度
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 24
        else:
            self.seq_len, self.label_len, self.pred_len = size

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.__read_data__()

    # ------------------ 核心预处理 ------------------ #
    def __read_data__(self):
        self.scaler = StandardScaler()

        csv_path = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(csv_path)

        # 检查基本列
        if "date" not in df_raw.columns:
            raise ValueError(f"{csv_path} 必须包含列 'date'")
        if self.target not in df_raw.columns:
            raise ValueError(f"{csv_path} 中找不到目标列 '{self.target}'")

        # 按时间排序
        df_raw["date"] = pd.to_datetime(df_raw["date"])
        df_raw = df_raw.sort_values("date").reset_index(drop=True)

        # 统一列顺序：date, [其它特征...], target
        cols = df_raw.columns.tolist()
        cols.remove("date")
        cols.remove(self.target)
        df_raw = df_raw[["date"] + cols + [self.target]]

        # 选择特征列
        if self.features == "S":
            df_data = df_raw[[self.target]]
        else:  # 'M' 或 'MS'
            df_data = df_raw[df_raw.columns[1:]]  # 除 date 外全用

        # 标准化：这里是对整条序列拟合，如果你想只用 train 段拟合，
        # 可以在外部再传入一个 pre_fitted_scaler 替换掉这个。
        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 时间特征
        df_stamp = df_raw[["date"]]
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.dt.month
            df_stamp["day"] = df_stamp.date.dt.day
            df_stamp["weekday"] = df_stamp.date.dt.weekday
            df_stamp["hour"] = df_stamp.date.dt.hour
            data_stamp = df_stamp.drop(columns=["date"]).values
        else:
            data_stamp = time_features(df_stamp["date"].values, freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # 整条时间序列（之后通过 sliding window 划分成样本）
        self.data_x = data
        self.data_y = data  # 这里默认预测同一组变量，外部可以只取 target 通道
        self.data_stamp = data_stamp

    # ------------------ Dataset 接口 ------------------ #
    def __getitem__(self, index: int):
        """
        返回一个样本：
            输入:  seq_x       长度 = seq_len
            目标:  seq_y       长度 = label_len + pred_len
            时间编码: seq_x_mark, seq_y_mark
        不区分 train/val/test，索引由外部控制。
        """
        s_begin = index
        s_end = s_begin + self.seq_len

        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self) -> int:
        # 能取到的最大起点是：len - seq_len - pred_len
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """把模型输出从标准化空间还原为原始数值。"""
        return self.scaler.inverse_transform(data)
