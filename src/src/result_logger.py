import csv
import os
import pandas as pd
from datetime import datetime

class ResultLogger:
    def __init__(self, csv_path="results/conformal_results.csv"):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.csv_path = csv_path

        # 如果文件不存在，写入表头
        if not os.path.exists(csv_path):
            with open(csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "setting",
                    "mse",
                    "mae",
                    "coverage",
                    "avg_width",
                    "final_alpha",
                    "calib_mse",
                    "data_path",
                    "cp_mode",
                    "comment",
                ])

    def log(self, setting: str, metrics: dict, extra_info: dict = None):
        """追加一行实验结果到 CSV"""
        extra_info = extra_info or {}
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        row = [
            timestamp,
            setting,
            metrics.get("mse"),
            metrics.get("mae"),
            metrics.get("coverage"),
            metrics.get("avg_width"),
            metrics.get("final_alpha"),
            metrics.get("calib_mse"),
            extra_info.get("data_path"),
            extra_info.get("cp_mode"),
            extra_info.get("comment", ""),
        ]

        with open(self.csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def save(self):
        """为了兼容 exp_conformal.py，这里其实什么都不用做"""
        pass

    def to_excel(self):
        """将 CSV 转为 Excel 文件"""
        df = pd.read_csv(self.csv_path)
        excel_path = self.csv_path.replace(".csv", ".xlsx")
        df.to_excel(excel_path, index=False)
        return excel_path
