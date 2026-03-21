import csv
import os
import pandas as pd
from typing import Dict


class ResultLogger:

    CONFORMAL_HEADER = [
        "setting",
        "dataset",
        "base_model",
        "cp_mode",
        "run_mode",
        "alpha",
        "seed",
        "x_lag",
        "ablation_mode",
        "target_coverage",
        "coverage",
        "coverage_gap",
        "avg_width",
        "ces",
        "rcs",
        "point_mse",
        "point_mae",
        "cache_path",
        "comment",
    ]

    ADAPTIVE_HEADER = [
        "setting",
        "dataset",
        "base_model",
        "cp_mode",
        "run_mode",
        "alpha",
        "seed",
        "x_lag",
        "ablation_mode",
        "target_coverage",
        "worst_window_coverage",
        "width_step_mean",
        "width_std",
        "control_alpha_step_mean",
        "control_alpha_std",
        "cache_path",
        "comment",
    ]

    def __init__(
        self,
        conformal_csv_path: str = "results/conformal_results.csv",
        adaptive_csv_path: str = "results/adaptive_conformal_results.csv",
    ):
        self.conformal_csv_path = conformal_csv_path
        self.adaptive_csv_path = adaptive_csv_path

        os.makedirs(os.path.dirname(self.conformal_csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.adaptive_csv_path), exist_ok=True)

        self._ensure_header(self.conformal_csv_path, self.CONFORMAL_HEADER)
        self._ensure_header(self.adaptive_csv_path, self.ADAPTIVE_HEADER)

    @staticmethod
    def _ensure_header(csv_path: str, header: list):
        if not os.path.exists(csv_path):
            with open(csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)

    @staticmethod
    def _append_row(csv_path: str, header: list, row_dict: Dict):
        row = [row_dict.get(col, None) for col in header]
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def log_conformal(
        self,
        setting: str,
        dataset: str,
        base_model: str,
        cp_mode: str,
        run_mode: str,
        alpha: float,
        seed: int,
        x_lag: int,
        ablation_mode: str,
        target_coverage: float,
        metrics: Dict,
        cache_path: str = "",
        comment: str = "",
    ):
        row = {
            "setting": setting,
            "dataset": dataset,
            "base_model": base_model,
            "cp_mode": cp_mode,
            "run_mode": run_mode,
            "alpha": alpha,
            "seed": seed,
            "x_lag": x_lag,
            "ablation_mode": ablation_mode,
            "target_coverage": target_coverage,
            "coverage": metrics.get("coverage"),
            "coverage_gap": metrics.get("coverage_gap"),
            "avg_width": metrics.get("avg_width"),
            "ces": metrics.get("ces"),
            "rcs": metrics.get("rcs"),
            "point_mse": metrics.get("point_mse"),
            "point_mae": metrics.get("point_mae"),
            "cache_path": cache_path,
            "comment": comment,
        }
        self._append_row(self.conformal_csv_path, self.CONFORMAL_HEADER, row)

    def log_adaptive(
        self,
        setting: str,
        dataset: str,
        base_model: str,
        cp_mode: str,
        run_mode: str,
        alpha: float,
        seed: int,
        x_lag: int,
        ablation_mode: str,
        target_coverage: float,
        adaptive_metrics: Dict,
        cache_path: str = "",
        comment: str = "",
    ):
        row = {
            "setting": setting,
            "dataset": dataset,
            "base_model": base_model,
            "cp_mode": cp_mode,
            "run_mode": run_mode,
            "alpha": alpha,
            "seed": seed,
            "x_lag": x_lag,
            "ablation_mode": ablation_mode,
            "target_coverage": target_coverage,
            "worst_window_coverage": adaptive_metrics.get("worst_window_coverage"),
            "width_step_mean": adaptive_metrics.get("width_step_mean"),
            "width_std": adaptive_metrics.get("width_std"),
            "control_alpha_step_mean": adaptive_metrics.get("control_alpha_step_mean"),
            "control_alpha_std": adaptive_metrics.get("control_alpha_std"),
            "cache_path": cache_path,
            "comment": comment,
        }
        self._append_row(self.adaptive_csv_path, self.ADAPTIVE_HEADER, row)

    def save(self):
        pass

    def to_excel(self):
        out = {}

        df1 = pd.read_csv(self.conformal_csv_path)
        xlsx1 = self.conformal_csv_path.replace(".csv", ".xlsx")
        df1.to_excel(xlsx1, index=False)
        out["conformal"] = xlsx1

        df2 = pd.read_csv(self.adaptive_csv_path)
        xlsx2 = self.adaptive_csv_path.replace(".csv", ".xlsx")
        df2.to_excel(xlsx2, index=False)
        out["adaptive"] = xlsx2

        return out