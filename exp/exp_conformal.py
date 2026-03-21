# exp/exp_conformal.py
from __future__ import annotations

import os
import copy
import inspect
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque
import csv

import numpy as np
import matplotlib.pyplot as plt

from .exp_basic import ExpBasic
from src.base_conformal.builder import build_conformal_predictor
from src.utils import (
    compute_coverage,
    compute_average_width,
    compute_w_ref,
    compute_ces,
    compute_rcs,
    compute_worst_window_coverage,
    compute_alpha_step_mean,
    compute_series_std,
)
from src.result_logger import ResultLogger
from src.utils.cache_loader import load_cache_for_conformal


# ============================================================
# Robust CP call helpers
# ============================================================

def _call_with_accepted_kwargs(fn, **kwargs):
    sig = inspect.signature(fn)
    params = sig.parameters

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**kwargs)

    filtered = {k: v for k, v in kwargs.items() if k in params}
    return fn(**filtered)


def _normalize_interval(interval):
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
        arr2 = arr.reshape(-1, 2)
        return float(arr2[0, 0]), float(arr2[0, 1])

    raise ValueError(
        f"Unrecognized interval format: type={type(interval)} "
        f"shape={getattr(arr, 'shape', None)}"
    )


def _cp_predict(cp, *, y_pred, uncertainty=None, x=None, step=None, horizon=None):
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

        # conditioning
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
    fn = None
    if hasattr(cp, "update"):
        fn = cp.update
    elif hasattr(cp, "observe"):
        fn = cp.observe
    else:
        return

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
    kwargs = {k: v for k, v in kwargs.items() if k is not None}
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    _call_with_accepted_kwargs(fn, **kwargs)


def _try_get_alpha(cp) -> float:
    if hasattr(cp, "_last_alpha_sampled"):
        try:
            return float(cp._last_alpha_sampled)
        except Exception:
            pass

    if hasattr(cp, "alpha_history") and isinstance(getattr(cp, "alpha_history"), list) and len(cp.alpha_history) > 0:
        try:
            return float(cp.alpha_history[-1])
        except Exception:
            pass

    if hasattr(cp, "alpha"):
        try:
            return float(cp.alpha)
        except Exception:
            return float("nan")

    return float("nan")


def _unified_cp_step(cp, *, y_pred, y_true, uncertainty, x=None, step=None, horizon=None, update=True):
    a_before = _try_get_alpha(cp)
    interval = _cp_predict(cp, y_pred=y_pred, uncertainty=uncertainty, x=x, step=step, horizon=horizon)
    a_used = _try_get_alpha(cp)

    if update:
        _cp_update(cp, y_true=y_true, y_pred=y_pred, interval=interval, x=x, step=step)

    a_after = _try_get_alpha(cp)
    if hasattr(cp, "_alpha") and hasattr(cp._alpha, "mu_global"):
        try:
            a_after = float(cp._alpha.mu_global)
        except Exception:
            pass

    return interval, float(a_used if np.isfinite(a_used) else a_before), float(a_after)


# ============================================================
# Helpers
# ============================================================

def _rolling_quantiles(arr: List[float]):
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")

    med = float(np.median(a))
    q90 = float(np.quantile(a, 0.90))
    q25 = float(np.quantile(a, 0.25))
    q75 = float(np.quantile(a, 0.75))
    return med, q90, float(q75 - q25)


def _mean_abs_step(series: List[float]) -> float:
    if len(series) < 2:
        return float("nan")
    a = np.asarray(series, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return float("nan")
    return float(np.mean(np.abs(np.diff(a))))


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _write_dynamics_csv(path: str, rows: List[Dict]):
    _ensure_dir(os.path.dirname(path))
    if len(rows) == 0:
        return
    header = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(k, None) for k in header])

def _write_rows_csv(path: str, rows: List[Dict]):
    _ensure_dir(os.path.dirname(path))
    if len(rows) == 0:
        return
    header = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(k, None) for k in header])

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

        self.cp = build_conformal_predictor(args)

        self.calib_alpha_history: List[float] = []
        self.test_alpha_history: List[float] = []
        self.test_alpha_after_update_history: List[float] = []
        self.test_interval_widths: List[float] = []
        self.test_dynamics: List[Dict] = []

        # rolling uncertainty proxy from past absolute residuals
        self._err_hist = deque(maxlen=int(getattr(args, "unc_window", 256)))

    def _rolling_uncertainty(self) -> float:
        if len(self._err_hist) >= 8:
            return float(np.std(np.asarray(self._err_hist, dtype=float)) + 1e-6)
        return 1.0

    def calibrate_from_cache(self, cache_data) -> float:
        init_len = int(getattr(self.args, "spectral_window", 64))
        if hasattr(self.cp, "initialize") and callable(getattr(self.cp, "initialize")):
            try:
                self.cp.initialize(initial_data=np.zeros(init_len, dtype=float))
            except TypeError:
                try:
                    self.cp.initialize()
                except Exception:
                    pass

        calib_roll_window = int(getattr(self.args, "calib_window", 200))
        calib_print_every = int(getattr(self.args, "calib_print_every", 200))

        resid_roll = deque(maxlen=calib_roll_window)
        control_roll = deque(maxlen=calib_roll_window)

        all_abs_errors: List[float] = []
        self.calib_alpha_history = []
        calib_y_true_all: List[float] = []

        val_y_true = cache_data["val_y_true"]
        val_y_pred = cache_data["val_y_pred"]
        val_x = cache_data["val_x"]
        val_step = cache_data["val_step"]

        for yt, yp, x_i, step_idx in zip(val_y_true, val_y_pred, val_x, val_step):
            unc = self._rolling_uncertainty()

            _, a_used, a_after = _unified_cp_step(
                self.cp,
                y_pred=float(yp),
                y_true=float(yt),
                uncertainty=float(unc),
                x=x_i,
                step=int(step_idx),
                horizon=None,
                update=True,
            )

            control_state = float(a_after) if np.isfinite(a_after) else float(a_used)
            self.calib_alpha_history.append(float(a_used))

            err = float(abs(float(yt) - float(yp)))
            self._err_hist.append(err)
            all_abs_errors.append(err)
            calib_y_true_all.append(float(yt))

            resid_roll.append(err)
            control_roll.append(control_state)

            if calib_print_every > 0 and (int(step_idx) % calib_print_every == 0):
                med, q90, iqr = _rolling_quantiles(list(resid_roll))
                c_step = _mean_abs_step(list(control_roll))
                c_std = float(np.std(np.asarray(control_roll, dtype=float))) if len(control_roll) > 1 else float("nan")
                print(
                    f"[Calib-Cache] step={step_idx} | "
                    f"resid_med={med:.4f} | resid_q90={q90:.4f} | resid_iqr={iqr:.4f} | "
                    f"control_step={c_step:.6f} | control_std={c_std:.6f}"
                )

        calib_mse = float(np.mean(np.square(all_abs_errors))) if len(all_abs_errors) > 0 else float("nan")
        self.calib_y_true_arr = np.array(calib_y_true_all, dtype=float)

        return calib_mse

    def evaluate_from_cache(
        self,
        cache_data,
        update: bool = True,
        *,
        setting: str,
        target_coverage: float,
        alpha_nominal: float,
        w_ref: Optional[float] = None,
    ):
        intervals: List[Tuple[float, float]] = []
        y_true_all: List[float] = []
        y_pred_all: List[float] = []

        self.test_interval_widths = []
        self.test_alpha_history = []
        self.test_alpha_after_update_history = []
        self.test_dynamics = []

        # 新增：坏点/极端点/全量点记录
        self.bad_points: List[Dict] = []
        self.extreme_width_points: List[Dict] = []
        self.all_test_points: List[Dict] = []

        test_window = int(getattr(self.args, "test_window", 100))
        test_print_every = int(getattr(self.args, "test_print_every", 100))
        dyn_stride = int(getattr(self.args, "dynamics_stride", 1))

        covered_roll = deque(maxlen=test_window)
        width_roll = deque(maxlen=test_window)
        control_roll = deque(maxlen=test_window)

        cp_used = self.cp if update else copy.deepcopy(self.cp)
        w_ref_used = float(w_ref) if (w_ref is not None and np.isfinite(w_ref)) else 1.0

        test_y_true = cache_data["test_y_true"]
        test_y_pred = cache_data["test_y_pred"]
        test_x = cache_data["test_x"]
        test_step = cache_data["test_step"]

        for yt, yp, x_i, t in zip(test_y_true, test_y_pred, test_x, test_step):
            unc = self._rolling_uncertainty()

            interval, a_used, a_after = _unified_cp_step(
                cp_used,
                y_pred=float(yp),
                y_true=float(yt),
                uncertainty=float(unc),
                x=x_i,
                step=int(t),
                horizon=None,
                update=update,
            )

            intervals.append(interval)
            y_true_all.append(float(yt))
            y_pred_all.append(float(yp))

            lo, hi = interval
            width_t = float(hi - lo)

            # ---------- 坏点检测 ----------
            bad_reasons = []

            if not np.isfinite(lo):
                bad_reasons.append("lo_nonfinite")
            if not np.isfinite(hi):
                bad_reasons.append("hi_nonfinite")
            if not np.isfinite(width_t):
                bad_reasons.append("width_nonfinite")
            if np.isfinite(lo) and np.isfinite(hi) and hi < lo:
                bad_reasons.append("hi_lt_lo")

            if not np.isfinite(a_used):
                bad_reasons.append("alpha_used_nonfinite")
            else:
                if a_used <= 0:
                    bad_reasons.append("alpha_used_le_0")
                if a_used >= 1:
                    bad_reasons.append("alpha_used_ge_1")

            if not np.isfinite(a_after):
                bad_reasons.append("alpha_after_nonfinite")
            else:
                if a_after <= 0:
                    bad_reasons.append("alpha_after_le_0")
                if a_after >= 1:
                    bad_reasons.append("alpha_after_ge_1")

            if not np.isfinite(unc):
                bad_reasons.append("unc_nonfinite")

            covered_t = 1.0 if (np.isfinite(lo) and np.isfinite(hi) and lo <= float(yt) <= hi) else 0.0

            self.test_interval_widths.append(width_t)
            self.test_alpha_history.append(float(a_used))
            self.test_alpha_after_update_history.append(float(a_after))

            point_row = {
                "t": int(t),
                "y_true": float(yt),
                "y_pred": float(yp),
                "lo": float(lo) if np.isfinite(lo) else lo,
                "hi": float(hi) if np.isfinite(hi) else hi,
                "width": float(width_t) if np.isfinite(width_t) else width_t,
                "covered_t": float(covered_t),
                "a_used": float(a_used) if np.isfinite(a_used) else a_used,
                "a_after": float(a_after) if np.isfinite(a_after) else a_after,
                "unc": float(unc) if np.isfinite(unc) else unc,
            }
            self.all_test_points.append(point_row)

            if len(bad_reasons) > 0:
                self.bad_points.append({
                    **point_row,
                    "reason": "|".join(bad_reasons),
                })

            covered_roll.append(covered_t)
            width_roll.append(width_t)

            control_state = float(a_after) if np.isfinite(a_after) else float(a_used)
            control_roll.append(control_state)

            t1 = int(t) + 1

            if len(covered_roll) == test_window:
                cov_w = float(np.mean(np.asarray(covered_roll, dtype=float)))
                width_mean_w = float(np.mean(np.asarray(width_roll, dtype=float)))

                ces_w = float(compute_ces(
                    coverage=cov_w,
                    target_coverage=float(target_coverage),
                    avg_width=width_mean_w,
                    w_ref=float(w_ref_used),
                    alpha=float(alpha_nominal),
                ))
                rcs_w = float(compute_rcs(
                    coverage=cov_w,
                    target_coverage=float(target_coverage),
                    avg_width=width_mean_w,
                    w_ref=float(w_ref_used),
                    alpha=float(alpha_nominal),
                ))
            else:
                cov_w = float("nan")
                width_mean_w = float("nan")
                ces_w = float("nan")
                rcs_w = float("nan")

            if test_print_every > 0 and (t1 % test_print_every == 0):
                gap_w = (cov_w - float(target_coverage)) if np.isfinite(cov_w) else float("nan")
                print(
                    f"[Test-Cache] t={t1} | "
                    f"cov@{test_window}={cov_w:.4f} (gap={gap_w:+.4f}) | "
                    f"width@{test_window}={width_mean_w:.4f} | "
                    f"CES@{test_window}={ces_w:.4f} | RCS@{test_window}={rcs_w:.4f}"
                )

            if dyn_stride <= 1 or (t1 % dyn_stride == 0):
                width_std_w = float(np.std(np.asarray(width_roll, dtype=float))) if len(width_roll) > 1 else float("nan")
                width_step_mean_w = _mean_abs_step(list(width_roll))
                control_std_w = float(np.std(np.asarray(control_roll, dtype=float))) if len(control_roll) > 1 else float("nan")
                control_step_mean_w = _mean_abs_step(list(control_roll))

                self.test_dynamics.append({
                    "t": t1,
                    "cov_w": cov_w,
                    "gap_w": (cov_w - float(target_coverage)) if np.isfinite(cov_w) else float("nan"),
                    "width_mean_w": width_mean_w,
                    "width_std_w": width_std_w,
                    "width_step_mean_w": width_step_mean_w,
                    "ces_w": ces_w,
                    "rcs_w": rcs_w,
                    "control_state": control_state,
                    "control_std_w": control_std_w,
                    "control_step_mean_w": control_step_mean_w,
                    "covered_t": covered_t,
                    "width_t": width_t,
                })

            err = float(abs(float(yt) - float(yp)))
            self._err_hist.append(err)

        y_true_arr = np.array(y_true_all, dtype=float)
        y_pred_arr = np.array(y_pred_all, dtype=float)

        coverage = compute_coverage(y_true_arr, intervals)
        avg_width = compute_average_width(intervals)
        mse = float(np.mean((y_true_arr - y_pred_arr) ** 2)) if len(y_true_arr) else float("nan")
        mae = float(np.mean(np.abs(y_true_arr - y_pred_arr))) if len(y_true_arr) else float("nan")

        w_ref_final = float(w_ref) if (w_ref is not None and np.isfinite(w_ref)) else float(compute_w_ref(y_true_arr, method="iqr"))

        final_ces = float(compute_ces(
            coverage=float(coverage),
            target_coverage=float(target_coverage),
            avg_width=float(avg_width),
            w_ref=float(w_ref_final),
            alpha=float(alpha_nominal),
        ))
        final_rcs = float(compute_rcs(
            coverage=float(coverage),
            target_coverage=float(target_coverage),
            avg_width=float(avg_width),
            w_ref=float(w_ref_final),
            alpha=float(alpha_nominal),
        ))

        final_gap = float(coverage) - float(target_coverage)
        print(
            f"[Test] FINAL | "
            f"cov={coverage:.4f} (gap={final_gap:+.4f}) | "
            f"width={avg_width:.4f} | "
            f"CES={final_ces:.4f} | "
            f"RCS={final_rcs:.4f}"
        )

        # ---------- bad/extreme summary ----------
        los = np.array([iv[0] for iv in intervals], dtype=float) if len(intervals) else np.asarray([], dtype=float)
        his = np.array([iv[1] for iv in intervals], dtype=float) if len(intervals) else np.asarray([], dtype=float)
        widths = his - los if len(intervals) else np.asarray([], dtype=float)

        finite_widths = widths[np.isfinite(widths)]
        if finite_widths.size > 0:
            med_w = float(np.median(finite_widths))
            q99_w = float(np.quantile(finite_widths, 0.99))
            extreme_thr = float(max(5.0 * med_w, q99_w))
        else:
            med_w = float("nan")
            q99_w = float("nan")
            extreme_thr = float("nan")

        if np.isfinite(extreme_thr):
            for row in self.all_test_points:
                w = row["width"]
                if np.isfinite(w) and w > extreme_thr:
                    self.extreme_width_points.append({
                        **row,
                        "reason": "width_extreme",
                        "extreme_threshold": extreme_thr,
                    })

        alpha_used_arr = np.array(self.test_alpha_history, dtype=float)
        alpha_after_arr = np.array(self.test_alpha_after_update_history, dtype=float)

        print("\n[Bad-point summary]")
        print("total points:", len(intervals))
        print("lo nonfinite:", int(np.sum(~np.isfinite(los))) if los.size else 0)
        print("hi nonfinite:", int(np.sum(~np.isfinite(his))) if his.size else 0)
        print("width nonfinite:", int(np.sum(~np.isfinite(widths))) if widths.size else 0)
        print("hi < lo:", int(np.sum((np.isfinite(los) & np.isfinite(his)) & (his < los))) if widths.size else 0)

        print("alpha_used nonfinite:", int(np.sum(~np.isfinite(alpha_used_arr))) if alpha_used_arr.size else 0)
        print("alpha_after nonfinite:", int(np.sum(~np.isfinite(alpha_after_arr))) if alpha_after_arr.size else 0)
        print("alpha_used <= 0:", int(np.sum(np.isfinite(alpha_used_arr) & (alpha_used_arr <= 0))) if alpha_used_arr.size else 0)
        print("alpha_used >= 1:", int(np.sum(np.isfinite(alpha_used_arr) & (alpha_used_arr >= 1))) if alpha_used_arr.size else 0)
        print("alpha_after <= 0:", int(np.sum(np.isfinite(alpha_after_arr) & (alpha_after_arr <= 0))) if alpha_after_arr.size else 0)
        print("alpha_after >= 1:", int(np.sum(np.isfinite(alpha_after_arr) & (alpha_after_arr >= 1))) if alpha_after_arr.size else 0)

        if finite_widths.size > 0:
            finite_width_mean = float(np.mean(finite_widths))
            finite_width_median = float(np.median(finite_widths))
            finite_width_max = float(np.max(finite_widths))
        else:
            finite_width_mean = float("nan")
            finite_width_median = float("nan")
            finite_width_max = float("nan")

        print("finite width mean:", finite_width_mean)
        print("finite width median:", finite_width_median)
        print("finite width max:", finite_width_max)
        print("extreme width threshold:", extreme_thr)
        print("num hard bad points:", len(self.bad_points))
        print("num extreme width points:", len(self.extreme_width_points))

        if len(self.bad_points) > 0:
            print("[First 10 hard bad points]")
            for row in self.bad_points[:10]:
                print(row)

        if len(self.extreme_width_points) > 0:
            print("[First 10 extreme width points]")
            for row in self.extreme_width_points[:10]:
                print(row)

        base_results_dir = getattr(self.args, "results_dir", "results")
        if hasattr(self.args, "result_tag") and self.args.result_tag == "ablation":
            dyn_dir = os.path.join(base_results_dir, "ablation", "dynamics")
        else:
            dyn_dir = os.path.join(base_results_dir, "dynamics")

        dyn_path = os.path.join(dyn_dir, f"{setting}.csv")
        _write_dynamics_csv(dyn_path, self.test_dynamics)
        print(f"[Dynamics] saved: {dyn_path}")

        # ---------- save bad/extreme/all points ----------
        bad_dir = os.path.join(base_results_dir, "bad_points")
        extreme_dir = os.path.join(base_results_dir, "extreme_width_points")
        all_points_dir = os.path.join(base_results_dir, "all_test_points")

        bad_path = os.path.join(bad_dir, f"{setting}.csv")
        extreme_path = os.path.join(extreme_dir, f"{setting}.csv")
        all_points_path = os.path.join(all_points_dir, f"{setting}.csv")

        if len(self.bad_points) > 0:
            _write_rows_csv(bad_path, self.bad_points)
            print(f"[Bad points] saved: {bad_path}")

        if len(self.extreme_width_points) > 0:
            _write_rows_csv(extreme_path, self.extreme_width_points)
            print(f"[Extreme width points] saved: {extreme_path}")

        if len(self.all_test_points) > 0:
            _write_rows_csv(all_points_path, self.all_test_points)
            print(f"[All test points] saved: {all_points_path}")

        base_model_name = getattr(self.args, "base_model", "cache")
        cp_mode = getattr(self.args, "cp_mode", "cp")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{setting}_{timestamp}"

        pi_path = os.path.join("v_results", "prediction_intervals", f"{prefix}_prediction_intervals.png")
        plot_prediction_intervals(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            intervals=intervals,
            save_path=pi_path,
            max_points=200,
        )

        alpha_curve = (
            self.test_alpha_after_update_history
            if len(self.test_alpha_after_update_history) > 0
            else self.test_alpha_history
        )
        alpha_path = os.path.join("v_results", "alpha_curves", f"{prefix}_alpha_control.png")
        plot_series(
            values=alpha_curve,
            save_path=alpha_path,
            title=f"Adaptive control signal on test set ({base_model_name}, {cp_mode})",
            ylabel="alpha_state",
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

        return coverage, avg_width, final_ces, final_rcs, w_ref_final, mse, mae, intervals, y_true_arr

    def run(self, setting: str = None):
        os.makedirs("results", exist_ok=True)
        os.makedirs("v_results", exist_ok=True)
        for sub in ["prediction_intervals", "alpha_curves", "interval_widths"]:
            os.makedirs(os.path.join("v_results", sub), exist_ok=True)
        os.makedirs(os.path.join(getattr(self.args, "results_dir", "results"), "dynamics"), exist_ok=True)

        conformal_csv = getattr(self.args, "conformal_csv_path", os.path.join("results", "conformal_results.csv"))
        adaptive_csv = getattr(self.args, "adaptive_csv_path", os.path.join("results", "adaptive_conformal_results.csv"))

        logger = ResultLogger(
            conformal_csv_path=conformal_csv,
            adaptive_csv_path=adaptive_csv,
        )

        cache_path = getattr(self.args, "cache_path", None)
        if cache_path is None:
            raise ValueError("args.cache_path is required in cache-only conformal mode.")

        cache_data = load_cache_for_conformal(
            cache_path=cache_path,
            x_lag=int(getattr(self.args, "x_lag", 24)),
        )

        _ = self.calibrate_from_cache(cache_data)

        if hasattr(self.cp, "start_test") and callable(getattr(self.cp, "start_test")):
            try:
                self.cp.start_test()
            except Exception:
                pass

        run_mode = getattr(self.args, "run_mode", "online")
        update_on_test = (run_mode == "online")

        cp_mode = getattr(self.args, "cp_mode", "cp")
        alpha_nominal = float(getattr(self.args, "alpha", 0.1))
        target_coverage = 1.0 - alpha_nominal

        if setting is None:
            dataset_name = os.path.basename(getattr(self.args, "data_path", "dataset"))
            base_model = getattr(self.args, "base_model", "cache")
            seed = getattr(self.args, "seed", "NA")
            setting = f"{dataset_name}_model{base_model}_cp{cp_mode}_mode{run_mode}_seed{seed}"

        w_ref_calib = None
        if hasattr(self, "calib_y_true_arr") and self.calib_y_true_arr is not None and len(self.calib_y_true_arr) > 0:
            w_ref_calib = float(compute_w_ref(self.calib_y_true_arr, method="iqr"))
            if w_ref_calib is None or (not np.isfinite(w_ref_calib)):
                w_ref_calib = 1.0

        coverage, avg_width, ces, rcs, w_ref, mse, mae, intervals, y_true_arr = self.evaluate_from_cache(
            cache_data,
            update=update_on_test,
            setting=setting,
            target_coverage=target_coverage,
            alpha_nominal=alpha_nominal,
            w_ref=w_ref_calib,
        )

        coverage_gap = float(abs(float(coverage) - float(target_coverage)))

        dataset = setting.split("_")[0] if "_" in setting else "unknown"

        logger.log_conformal(
    setting=setting,
    dataset=dataset,
    base_model=getattr(self.args, "base_model", "cache"),
    cp_mode=cp_mode,
    run_mode=run_mode,
    alpha=float(alpha_nominal),
    seed=int(getattr(self.args, "seed", 0)),
    x_lag=int(getattr(self.args, "x_lag", 24)),
    ablation_mode=str(getattr(self.args, "ablation_mode", "M0")),
    target_coverage=float(target_coverage),
    metrics={
        "coverage": float(coverage),
        "coverage_gap": float(coverage_gap),
        "avg_width": float(avg_width),
        "ces": float(ces),
        "rcs": float(rcs),
        "point_mse": float(mse),
        "point_mae": float(mae),
    },
    cache_path=str(getattr(self.args, "cache_path", "")),
    comment=getattr(self.args, "comment", ""),
)

        adaptive_modes = {"acp", "aci", "agaci", "cptc", "hopcpt"}
        control_alpha_series = (
            self.test_alpha_after_update_history
            if len(self.test_alpha_after_update_history)
            else self.test_alpha_history
        )
        is_adaptive = (cp_mode.lower() in adaptive_modes)

        if is_adaptive:
            width_series = self.test_interval_widths
            width_step_mean = float(compute_alpha_step_mean(width_series))
            width_std = float(compute_series_std(width_series))

            control_alpha_step_mean = float(compute_alpha_step_mean(control_alpha_series))
            control_alpha_std = float(compute_series_std(control_alpha_series))

            worst_window = int(getattr(self.args, "worst_window", 100))
            worst_cov = float(compute_worst_window_coverage(y_true_arr, intervals, window=worst_window))

            logger.log_adaptive(
                setting=setting,
                dataset=dataset,
                base_model=getattr(self.args, "base_model", "cache"),
                cp_mode=cp_mode,
                run_mode=run_mode,
                alpha=float(alpha_nominal),
                seed=int(getattr(self.args, "seed", 0)),
                x_lag=int(getattr(self.args, "x_lag", 24)),
                ablation_mode=str(getattr(self.args, "ablation_mode", "M0")),
                target_coverage=float(target_coverage),
                adaptive_metrics={
                "worst_window_coverage": worst_cov,
                "width_step_mean": width_step_mean,
                "width_std": width_std,
                "control_alpha_step_mean": control_alpha_step_mean,
                "control_alpha_std": control_alpha_std,
            },
            cache_path=str(getattr(self.args, "cache_path", "")),
            comment=getattr(self.args, "comment", ""),
        )

        excel_paths = logger.to_excel()

        print("✔ Results saved to:")
        print("   CSV :", conformal_csv)
        print("   CSV :", adaptive_csv)
        print("   XLSX:", excel_paths)