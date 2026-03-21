# exp/exp_ablation.py
from __future__ import annotations

import os
from copy import deepcopy

from .exp_conformal import ExpConformal

def apply_ablation_overrides(a):
    mode = getattr(a, "ablation_mode", "M0")

    # defaults = full model
    a.use_spectral = True
    a.use_regime = True
    a.use_cem = True
    a.width_weight = 1.0

    if mode == "M1":        # w/o spectral
        a.use_spectral = False
    elif mode == "M2":      # w/o regime
        a.use_regime = False
    elif mode == "M3":      # w/o adaptive alpha
        a.use_cem = False
    elif mode == "M4":      # w/o width penalty
        a.width_weight = 0.0


def run_ablation(args):
    """
    Run ACP ablations full; w/o spectral; w/o regime; w/o adaptive alpha; w/o width penalty
    """

    modes = ["M0", "M1", "M2", "M3", "M4"]

    base_cp_mode = getattr(args, "cp_mode", "acp").lower()
    if base_cp_mode != "acp":
        raise ValueError(f"[Ablation] This runner is for ACP only; got cp_mode={base_cp_mode}")

    base_comment = getattr(args, "comment", "")

    ablation_dir = os.path.join(getattr(args, "results_dir", "./results"), "ablation")
    os.makedirs(ablation_dir, exist_ok=True)

    for mode in modes:
        a = deepcopy(args)
        a.ablation_mode = mode
        a.comment = (base_comment + f" | ablation={mode}").strip()

        a.result_tag = "ablation"
        a.conformal_csv_path = os.path.join(ablation_dir, "ablation_conformal_results.csv")
        a.adaptive_csv_path  = os.path.join(ablation_dir, "ablation_adaptive_results.csv")

        a.des = f"{getattr(args,'des','ablation')}_{mode}"

        apply_ablation_overrides(a)
        exp = ExpConformal(a)

        dataset_name = os.path.basename(a.data_path)
        base_model = getattr(a, "base_model", "linear")
        run_mode = getattr(a, "run_mode", "online")
        seed = getattr(a, "seed", "NA")
        lags = getattr(exp.data_cfg, "lags", getattr(a, "lags", "NA"))

        setting = f"{dataset_name}_lags{lags}_model{base_model}_cp{a.cp_mode}_mode{run_mode}_seed{seed}_ABL{mode}"
        exp.run(setting=setting)

    print(f"[Ablation] Done. Check {ablation_dir}/*.csv and v_results/*.")