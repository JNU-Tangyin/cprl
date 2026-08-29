#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ACP_ROOT = ROOT.parent
DEFAULT_OUT = ROOT / "time_series_library" / "dataset" / "new_benchmarks"
DEFAULT_MODELS = [
    "Autoformer",
    "Crossformer",
    "DLinear",
    "iTransformer",
    "PatchTST",
    "TimesNet",
    "Transformer",
    "TSMixer",
]


def tsl_setting_name(row: "DatasetRow", model: str, des: str, data: str = "custom") -> str:
    return (
        f"long_term_forecast_{row.dataset}_cache_{model}_{data}"
        f"_ft{row.features}_sl{row.seq_len}_ll{row.label_len}_pl{row.pred_len}"
        "_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1"
        f"_ebtimeF_dtTrue_{des}_0"
    )


@dataclass(frozen=True)
class DatasetRow:
    dataset: str
    family: str
    csv_path: Path
    data_path: str
    n_rows: int
    features: str
    target: str
    freq: str
    seq_len: int
    label_len: int
    pred_len: int
    enc_in: int
    dec_in: int
    c_out: int


def clean_numeric(values: Iterable[object]) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce")


def parse_fred_ts(path: Path) -> dict[str, tuple[pd.Timestamp, list[float]]]:
    series: dict[str, tuple[pd.Timestamp, list[float]]] = {}
    in_data = False
    with path.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if not in_data:
                if line.lower() == "@data":
                    in_data = True
                continue
            name, timestamp, values = line.split(":", 2)
            start = pd.to_datetime(timestamp.split()[0], errors="raise")
            y = clean_numeric(values.split(",")).dropna().astype(float).tolist()
            series[name] = (pd.Timestamp(start), y)
    return series


def write_bci(out_dir: Path, source_root: Path) -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    company_map = {
        "AMD": "bci_amd_vol",
        "Amazon": "bci_amazon_vol",
        "Nvidia": "bci_nvidia_vol",
    }
    out_subdir = out_dir / "bci_vol"
    out_subdir.mkdir(parents=True, exist_ok=True)

    for source_name, dataset in company_map.items():
        candidates = [
            source_root / "Variation of Local Fractional Coverage" / f"{source_name}-fc.csv",
            source_root / "data" / "vlfc" / f"{source_name}-fc.csv",
            source_root / "vlfc" / f"{source_name}-fc.csv",
        ]
        src = next((path for path in candidates if path.exists()), candidates[0])
        df = pd.read_csv(src)
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d"),
                "1e2Rt": clean_numeric(df["1e2Rt"]),
                "muhat": clean_numeric(df["muhat"]),
                "OT": clean_numeric(df["1e2Vt"]),
            }
        ).dropna()
        dest = out_subdir / f"{dataset}.csv"
        out.to_csv(dest, index=False)
        rows.append(
            DatasetRow(
                dataset=dataset,
                family="BCI-Vol",
                csv_path=dest,
                data_path=str(dest.relative_to(ROOT)),
                n_rows=len(out),
                features="MS",
                target="OT",
                freq="d",
                seq_len=96,
                label_len=48,
                pred_len=96,
                enc_in=3,
                dec_in=3,
                c_out=1,
            )
        )
    return rows


def write_fred(out_dir: Path, fred_path: Path, limit: int | None = None) -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    out_subdir = out_dir / "fred_md"
    out_subdir.mkdir(parents=True, exist_ok=True)

    for idx, (name, (start, values)) in enumerate(parse_fred_ts(fred_path).items(), start=1):
        if limit is not None and idx > limit:
            break
        dataset = f"fred_md_{name.lower()}"
        dates = pd.date_range(start=start.normalize(), periods=len(values), freq="MS")
        out = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "OT": values})
        dest = out_subdir / f"{dataset}.csv"
        out.to_csv(dest, index=False)
        rows.append(
            DatasetRow(
                dataset=dataset,
                family="FRED-MD",
                csv_path=dest,
                data_path=str(dest.relative_to(ROOT)),
                n_rows=len(out),
                features="S",
                target="OT",
                freq="m",
                seq_len=60,
                label_len=30,
                pred_len=12,
                enc_in=1,
                dec_in=1,
                c_out=1,
            )
        )
    return rows


def write_manifest(rows: list[DatasetRow], out_dir: Path) -> None:
    manifest = out_dir / "manifest.csv"
    fieldnames = list(DatasetRow.__dataclass_fields__.keys())
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            d = row.__dict__.copy()
            d["csv_path"] = str(row.csv_path)
            writer.writerow(d)

    lines = [
        "# New Dataset Forecast-Cache Manifest",
        "",
        "Upload `time_series_library/dataset/new_benchmarks/` to the same relative path on AutoDL.",
        "",
        "| dataset | family | rows | features | seq_len | label_len | pred_len | freq | data_path |",
        "|---|---|---:|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.dataset} | {row.family} | {row.n_rows} | {row.features} | "
            f"{row.seq_len} | {row.label_len} | {row.pred_len} | {row.freq} | {row.data_path} |"
        )
    (out_dir / "manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_autodl_commands(rows: list[DatasetRow], out_dir: Path, models: list[str], seeds: list[int]) -> None:
    lines = [
        "# AutoDL Forecast Cache Commands",
        "",
        "Run these commands from the `cprl/` directory so `forecast_cache_seed*/` is written",
        "under `cprl/`, where `scripts/run_cache_experiments.py` can discover it.",
        "",
        "This uses `scripts/run_tsl_forecast_cache.py`, which matches the existing cache",
        "protocol: train a time_series_library forecaster, export pred_len-step forecasts",
        "on val/test, and average overlapping horizon forecasts by absolute time index.",
        "",
    ]
    for row in rows:
        data_path = row.data_path
        for seed in seeds:
            for model in models:
                cmd = (
                    "python3 scripts/run_tsl_forecast_cache.py "
                    f"--model_id {row.dataset}_cache --model {model} --data custom "
                    f"--root_path {Path(data_path).parent}/ --data_path {Path(data_path).name} "
                    f"--features {row.features} --target {row.target} --freq {row.freq} "
                    f"--seq_len {row.seq_len} --label_len {row.label_len} --pred_len {row.pred_len} "
                    f"--enc_in {row.enc_in} --dec_in {row.dec_in} --c_out {row.c_out} "
                    f"--des {row.dataset}_{model}_seed{seed} "
                    f"--seed {seed} --train_epochs 20 --batch_size 32 "
                    "--use_gpu --gpu_type cuda"
                )
                lines.append(cmd)
    (out_dir / "autodl_forecast_commands.sh").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_remote_code_commands(rows: list[DatasetRow], out_dir: Path, models: list[str], seeds: list[int]) -> None:
    lines = [
        "# AutoDL Forecast Cache Commands for /root/Code_副本",
        "",
        "Run from `/root/Code_副本`.",
        "This targets the external legacy Time-Series-Library environment, where forecast-cache export is exposed through `run.py --export_forecast_cache`; the CPRL repository entry point remains `run_exp.py`.",
        "Prepared CSVs should be uploaded under `dataset/new_benchmarks/`.",
        "",
    ]
    for row in rows:
        remote_data_path = Path(row.data_path)
        rel_parts = remote_data_path.parts
        # Local prepared data lives under time_series_library/dataset/new_benchmarks;
        # remote Code_副本 uses dataset/ as its dataset root.
        try:
            idx = rel_parts.index("new_benchmarks")
            remote_rel = Path("dataset").joinpath(*rel_parts[idx:])
        except ValueError:
            remote_rel = remote_data_path
        root_path = remote_rel.parent
        data_file = remote_rel.name
        for seed in seeds:
            for model in models:
                des = row.dataset
                setting = tsl_setting_name(row, model=model, des=des)
                cmd = (
                    "/root/miniconda3/bin/python run.py "
                    "--task_name long_term_forecast --is_training 1 --export_forecast_cache "
                    f"--model_id {row.dataset}_cache --model {model} --data custom "
                    f"--root_path {root_path}/ --data_path {data_file} "
                    f"--features {row.features} --target {row.target} --freq {row.freq} "
                    f"--seq_len {row.seq_len} --label_len {row.label_len} --pred_len {row.pred_len} "
                    f"--enc_in {row.enc_in} --dec_in {row.dec_in} --c_out {row.c_out} "
                    f"--des {des} --seed {seed} "
                    f"--checkpoints /root/autodl-tmp/checkpoints_new/seed{seed}/ "
                    f"--train_epochs 20 --batch_size 32 --use_gpu True --gpu_type cuda "
                    f"--cache_save_path /root/autodl-tmp/forecast_cache_seed{seed}/{setting}/forecast_full.npz"
                )
                lines.append(cmd)
    (out_dir / "autodl_forecast_commands_remote_code.sh").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_remote_skip_manifest(rows: list[DatasetRow], out_dir: Path, models: list[str], seeds: list[int]) -> None:
    skip_paths: list[str] = []
    for row in rows:
        for seed in seeds:
            for model in models:
                setting = tsl_setting_name(row, model=model, des=row.dataset)
                local_cache = ROOT / f"forecast_cache_seed{seed}" / setting / "forecast_full.npz"
                if local_cache.exists():
                    skip_paths.append(
                        f"/root/autodl-tmp/forecast_cache_seed{seed}/{setting}/forecast_full.npz"
                    )

    skip_file = out_dir / "local_existing_cache_paths.txt"
    skip_file.write_text("\n".join(sorted(dict.fromkeys(skip_paths))) + ("\n" if skip_paths else ""), encoding="utf-8")


def write_remote_resume_wrapper(out_dir: Path) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -u",
        'src="dataset/new_benchmarks/autodl_forecast_commands_remote_code.sh"',
        'skip_file="dataset/new_benchmarks/local_existing_cache_paths.txt"',
        'while IFS= read -r cmd; do',
        '  if [[ "$cmd" != python* && "$cmd" != /root/miniconda3/bin/python* ]]; then',
        '    continue',
        '  fi',
        '  cache_path=$(printf "%s\\n" "$cmd" | sed -n "s/.*--cache_save_path \\([^ ]*\\).*/\\1/p")',
        '  if [ -n "$cache_path" ] && [ -f "$cache_path" ]; then',
        '    echo "[SKIP] $cache_path"',
        '    continue',
        '  fi',
        '  if [ -n "$cache_path" ] && [ -f "$skip_file" ] && grep -Fxq "$cache_path" "$skip_file"; then',
        '    echo "[SKIP_LOCAL] $cache_path"',
        '    continue',
        '  fi',
        '  echo "[RUN] $cache_path"',
        '  eval "$cmd"',
        '  status=$?',
        '  if [ $status -eq 0 ]; then',
        '    rm -rf /root/autodl-tmp/checkpoints_new',
        '    echo "[CLEAN] /root/autodl-tmp/checkpoints_new"',
        '  fi',
        'done < "$src"',
    ]
    (out_dir / "autodl_run_missing_remote.sh").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare BCI-Vol and FRED-MD CSVs for time_series_library.")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--bci_root", default=str(ACP_ROOT / "Datasets" / "BCI"))
    parser.add_argument("--fred_path", default=str(ACP_ROOT / "Datasets" / "fred_md" / "fred_md_dataset.ts"))
    parser.add_argument("--fred_limit", type=int, default=None, help="Limit FRED-MD indicators to the first N series")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    bci_root = Path(args.bci_root)
    fred_path = Path(args.fred_path)

    rows: list[DatasetRow] = []
    rows.extend(write_bci(out_dir, bci_root))
    rows.extend(write_fred(out_dir, fred_path, limit=args.fred_limit))
    rows = sorted(rows, key=lambda r: (r.family, r.dataset))

    write_manifest(rows, out_dir)
    write_autodl_commands(rows, out_dir, args.models, args.seeds)
    write_remote_code_commands(rows, out_dir, args.models, args.seeds)
    write_remote_skip_manifest(rows, out_dir, args.models, args.seeds)
    write_remote_resume_wrapper(out_dir)

    print(f"[Done] wrote {len(rows)} datasets under {out_dir}")
    print(f"[Done] manifest: {out_dir / 'manifest.csv'}")
    print(f"[Done] AutoDL commands: {out_dir / 'autodl_forecast_commands.sh'}")
    print(f"[Done] remote Code_副本 commands: {out_dir / 'autodl_forecast_commands_remote_code.sh'}")
    print(f"[Done] remote skip manifest: {out_dir / 'local_existing_cache_paths.txt'}")
    print(f"[Done] remote resume wrapper: {out_dir / 'autodl_run_missing_remote.sh'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
