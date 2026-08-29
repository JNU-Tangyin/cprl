# Benchmark Data

This repository does not track raw benchmark datasets, preprocessed benchmark CSVs, or forecast caches. Download the datasets from their public sources and place them under the paths below before training the base forecasters.

## Standard Forecasting Benchmarks

ETT, Weather, and Exchange Rate are taken from the public Time-Series-Library benchmark collection.

| Dataset | Source | Expected local path |
|---|---|---|
| ETTh1, ETTh2, ETTm1, ETTm2 | THUML Time-Series-Library / TSLib datasets | `time_series_library/dataset/ETT-small/*.csv` |
| Weather | THUML Time-Series-Library / TSLib datasets | `time_series_library/dataset/weather/weather.csv` |
| Exchange Rate | THUML Time-Series-Library / TSLib datasets | `dataset/exchange_rate/exchange_rate.csv` |

Useful links:

- Time-Series-Library GitHub: <https://github.com/thuml/Time-Series-Library>
- Time-Series-Library dataset mirror: <https://huggingface.co/datasets/thuml/Time-Series-Library>

## FRED-MD

FRED-MD is sourced from the FRED-MD database maintained by McCracken and Ng through the Federal Reserve Bank of St. Louis. The official source is not GitHub, so papers should cite the St. Louis Fed database page and the original FRED-MD reference. A GitHub-hosted option is the `nk027/bvar` repository, whose R package includes the FRED-MD and FRED-QD datasets.

Expected source location before preprocessing:

```text
../Datasets/fred_md/fred_md_dataset.ts
```

Expected project-local files after preprocessing:

```text
time_series_library/dataset/new_benchmarks/fred_md/fred_md_t1.csv
time_series_library/dataset/new_benchmarks/fred_md/fred_md_t2.csv
...
time_series_library/dataset/new_benchmarks/fred_md/fred_md_t107.csv
```

Each FRED-MD target is converted into a single-target monthly forecasting file with columns:

```text
date,OT
```

Useful link:

- FRED-MD official database page: <https://research.stlouisfed.org/econ/mccracken/fred-databases/>
- GitHub-hosted R package with FRED-MD data: <https://github.com/nk027/bvar>

## BCI Equity Volatility

BCI-Vol uses the AMD, Amazon, and Nvidia CSV files distributed with the Bellman Conformal Inference repository. In this project, the `vlfc` files are converted into the Time-Series-Library CSV format.

In the BCI repository, the source files are located under:

```text
data/vlfc/AMD-fc.csv
data/vlfc/Amazon-fc.csv
data/vlfc/Nvidia-fc.csv
```

Expected source files before preprocessing:

```text
../Datasets/BCI/Variation of Local Fractional Coverage/AMD-fc.csv
../Datasets/BCI/Variation of Local Fractional Coverage/Amazon-fc.csv
../Datasets/BCI/Variation of Local Fractional Coverage/Nvidia-fc.csv
```

Expected project-local files after preprocessing:

```text
time_series_library/dataset/new_benchmarks/bci_vol/bci_amd_vol.csv
time_series_library/dataset/new_benchmarks/bci_vol/bci_amazon_vol.csv
time_series_library/dataset/new_benchmarks/bci_vol/bci_nvidia_vol.csv
```

The converted files use:

```text
date,1e2Rt,muhat,OT
```

where `OT` is the target volatility series used by the conformal experiments.

Useful link:

- Bellman Conformal Inference GitHub: <https://github.com/ZitongYang/bellman-conformal-inference>

## Preprocessing

After placing the raw FRED-MD and BCI files in the source locations above, run:

```bash
python scripts/prepare_new_datasets.py
```

If the BCI repository is cloned locally, the preprocessing script can read its native layout directly:

```bash
python scripts/prepare_new_datasets.py \
  --bci_root path/to/bellman-conformal-inference
```

This writes the project-local benchmark files under:

```text
time_series_library/dataset/new_benchmarks/
```

The generated CSVs and forecast caches are ignored by Git. For the standard experimental protocol, train each base forecaster first and export a deterministic `forecast_full.npz`; then run the conformal methods on the same cache.
