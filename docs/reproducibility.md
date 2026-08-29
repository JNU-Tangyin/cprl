# Reproducibility Notes

This project separates base forecasting from conformal calibration.

1. Train or load a base forecaster with a fixed seed.
2. Export validation and test predictions to `forecast_full.npz`.
3. Reuse the same cache for every conformal method.
4. Compare conformal methods using the generated CSV summaries.

## Environment

The recommended baseline environment is Python 3.9 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For editable development:

```bash
pip install -e ".[dev]"
```

For optional forecasting backbones:

```bash
pip install -e ".[forecasting]"
```

## Deterministic Forecast Cache Protocol

Generate a cache with a fixed seed:

```bash
python scripts/run_tsl_forecast_cache.py \
  --model_id ETTh1_cache \
  --model DLinear \
  --data custom \
  --root_path time_series_library/dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --features MS \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --seed 2021
```

Run conformal calibration against the same cache:

```bash
python run_exp.py \
  --data_path time_series_library/dataset/ETT-small/ETTh1.csv \
  --cache_path forecast_cache_seed2021/<forecast-setting>/forecast_full.npz \
  --base_model DLinear \
  --cp_mode acp \
  --run_mode online \
  --lags 96 \
  --x_lag 96 \
  --alpha 0.1 \
  --seed 2011 \
  --results_dir results/ETTh1_DLinear_acp
```

Use the same `--cache_path`, `--lags`, `--x_lag`, `--alpha`, and data split settings when comparing conformal methods.

## Reporting Checklist

For each table or figure, record:

- dataset name and target column
- base forecaster, forecast seed, and cache path
- conformal method and CP seed
- `alpha`, `run_mode`, `lags`, `x_lag`, calibration window, and coverage window
- coverage, absolute coverage gap, average width, CES, RCS, and runtime

Generated caches, logs, and figures should be hosted outside GitHub if they are needed for exact reproduction. The benchmark CSV inputs committed under `time_series_library/dataset/` are sufficient to regenerate forecast caches.
