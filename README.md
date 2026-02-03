# CPRL

Adaptive Conformal Prediction via Frequency-Domain Wasserstein Calibration and Latent State Modeling

---

## 1. Overview

This repository implements multiple **Conformal Prediction (CP)** methods for **time series forecasting**, with a focus on:

- **Adaptive Conformal Prediction (ACP)**: Dynamically adjusts the coverage control signal over time.
- Frequency-domain Wasserstein calibration combined with latent state modeling.
- Support for various CP baselines: `standard`, `aci`, `agaci`, `nex`, `cqr`, `dfpi`, `enbpi`, `cptc`, `hopcpt`, etc.
- Flexibility to use both simple linear models and neural network models from `time_series_library` as the **base point-forecasting model**.

The main entry point is `run.py`. You can perform the full experimental pipeline via command-line arguments:
- Load a univariate time series from a CSV file.
- Construct lagged features and split into train / calibration / test sets.
- Train the base forecasting model.
- Calibrate and evaluate the CP method online, outputting numerical results and visualizations.

---

## 2. Environment Setup

### 2.1 Basic Dependencies

Python 3.9+ is recommended. Key dependencies include:

- `numpy`
- `pandas`
- `torch`
- `matplotlib`
- Other dependencies (if a `requirements.txt` is provided in the repo).

Install using pip:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, manually install the core packages:

```bash
pip install numpy pandas torch matplotlib
```

### 2.2 Optional: time_series_library

To use advanced deep learning models (instead of the default linear model) as the base forecaster:

- Install and ensure `time_series_library` is importable.
- Use the `--base_model` argument to specify a model name registered in the library.

If the library fails to import, the code gracefully degrades to support only the `linear` base model and prints:

> `[Warning] MODEL_REGISTRY unavailable; only 'linear' base model is usable.`

---

## 3. Data Format

The codebase currently assumes a **univariate time series** stored in a CSV file:

- Use `--data_path` to specify the CSV file path (required).
- By default, the **last column** is treated as the target variable.
- Use `--target_col` to explicitly specify a column name.

Example CSV (can contain multiple columns):

```text
timestamp,value
2020-01-01 00:00:00, 1.23
2020-01-01 01:00:00, 1.27
...
```

Notes:

- The target series is automatically normalized.
- Use `--lags` to control the window size: the series is transformed into supervised samples `(X, y)`, where `X` is a lagged window of length `lags` and `y` is the next-step target.
- Data is split **chronologically** into train / calibration / test (no shuffling).

---

## 4. Quick Start

### 4.1 Minimal Example

Run with the default linear base model and ACP mode with online updates:

```bash
python run.py \
  --data_path path/to/your_series.csv \
  --lags 96 \
  --train_ratio 0.6 \
  --calib_ratio 0.2 \
  --alpha 0.1 \
  --cp_mode acp \
  --run_mode online
```

Key arguments (commonly used):

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_path` | Path to CSV data file | (required) |
| `--target_col` | Target column name; uses last column if omitted | `None` |
| `--lags` | Lag window size (lookback) | `96` |
| `--train_ratio` | Proportion of data for training | `0.6` |
| `--calib_ratio` | Proportion of data for calibration | `0.2` |
| `--alpha` | Nominal significance level; target coverage = `1 - alpha` | `0.1` |
| `--cp_mode` | CP method: `acp`, `standard`, `aci`, `agaci`, `nex`, `cqr`, `dfpi`, `enbpi`, `cptc`, `hopcpt` | `acp` |
| `--run_mode` | `online` (update CP during test) or `eval` (no update, inference only) | `online` |
| `--results_dir` | Directory for numerical results | `./results` |

### 4.2 Choosing the Base Model

- Use the built-in linear model (default):

```bash
python run.py --data_path path/to/series.csv --base_model linear
```

- Use a registered model from `time_series_library` (e.g., `Autoformer`):

```bash
python run.py \
  --data_path path/to/series.csv \
  --base_model Autoformer
```

If the specified model is not in `MODEL_REGISTRY`, an error is raised showing available options.

### 4.3 Device Selection (CPU / CUDA / MPS)

| Argument | Description |
|----------|-------------|
| `--use_gpu` | Enable GPU usage (only effective when `--gpu_type cuda` and CUDA is available) |
| `--gpu` | GPU device index | `0` |
| `--gpu_type` | Backend: `cuda`, `mps`, or `cpu` | `cuda` |

Example: Use CUDA GPU 0:

```bash
python run.py \
  --data_path path/to/series.csv \
  --use_gpu \
  --gpu_type cuda \
  --gpu 0
```

On macOS with Apple Silicon, you can try MPS:

```bash
python run.py \
  --data_path path/to/series.csv \
  --gpu_type mps
```

---

## 5. Outputs and Visualizations

After execution, the following outputs are generated (paths may vary slightly based on the `setting` string):

### 5.1 Numerical Results

- `results/conformal_results.csv`
- `results/adaptive_conformal_results.csv`
- Corresponding Excel files (via `ResultLogger.to_excel()`) for convenient comparison across experiments.

### 5.2 Dynamics Logs

- `results/dynamics/<setting>.csv`  
  Records rolling-window metrics such as coverage, interval width, and control signal evolution over time.

### 5.3 Visualizations (under `v_results/`)

| Path | Description |
|------|-------------|
| `v_results/prediction_intervals/*.png` | Time-series plot of true values, point predictions, and prediction intervals. |
| `v_results/alpha_curves/*.png` | Adaptive control signal (alpha or equivalent) over the test set. |
| `v_results/interval_widths/*.png` | Prediction interval width over time. |

File names typically include: dataset name, base model, CP mode, run mode, random seed, and timestamp.

---

## 6. Code Structure

Key modules relevant to the main experimental workflow:

| File | Description |
|------|-------------|
| `run.py` | Command-line entry point: parses arguments, seeds, device selection, instantiates `ExpConformal`, and invokes `run()`. |
| `exp/exp/exp_basic.py` | `ExpBasic`: Base experiment class. Loads CSV, normalizes, constructs lagged features, chronological split, and builds `DataLoader`s. |
| `exp/exp/exp_conformal.py` | `ExpConformal(ExpBasic)`: Full pipeline. Defines base model (linear or neural), builds CP predictor via `build_conformal_predictor`, implements `train_model`, `calibrate`, and `evaluate`. Computes metrics (coverage, width, CES, RCS) and saves plots. |
| `src/utils.py` | Data preprocessing, metric computation (coverage, width, CES, RCS, worst-window coverage), and print utilities. |
| `src/base_conformal/` | Implementations and builders for various conformal predictors. |
| `src/result_logger.py` | Logging to CSV and Excel. |

---

## 7. Reproducing Experiments

Recommended workflow:

1. Prepare a univariate CSV time series (ensure sufficient length for `lags + train + calib + test` samples).
2. Choose an appropriate `--lags` (e.g., 48, 96, 168 for hourly data).
3. Set `--train_ratio` and `--calib_ratio` so that all splits have adequate samples.
4. Select `--cp_mode` and `--alpha` based on your desired CP method and target coverage.
5. Run `python run.py` with your chosen configuration.
6. Inspect results in `results/` (numerical) and `v_results/` (visualizations).

---

## 8. Citation

If you use this code in your research, please cite:

> Adaptive Conformal Prediction via Frequency-Domain Wasserstein Calibration and Latent State Modeling

*(Add BibTeX or official citation format here if available)*

---

## 9. License

*(Add license information here, e.g., MIT, Apache 2.0, etc.)*
