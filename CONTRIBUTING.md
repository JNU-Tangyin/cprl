# Contributing

CPRL is a research codebase. Contributions should preserve reproducibility, keep experimental artifacts out of Git, and document changes that affect reported metrics.

## Development Setup

Create an isolated Python environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

Optional forecasting backbones may require extra dependencies:

```bash
pip install -e ".[forecasting]"
```

## Before Opening a Pull Request

Run a lightweight syntax check:

```bash
PYTHONPYCACHEPREFIX=/tmp/cprl_pycache python -m compileall run_exp.py exp src scripts utils time_series_library
```

For changes that affect conformal methods, also run at least one small deterministic experiment and record:

- dataset and local data path
- base forecaster or forecast cache path
- `cp_mode`, `run_mode`, `alpha`, `lags`, and seed
- coverage, average width, and runtime

## Repository Hygiene

Do not commit generated artifacts:

- `results/`, `v_results/`, `all_results/`, `analysis_figures/`, `experiment_logs/`
- `forecast_cache_seed*/`
- raw or preprocessed benchmark datasets
- `.npz`, `.npy`, `.pkl`, `.joblib`, `.xlsx`, and compressed archives

If a generated file was already tracked by Git, `.gitignore` is not enough. Remove it from the index with `git rm --cached <path>` while keeping the local copy if needed.

## Documentation

Update `README.md` or `docs/` whenever you change:

- command-line arguments or defaults
- forecast cache format
- dataset locations or preprocessing steps
- result metrics or output file names
