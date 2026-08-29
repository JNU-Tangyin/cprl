# GitHub Release Checklist

Use this checklist before pushing CPRL as a public research repository.

## 1. Inspect the Worktree

```bash
git status --short
git diff --stat
```

Confirm that source-code changes are intentional and that generated files are not being added.

## 2. Remove Generated Files From the Git Index

The project `.gitignore` excludes generated artifacts, but files already tracked by Git remain tracked. Remove tracked artifacts from the index while keeping local copies:

```bash
git rm --cached -r all_results results v_results analysis_figures experiment_logs
git rm --cached -r dataset time_series_library/dataset
git rm --cached .DS_Store
git rm --cached '*.xlsx' '*.xls' '*.npz' '*.npy' '*.pkl' '*.joblib' '*.tgz' '*.tar.gz' '*.zip'
```

If a path is already deleted locally, `git rm --cached` may report that it is missing. In that case, stage the deletion with `git add -u`.

## 3. Keep Public Repository Contents Focused

Commit these categories:

- source code under `src/`, `exp/`, `scripts/`, `utils/`, `data_provider/`, and `time_series_library/`
- lightweight configuration such as `.gitignore`, `.gitattributes`, `requirements.txt`, and `pyproject.toml`
- public documentation under `README.md`, `docs/`, `CONTRIBUTING.md`, `CITATION.cff`, and `LICENSE`

Do not commit local datasets, forecast caches, results, notebook checkpoints, local logs, archives, or machine-specific files.

## 4. Run a Lightweight Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/cprl_pycache python -m compileall run_exp.py exp src scripts utils time_series_library
```

If dependencies are installed and a small local CSV is available, also run:

```bash
python run_exp.py \
  --data_path path/to/your_series.csv \
  --base_model linear \
  --cp_mode acp \
  --run_mode online \
  --results_dir results/smoke_test
```

## 5. Commit and Push

```bash
git add README.md docs CONTRIBUTING.md CITATION.cff pyproject.toml requirements.txt .github .gitignore .gitattributes
git add -u
git commit -m "Prepare CPRL research repository"
git push origin main
```

Adjust the branch name if the repository uses `master` or another default branch.

## 6. Configure GitHub Repository Metadata

After pushing:

- add a concise repository description
- add topics such as `conformal-prediction`, `time-series`, `uncertainty-quantification`, and `forecasting`
- enable Issues if external users should report bugs
- create a release or tag after the paper version is frozen
- archive exact datasets, caches, and result tables externally, then link them from `README.md`
