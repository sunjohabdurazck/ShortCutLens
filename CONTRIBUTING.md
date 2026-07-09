# Contributing to ShortcutLens

Thanks for considering a contribution. This project follows a lightweight,
standard open-source workflow.

## Setup

```bash
git clone https://github.com/<username>/shortcut-lens.git
cd shortcut-lens
pip install -e ".[xgboost,dev]"
```

## Before opening a PR

```bash
black shortcut_lens tests
flake8 shortcut_lens --max-line-length=100 --extend-ignore=E203,W503
pytest tests/ -v --cov=shortcut_lens
python experiments/reproduce_all.py --quick   # end-to-end smoke test
```

All four must pass locally -- they are exactly what CI (`.github/workflows/tests.yml`) runs.

## Adding a new shortcut type

1. Add an `inject_<name>` method to `ShortcutInjector` in `shortcut_lens/injectors.py`,
   following the existing docstring style (describe the real-world mechanism it simulates).
2. Add it to `ShortcutInjector.VALID_TYPES`.
3. Add unit tests to `tests/test_injectors.py` covering: shape correctness,
   monotonicity of correlation with `r`, and reproducibility under a fixed seed.
4. Add it to the `shortcut_types` list in the relevant `experiments/configs/*.yaml`.

## Adding a new dataset

1. Add a `load_<name>` function to `shortcut_lens/utils.py` returning
   `(X, y)` as NumPy arrays with `y` binarized to `{0, 1}`.
2. Register it in `DATASET_LOADERS`.
3. Add a config file under `experiments/configs/`.
4. Update the dataset table in `README.md`.

## Code style

- `black` formatting, 100-character line length.
- Type hints on public function signatures where practical.
- Every public function gets a NumPy-style docstring.

## Reporting issues

Please include: Python version, OS, the exact command you ran, and the full
traceback. If it's a numerical/statistical discrepancy, include the
`correlation_strength`, `shortcut_type`, and `random_state` used.
