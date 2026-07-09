# Notebooks

Interactive companions to the `shortcut_lens` package and `experiments/` scripts.
Run `jupyter lab` from the repo root after `pip install -e ".[dev]"`.

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | EDA for all six benchmark datasets (class balance, feature distributions, missingness) |
| `02_shortcut_visualization.ipynb` | Visualizes what each of the 5 injected shortcut types looks like at varying `r` |
| `03_main_results.ipynb` | Loads `experiments/results/all_results.json` and reproduces every paper figure interactively |
| `04_statistical_tests.ipynb` | Runs and inspects all pairwise Wilcoxon + Benjamini-Hochberg significance tests |

These notebooks are intentionally not committed with pre-run output cells (see `.gitignore` for `.ipynb_checkpoints/`) -- regenerate them locally after running `experiments/reproduce_all.py`.
