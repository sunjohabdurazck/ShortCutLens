# Setup Guide

A complete, from-scratch walkthrough for getting ShortcutLens running locally,
running experiments, generating figures, and building the paper. If you just
want the short version, see the [Quickstart](README.md#quickstart) in the
README — this document is the long version, with troubleshooting.

## 1. Clone the repository

```bash
git clone https://github.com/<your-username>/shortcut-lens.git
cd shortcut-lens
```

(If you received this project as a zip instead of a git remote, unzip it and
`cd shortcut-lens`, then optionally `git init` to start tracking it yourself.)

## 2. Create a virtual environment

```bash
python -m venv venv

# Activate it:
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

Requires Python 3.9+.

## 3. Install dependencies

```bash
# Editable install with XGBoost + dev tooling (pytest, black, flake8, jupyter)
pip install -e ".[xgboost,dev]"

# Or, a pinned reproducible environment instead of setup.py's ranges
pip install -r requirements.txt
```

## 4. Verify the directory structure

The repository already ships with the full structure below — this step is
only needed if you're reconstructing the project from scratch rather than
using the provided files:

```bash
mkdir -p experiments/configs experiments/results
mkdir -p paper/figures paper/tables
mkdir -p data notebooks tests
```

```
shortcut-lens/
├── shortcut_lens/          # installable package (injectors, pipeline, metrics, audit, visualizers, utils)
├── experiments/            # reproduce_all.py, generate_figures.py, configs/, results/
├── notebooks/               # EDA / results walkthroughs
├── paper/                   # main.tex, references.bib, figures/, tables/
├── data/                     # local dataset cache (gitignored)
├── tests/                    # unit + integration tests
├── setup.py / requirements.txt / LICENSE / README.md / SETUP.md (this file)
└── .github/workflows/tests.yml
```

## 5. Confirm the install worked

```bash
python -c "import shortcut_lens; print(shortcut_lens.__version__)"
# 0.1.0
```

## 6. Run the tests

```bash
pytest tests/ -v
```

All 20 tests should pass. This exercises `ShortcutInjector`, the CF Score /
statistical testing utilities, and a full end-to-end run of the four-condition
pipeline on synthetic data — so this step alone confirms the whole codebase is
wired together correctly, with no network access required.

## 7. Run a quick smoke test of the full pipeline

```bash
python experiments/reproduce_all.py --quick
```

This runs a small grid (2 datasets × 2 shortcut types × 3 correlation
strengths, 5-fold CV) and writes `experiments/results/all_results.json`.
Takes a couple of minutes. **This step requires internet access**, since it
downloads real datasets (UCI / OpenML) on first run and caches them under
`~/.shortcut_lens_cache`.

## 8. Run the full experimental grid

```bash
# Everything: 6 datasets x 5 shortcut types x 12 correlation strengths x ~9 classifiers x 5-fold CV
python experiments/reproduce_all.py
```

This is thousands of model fits and can take a few hours on a laptop CPU.
Useful variants:

```bash
# One dataset only, quick grid
python experiments/reproduce_all.py --datasets heart_disease --quick

# A specific subset of datasets, full grid
python experiments/reproduce_all.py --datasets heart_disease,mammography,madelon

# More/fewer CV folds
python experiments/reproduce_all.py --n-folds 10
```

## 9. Generate figures

```bash
python experiments/generate_figures.py
# or with explicit paths:
python experiments/generate_figures.py --results experiments/results/all_results.json --out-dir paper/figures
```

Produces CF-curve plots and phase-transition heatmaps as vector PDFs in
`paper/figures/`, ready to drop into the LaTeX paper.

## 10. Build the paper

```bash
cd paper
latexmk -pdf main.tex
# or, without latexmk:
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Expected output structure after a full run

```
shortcut-lens/
├── paper/
│   ├── main.pdf
│   └── figures/
│       ├── cf_curve_heart_disease_label_proxy.pdf
│       ├── cf_curve_adult_income_label_proxy.pdf
│       ├── heatmap_label_proxy.pdf
│       └── ...
├── experiments/
│   └── results/
│       └── all_results.json
└── data/                      # cached datasets (gitignored)
```

## Common issues and solutions

**`ModuleNotFoundError: No module named 'shortcut_lens'`**
```bash
pip install -e .
```
You likely skipped the editable install, or are running from outside a venv
that has it installed.

**`MemoryError` or the full grid is too slow**
```bash
python experiments/reproduce_all.py --datasets heart_disease --quick
```
Narrow to fewer datasets and/or use `--quick` first; scale up once you've
confirmed the pipeline behaves as expected.

**XGBoost fails to install**
```bash
pip install xgboost==2.0.3
```
XGBoost is an optional extra (`pip install -e ".[xgboost]"`); the package
degrades gracefully and simply omits it from `get_model_suite()` if it isn't
installed (see `shortcut_lens/utils.py`).

**OpenML datasets (Adult Income, MADELON, the German Credit fallback) fail to download**
```bash
python -c "from sklearn.datasets import fetch_openml; fetch_openml(name='adult', version=2)"
```
Run that in isolation first to surface the underlying network error. OpenML
can be slow/flaky under load — retry, or increase your HTTP client's timeout.

**UCI downloads (Heart Disease, Credit Default, Mammographic Mass) return 403 or fail**
Some restricted or sandboxed network environments (e.g. CI runners with an
egress allowlist) block `archive.ics.uci.edu` outright. This is expected in
such environments — the loaders work normally with regular internet access.
If you hit this outside a sandbox, check for a firewall/proxy blocking the
domain, or download the `.data`/`.xls` file manually and place it at
`~/.shortcut_lens_cache/<dataset_name>.csv`.

**No local Bangla/regional dataset configured**
`load_bangla_dataset()` automatically falls back to the UCI Statlog German
Credit dataset (via OpenML) so the pipeline is runnable end-to-end without
extra setup. To use your own dataset, drop a CSV with the label in the last
column at `data/bangla_dataset.csv`.

**Matplotlib figures not rendering in headless environments (WSL, SSH, CI)**
```python
import matplotlib
matplotlib.use("Agg")
```
`shortcut_lens/visualizers.py` only ever saves figures to disk (`plt.savefig`)
and never calls `plt.show()`, so this is rarely needed — but set the backend
explicitly if your environment's default backend errors on import.

## Next steps

- Replace every `<username>` / `<your-username>` placeholder in `README.md`,
  `setup.py`, and `.github/workflows/tests.yml` with your actual GitHub handle.
- After a full run, fill in the real numbers in the README's
  [Results](README.md#results-summary) table and the paper's abstract.
- See [CONTRIBUTING.md](CONTRIBUTING.md) for the checks CI runs on every PR.
