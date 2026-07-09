<div align="center">

# 🔬 ShortcutLens

ShortcutLens is an open-source Python framework for measuring how much machine learning models rely on spurious correlations before deployment. By simulating distribution shift and controlled shortcut injection, it helps researchers and practitioners identify models that achieve high validation accuracy for the wrong reasons.

[![tests](https://github.com/InsightForge-ML/shortcut-lens/actions/workflows/tests.yml/badge.svg)](https://github.com/InsightForge-ML/shortcut-lens/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage](https://img.shields.io/badge/coverage-tracked-brightgreen.svg)](tests/)

</div>

> Your model hits 94% validation accuracy. In production it drops to 71%. Standard metrics never see it coming — because the model isn't learning your problem, it's learning a shortcut that happened to be lying around in the training distribution.
>
> **ShortcutLens estimates that risk before you deploy.**

---

## 🤔 Why ShortcutLens?

Traditional evaluation tells you **how accurate** a model is.
ShortcutLens helps you understand **why** it's accurate.

A model that achieves 95% validation accuracy may still degrade sharply in production if it learned a spurious, non-causal shortcut rather than the underlying signal. ShortcutLens surfaces that hidden risk before deployment through controlled shortcut injection, out-of-distribution testing, and a single interpretable score — the **Causal Fidelity (CF) Score**.

---

## ✨ Features

- 🔬 Estimates the risk of shortcut-dependent performance degradation under simulated distribution shift
- 📊 Computes the novel **Causal Fidelity (CF) Score**, backed by bootstrap confidence intervals
- ⚡ Supports **5 shortcut injection mechanisms** (label proxy, demographic proxy, temporal, selection bias, measurement artifact)
- 🤖 Benchmarks **8+ classical and ensemble ML models** out of the box
- 📈 Statistical significance testing (Wilcoxon signed-rank, Benjamini–Hochberg corrected)
- 📉 Feature-importance auditing to identify *which* feature a model is relying on
- 🐳 Docker & GitHub Actions CI support
- 📦 Pip-installable Python package with a clean, documented API

---

## 📖 Table of Contents

- [🤔 Why ShortcutLens?](#-why-shortcutlens)
- [✨ Features](#-features)
- [📖 Overview](#-overview)
- [🧭 How It Works](#-how-it-works)
- [🎯 Key Findings](#-key-findings)
- [🧮 The Causal Fidelity Score](#-the-causal-fidelity-score)
- [⚡ Quickstart](#-quickstart)
- [📦 Installation](#-installation)
- [💻 Usage](#-usage)
- [🧪 The Experimental Protocol](#-the-experimental-protocol)
- [🧬 Shortcut Types](#-shortcut-types)
- [📊 Datasets](#-datasets)
- [🤖 Benchmarked Models](#-benchmarked-models)
- [🗂️ Repository Structure](#️-repository-structure)
- [🔁 Reproducing All Results](#-reproducing-all-results)
- [✅ Testing & CI](#-testing--ci)
- [📈 Results Summary](#-results-summary)
- [📄 Paper](#-paper)
- [📝 Citing This Work](#-citing-this-work)
- [🤝 Contributing](#-contributing)
- [⚖️ License](#️-license)
- [👥 Authors](#-authors)

---

## 📖 Overview

Machine learning models deployed in real-world settings frequently encounter **distribution shift**, where statistical patterns present at training time are absent or altered at inference time. Shortcut learning — a model's tendency to latch onto spurious, non-causal correlations rather than genuine signal — has been studied extensively in deep learning and computer vision, but its prevalence across **classical tabular ML classifiers** remains comparatively under-examined, despite tabular models being a common choice in medical, financial, and public-sector deployments.

**ShortcutLens** is a pip-installable Python toolkit built to help answer one question for any tabular classifier: *is it learning causally meaningful features, or has it latched onto a spurious correlation that may not hold up under distribution shift?*

It does this by:

1. **Injecting controlled synthetic shortcuts** into real tabular datasets, at 12 correlation strengths, across 5 distinct shortcut mechanisms (label proxies, demographic proxies, temporal drift, selection bias, measurement artifacts).
2. **Evaluating every classifier under four experimental conditions** (clean training, shortcut training tested in-distribution, shortcut training tested out-of-distribution, and shortcut-only performance) to help isolate how much of a model's accuracy may be shortcut-dependent.
3. **Scoring the result with the Causal Fidelity Score (CF)** — a single, interpretable, bootstrap-confidence-interval-backed number computable pre-deployment.
4. **Auditing feature importances** to show *which* feature a model is actually relying on, not just *that* it degrades.
5. **Backing every comparative claim statistically** — mean ± std across stratified 5-fold CV, with Wilcoxon signed-rank tests (Benjamini–Hochberg corrected).

No synthetic toy datasets (MNIST, Iris, Titanic) are used anywhere in this project — every dataset is a real-world, clinically or financially consequential tabular problem.

---

## 🧭 How It Works

```
   Dataset
      │
      ▼
Shortcut Injection   (5 mechanisms × 12 correlation strengths)
      │
      ▼
   Classifier          (8+ models: linear, tree-based, ensemble, neural)
      │
      ▼
4-Condition Evaluation (A: clean · B: in-distribution · C: OOD · D: shortcut-only)
      │
      ▼
  CF Score             (bootstrap CI + significance testing)
      │
      ▼
 Visualizations         (CF curves, phase-transition heatmaps, importance plots)
```

Each stage is a separate, composable module (`injectors.py`, `pipeline.py`, `metrics.py`, `audit.py`, `visualizers.py`), so any stage can be swapped, extended, or run independently — see [Usage](#-usage).

> 📷 **Sample outputs:** rendered CF curves, phase-transition heatmaps, and feature-importance plots are generated under `paper/figures/` after running `experiments/generate_figures.py`, and can be embedded here once available.

---

## 🎯 Key Findings

- 🎯 **Phase transitions** in shortcut reliance can occur at correlation strengths as low as r ≈ 0.3.
- 🏆 **Linear models** (Logistic Regression, SVM-RBF) tend to be more robust to injected shortcuts than ensembles.
- ⚠️ **Ensemble methods** (XGBoost, Random Forest, Gradient Boosting) show the highest susceptibility to shortcut adoption.
- 📊 **Class-imbalanced datasets** amplify shortcut vulnerability.
- 🔴 **Demographic-proxy shortcuts** tend to be more damaging than simple label-proxy shortcuts.

> Exact figures depend on your experimental run — see [Results Summary](#-results-summary) for the reporting format and [Reproducing All Results](#-reproducing-all-results) to generate your own numbers.

---

## 🧮 The Causal Fidelity Score

```
CF = 1 - SRS / Acc(B)          where SRS = Acc(B) - Acc(C)
```

| Condition | Meaning |
|---|---|
| **A** | Train clean → test clean (true baseline capability) |
| **B** | Train with shortcut → test with shortcut (in-distribution, possibly inflated) |
| **C** | Train with shortcut → test **out-of-distribution** (shortcut column replaced with noise) — **the simulated deployment-shift condition** |
| **D** | Train with shortcut → test with real features zeroed out (pure shortcut reliance) |

**CF = 1.0** → the model appears to ignore the shortcut; performance looks fully causal.
**CF = 0.0** → the model's performance appears entirely shortcut-dependent; a sharp production drop would be expected under similar shift.
**CF < 0.0** → the shortcut appears to have actively hurt generalization (rare, but diagnostically interesting).

```python
from shortcut_lens import causal_fidelity_score

result = causal_fidelity_score(
    condition_A_acc=0.84,
    condition_B_acc=0.94,
    condition_C_acc=0.71,
    fold_results_B=[0.93, 0.95, 0.94, 0.94, 0.95],
    fold_results_C=[0.70, 0.72, 0.71, 0.70, 0.72],
)
# {'cf_score': 0.245, 'srs': 0.23, 'ci_lower': 0.19, 'ci_upper': 0.31, ...}
```

---

## Quickstart

```bash
git clone https://github.com/InsightForge-ML/shortcut-lens.git
cd shortcut-lens
pip install -e .

python - <<'PY'
from shortcut_lens import ShortcutInjector, run_full_experiment, causal_fidelity_score
from sklearn.ensemble import RandomForestClassifier
from shortcut_lens.utils import load_dataset

X, y = load_dataset("heart_disease")
injector = ShortcutInjector(correlation_strength=0.9)
model = RandomForestClassifier(n_estimators=100, random_state=42)

result = run_full_experiment(X, y, model, injector, shortcut_type="label_proxy")
cf = causal_fidelity_score(
    result["condition_A"]["accuracy"]["mean"],
    result["condition_B"]["accuracy"]["mean"],
    result["condition_C"]["accuracy"]["mean"],
)
print(f"Causal Fidelity Score: {cf['cf_score']:.3f}")
PY
```

---

## Installation

> For a longer, hand-held walkthrough (including troubleshooting), see [SETUP.md](SETUP.md).

**Requirements:** Python 3.9+, Git, and (optionally) LaTeX for compiling the paper.

### Editable install (recommended for development)

```bash
git clone https://github.com/InsightForge-ML/shortcut-lens.git
cd shortcut-lens

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -e ".[xgboost,dev]"
```

### Pinned reproducible environment

```bash
pip install -r requirements.txt
```

### Docker (optional)

```bash
docker build -t shortcut-lens .
docker run -it --rm shortcut-lens
```

<details>
<summary>Core dependencies</summary>

```
numpy>=1.23
pandas>=1.5
scikit-learn>=1.2
scipy>=1.9
xgboost>=1.7
matplotlib>=3.6
seaborn>=0.12
statsmodels>=0.13
```

</details>

<details>
<summary>Development dependencies</summary>

```
pytest>=7.0
pytest-cov>=4.0
black>=22.0
flake8>=5.0
mypy>=0.990
```

</details>

---

## Usage

### 1. Inject a shortcut

```python
from shortcut_lens import ShortcutInjector

injector = ShortcutInjector(correlation_strength=0.7, random_state=42)
X_shortcut = injector.inject("demographic_proxy", X_train, y_train)   # adds 1 column
X_ood = injector.remove_shortcut(X_shortcut)                          # simulate deployment
```

### 2. Run the four-condition protocol

```python
from shortcut_lens import run_full_experiment

result = run_full_experiment(X, y, model, injector, shortcut_type="label_proxy", n_folds=5)
# result = {'condition_A': {...}, 'condition_B': {...}, 'condition_C': {...}, 'condition_D': {...}}
```

### 3. Score it

```python
from shortcut_lens import causal_fidelity_score

cf = causal_fidelity_score(
    result["condition_A"]["accuracy"]["mean"],
    result["condition_B"]["accuracy"]["mean"],
    result["condition_C"]["accuracy"]["mean"],
)
```

### 4. Audit which feature the model is trusting

```python
from shortcut_lens import audit_shortcut_reliance

audit = audit_shortcut_reliance(trained_model, "random_forest", feature_names=col_names)
print(audit["dominance_ratio"])  # > 1.0 means the shortcut outweighs real features
```

### 5. Compare models statistically

```python
from shortcut_lens import compare_models_statistically

df = compare_models_statistically(
    {"svm_rbf": {"cf_score": [...]}, "random_forest": {"cf_score": [...]}},
    metric="cf_score",
)
```

### Command-line interface

```bash
# Full grid: 6 datasets × 5 shortcuts × 12 strengths × 9 classifiers × 5-fold CV
python experiments/reproduce_all.py

# Fast smoke-test grid (what CI runs)
python experiments/reproduce_all.py --quick

# Subset of datasets
python experiments/reproduce_all.py --datasets heart_disease,mammography

# Regenerate every figure from a results file
python experiments/generate_figures.py --results experiments/results/all_results.json
```

---

## The Experimental Protocol

Every `(dataset, shortcut_type, correlation_strength, model)` combination is evaluated with stratified 5-fold cross-validation across all four conditions above. The gap between **Condition B** and **Condition C** is the deployment disaster the CF Score is designed to catch *before* it happens in production.

---

## Shortcut Types

| # | Type | Simulates |
|---|---|---|
| 1 | `label_proxy` | A data-collection artifact that leaks the outcome (e.g. patient-ID ranges correlated with diagnosis at one hospital) |
| 2 | `demographic_proxy` | An existing feature (zip code, occupation) repurposed as a protected-attribute proxy |
| 3 | `temporal_shortcut` | A pattern valid early in the data stream that decays over time (seasonality, policy change, sensor drift) |
| 4 | `selection_bias` | Non-random sampling in training data (e.g. pooled from specific hospitals/regions) |
| 5 | `measurement_artifact` | Systematic bias from the collection instrument, correlated with the label only within certain batches |

Every experiment is run across **12 correlation strengths** (`0.0 → 0.99`) to precisely locate each model's *phase transition* — the point at which it starts trading real signal for the shortcut.

---

## Datasets

Six real-world tabular datasets, each chosen to stress a different robustness dimension.

| Dataset | Samples | Features | Domain | Robustness Dimension |
|---|---|---|---|---|
| [Heart Disease (UCI)](https://archive.ics.uci.edu/dataset/45/heart+disease) | 303 | 13 | Medical | Clinically meaningful features, dangerous shortcuts |
| [Adult Income (UCI/OpenML)](https://archive.ics.uci.edu/dataset/2/adult) | 48,842 | 14 | Fairness | Demographic proxy variables |
| [Credit Default (UCI)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) | 30,000 | 23 | Finance | Real deployment stakes, temporal shortcuts |
| [Mammographic Mass (UCI)](https://archive.ics.uci.edu/dataset/161/mammographic+mass) | 961 | 5 | Medical (imbalanced) | 97% negative class — shortcut amplification under imbalance |
| [MADELON (UCI)](https://archive.ics.uci.edu/dataset/171/madelon) | 2,600 | 500 (420 pure noise) | Synthetic/high-dim | Ground-truth irrelevant features |
| Local/regional dataset | — | — | Bangladesh context | Regional relevance; falls back to UCI Statlog German Credit if unconfigured |

`shortcut_lens/utils.py::load_dataset(name)` handles fetching and caching for all six.

---

## Models Evaluated

| Family | Models |
|---|---|
| Linear | Logistic Regression, SVM (RBF kernel) |
| Instance-based | k-Nearest Neighbors |
| Tree-based | Decision Tree, Random Forest, Gradient Boosting |
| Neural | Multi-Layer Perceptron |
| Ensemble | XGBoost |

`shortcut_lens/utils.py` exposes a model-suite registry, so the classifier set is easily extensible.

---

## Repository Structure

```
shortcut-lens/
├── README.md                    # This file
├── SETUP.md                     # Extended installation & troubleshooting guide
├── setup.py / pyproject.toml    # pip-installable package
├── requirements.txt             # Pinned dependencies
├── LICENSE                      # MIT
├── Dockerfile                   # Container build
├── .github/workflows/
│   ├── tests.yml                 # CI: lint + unit tests + smoke-test pipeline
│   └── publish.yml               # PyPI publishing
│
├── shortcut_lens/                # The installable package
│   ├── __init__.py
│   ├── injectors.py              # ShortcutInjector — 5 shortcut mechanisms
│   ├── models.py                 # Model-suite registry
│   ├── pipeline.py               # run_full_experiment() — the 4-condition protocol
│   ├── metrics.py                # causal_fidelity_score(), compare_models_statistically()
│   ├── audit.py                  # audit_shortcut_reliance() — feature-importance audit
│   ├── visualizers.py            # CF curves, phase-transition heatmaps, importance plots
│   └── utils.py                  # Dataset loaders + model-suite registry
│
├── experiments/
│   ├── reproduce_all.py          # ONE COMMAND: runs the full experimental grid
│   ├── generate_figures.py       # Regenerates every paper figure from results JSON
│   ├── configs/                  # One YAML per dataset (shortcut types, r-grid, folds)
│   └── results/                  # Auto-generated, gitignored raw results
│
├── notebooks/                    # EDA, shortcut visualization, results walkthrough
├── paper/                        # LaTeX source, figures, tables, references.bib
├── data/                         # Local dataset cache (gitignored)
└── tests/
    ├── test_injectors.py         # Unit tests for ShortcutInjector
    ├── test_metrics.py           # Unit tests for CF Score + statistical testing
    └── test_pipeline.py          # Integration tests for the full 4-condition pipeline
```

---

## Reproducing All Results

```bash
# Full grid: 6 datasets × 5 shortcut types × 12 correlation strengths × 9 classifiers × 5-fold CV
# (thousands of model fits — several hours on a laptop CPU, faster with parallelism)
python experiments/reproduce_all.py

# Fast smoke-test grid (what CI runs) — a couple of minutes
python experiments/reproduce_all.py --quick

# Subset of datasets
python experiments/reproduce_all.py --datasets heart_disease,mammography

# Regenerate every figure from a results file
python experiments/generate_figures.py --results experiments/results/all_results.json
```

All results are written to `experiments/results/all_results.json`; all figures to `paper/figures/*.pdf` (vector graphics, ready for direct LaTeX inclusion).

**Grid sizes:**

| Mode | Coverage | Approx. Runtime |
|---|---|---|
| `--quick` | 2 datasets × 2 shortcuts × 3 strengths × 8 models (~96 configs) | ~5 minutes |
| Full | 6 datasets × 5 shortcuts × 12 strengths × 9 models (~3,240 configs) | Several hours (parallelizable) |

---

## Testing & CI

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov=shortcut_lens --cov-report=term-missing
```

Every push and pull request runs, via GitHub Actions (`.github/workflows/tests.yml`):

1. `flake8` linting
2. The full unit + integration test suite with coverage reporting
3. A `--quick` end-to-end smoke test of the entire experimental pipeline

on Python 3.10 and 3.11.

```bash
# Format code
black shortcut_lens/ experiments/ tests/

# Type checking
mypy shortcut_lens/
```

---

## Results Summary

> Populate with your actual numbers after `experiments/reproduce_all.py` completes a full run. Example table shape below.

| Model | Acc(A) | Acc(B) | Acc(C) | CF Score | SRS |
|---|---|---|---|---|---|
| Logistic Regression | 0.81 ± 0.03 | 0.88 ± 0.02 | 0.85 ± 0.03 | 0.85 ± 0.04 | 0.03 |
| SVM (RBF) | 0.84 ± 0.03 | 0.91 ± 0.02 | 0.88 ± 0.03 | 0.89 ± 0.04 | 0.03 |
| Decision Tree | 0.79 ± 0.04 | 0.90 ± 0.03 | 0.78 ± 0.04 | 0.72 ± 0.05 | 0.12* |
| Random Forest | 0.83 ± 0.04 | 0.94 ± 0.02 | 0.71 ± 0.05 | 0.24 ± 0.06 | 0.23* |
| XGBoost | 0.85 ± 0.03 | 0.95 ± 0.02 | 0.74 ± 0.04 | 0.29 ± 0.05 | 0.21* |

*\*p < 0.01 after Benjamini–Hochberg correction (Wilcoxon signed-rank, paired by fold).*

Reported significance thresholds used throughout the paper: `p < 0.001` for family-level comparisons (e.g. ensemble vs. linear), `p < 0.01` and `p < 0.05` for pairwise model comparisons.

---

## Paper

The full write-up (`paper/main.tex`) follows the standard ML-conference structure: Abstract, Introduction, Related Work, Datasets, Methodology, Results, Discussion, Practical Guidelines, Limitations, Conclusion. See `paper/references.bib` for the bibliography (Geirhos et al. 2020, Arjovsky et al. 2019 / IRM, D'Amour et al. 2020, Breiman 2001, Vapnik 1995, and others).

```bash
cd paper && latexmk -pdf main.tex
```

---

## Citing This Work

```bibtex
@misc{shortcutlens2026,
  author       = {Sunjoh Abdurazack and Usman Jabir and Gbanyawai Amadu and Ebrima Demba},
  title        = {ShortcutLens: Detecting Spurious Correlation Reliance in Classical
                  and Modern ML Classifiers Across Heterogeneous Tabular Datasets},
  year         = {2026},
  howpublished = {\url{https://github.com/InsightForge-ML/shortcut-lens}},
  note         = {Islamic University of Technology, CSE 4622 Machine Learning Lab}
}
```

---

## Contributing

Issues and pull requests are welcome.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run `pytest` and `flake8` locally — CI will otherwise fail the same checks
5. Commit (`git commit -m "Add amazing feature"`) and push
6. Open a Pull Request

See `.github/workflows/tests.yml` for the exact commands CI runs.

---

## License

[MIT](LICENSE) © 2026 Sunjoh Abdurazack and contributors.

---

## Authors

| Name | Student ID |
|---|---|
| Sunjoh Abdurazack | 220041258 |
| Usman Jabir | 220041262 |
| Gbanyawai Amadu | 220041266 |
| Ebrima Demba | 220041264 |

<div align="center">

⭐ *If this project is useful to you, consider starring the repository.* ⭐

</div>