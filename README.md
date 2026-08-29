<div align="center">

# 🔬 ShortcutLens

**Measuring Spurious Correlation Reliance in Tabular Classifiers with the Causal Fidelity Score**

ShortcutLens is a model-agnostic Python framework for measuring how much a tabular classifier's accuracy depends on spurious, non-causal shortcuts — before that dependence causes a silent failure in production.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

> A model can post 94% validation accuracy and still fall apart in deployment — not because the task changed, but because the model was never learning the task. It was learning a shortcut that happened to be lying around in the training distribution.
>
> **ShortcutLens estimates that risk before you deploy.**

---

## 🤔 Why ShortcutLens?

Standard validation tells you **how accurate** a model is. ShortcutLens tells you **how much of that accuracy you can trust**.

Shortcut learning — a model latching onto a spurious, non-causal correlation instead of the underlying signal — is well studied in deep vision models. It is comparatively unexamined in classical and ensemble classifiers on **tabular data**, despite tabular models being the dominant choice in healthcare, finance, and institutional decision-making.

ShortcutLens closes that gap with a controlled, model-agnostic diagnostic: it injects a synthetic shortcut into a real dataset at a known strength, trains normally, and then measures how much of the model's performance survives once that shortcut disappears at test time — summarized in a single interpretable number, the **Causal Fidelity (CF) Score**.

---

## ✨ Features

- 🔬 A four-condition evaluation protocol that isolates shortcut-supported accuracy from causally-grounded accuracy
- 📊 The **Causal Fidelity (CF) Score**, backed by non-parametric bootstrap confidence intervals (1,000 replicates over 5-fold CV)
- ⚡ **5 shortcut injection mechanisms** — label proxy, demographic proxy, temporal shortcut, selection bias, measurement artifact
- 📐 A **12-point shortcut-strength grid** (`r ∈ {0.0, 0.1, …, 0.9, 0.95, 0.99}`) for locating each model's shortcut-tolerance transition
- 🤖 A benchmark suite of **8 classifiers** spanning linear, kernel, instance-based, tree-based, boosting, and neural families
- 📈 Statistical comparison via two-sided Wilcoxon signed-rank tests with Benjamini–Hochberg correction, plus permutation-importance auditing
- 📦 Pip-installable package with a small, composable API (`injectors.py`, `pipeline.py`, `metrics.py`, `audit.py`, `visualizers.py`)

---

## Table of Contents

- [Why ShortcutLens?](#-why-shortcutlens)
- [Features](#-features)
- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [Key Findings](#-key-findings)
- [The Causal Fidelity Score](#-the-causal-fidelity-score)
- [Quickstart](#-quickstart)
- [Installation](#-installation)
- [Usage](#-usage)
- [The Experimental Protocol](#-the-experimental-protocol)
- [Shortcut Types](#-shortcut-types)
- [Datasets](#-datasets)
- [Benchmarked Models](#-benchmarked-models)
- [Repository Structure](#️-repository-structure)
- [Reproducing All Results](#-reproducing-all-results)
- [Testing & CI](#-testing--ci)
- [Results Summary](#-results-summary)
- [Paper](#-paper)
- [Citing This Work](#-citing-this-work)
- [Contributing](#-contributing)
- [License](#️-license)

---

## 📖 Overview

Machine learning models deployed in real-world settings frequently encounter **distribution shift**, where a statistical pattern present at training time weakens or disappears at inference time. Shortcut learning — a model's tendency to rely on such a pattern instead of genuine causal signal — has been studied extensively in deep learning and computer vision, but its prevalence in **classical and ensemble tabular classifiers** remains comparatively unexamined.

**ShortcutLens** answers one question for any tabular classifier: *is it relying on causally meaningful features, or has it latched onto a spurious correlation that may not survive deployment?*

It does this by:

1. **Injecting a controlled synthetic shortcut** into a real tabular dataset, using one of 5 mechanisms at one of 12 strengths.
2. **Evaluating the classifier under four experimental conditions** — clean baseline, shortcut-available, shortcut-removed, and shortcut-only — to isolate how much of its accuracy is shortcut-dependent.
3. **Scoring the result with the Causal Fidelity (CF) Score** — a single bootstrap-backed number computable before deployment.
4. **Auditing feature importances** via permutation importance to show *which* feature a model is actually relying on.
5. **Backing every comparative claim statistically**, with two-sided Wilcoxon signed-rank tests (Benjamini–Hochberg corrected, α = 0.05) across stratified 5-fold CV.

The benchmark evaluates 8 classifiers × 6 datasets × 5 mechanisms × 12 strengths × 4 conditions × 5 folds, with identical folds and preprocessing held fixed across conditions so that observed differences reflect the injected shortcut rather than experimental noise.

---

## 🧭 How It Works

```
   Dataset
      │
      ▼
Shortcut Injection    (5 mechanisms × 12 strengths)
      │
      ▼
   Classifier          (8 models: linear, kernel, instance-based,
      │                 tree-based, boosting, neural)
      ▼
4-Condition Evaluation  (A: clean · B: shortcut-available ·
      │                  C: shortcut-removed · D: shortcut-only)
      ▼
  CF Score              (bootstrap CI + Wilcoxon significance testing)
      │
      ▼
 Visualizations         (CF-vs-strength curves, phase-transition
                          heatmaps, importance plots)
```

Each stage is a separate, composable module, so any stage can be swapped, extended, or run independently — see [Usage](#-usage).

---

## 🎯 Key Findings

These findings summarize the paper's benchmark across 6 datasets, 5 shortcut mechanisms, and shortcut strengths r ≥ 0.5 (210 CF observations per classifier).

- 🏆 **k-Nearest Neighbors is the most robust classifier tested**, with the highest mean CF Score (**0.8197**) and the highest CF on several individually difficult conditions (e.g. CF = 0.81 on the SPAS dataset under demographic-proxy injection).
- ⚠️ **XGBoost is the least robust**, with the lowest mean CF Score (**0.6982**); tree-based and boosting models generally fall below the 0.7 operational threshold under demographic- and label-proxy injection.
- 🔴 **Demographic-proxy shortcuts cause the most severe degradation**, crossing the CF = 0.7 risk threshold at markedly lower correlation strengths than label-proxy shortcuts — because a demographic feature correlates with both the label *and* an existing real feature, creating a redundant predictive path that greedy learners exploit even at moderate strength.
- 🟢 **Measurement-artifact and temporal shortcuts leave CF comparatively stable** (often > 0.95), even as their generation strength approaches 1.0, since as constructed they lack that redundant path to the label.
- 📊 On the Heart Disease dataset under demographic-proxy injection at r = 0.5, kNN achieves significantly higher CF than Gradient Boosting, Decision Tree, Random Forest, XGBoost, and the MLP (Wilcoxon signed-rank, p < 0.05 after Benjamini–Hochberg correction); differences against Logistic Regression and SVM (RBF) were not statistically significant.

> Full per-dataset, per-mechanism numbers are in the paper's Table II (CF at r = 0.5) and Table IV (overall ranking) — see [Results Summary](#-results-summary).

---

## 🧮 The Causal Fidelity Score

```
SRS = B - C                    (Shortcut Reliance Score: absolute performance loss)
CF  = 1 - SRS / B  =  C / B    (Causal Fidelity Score: normalized performance retention)
```

| Condition | Train | Test | Purpose |
|---|---|---|---|
| **A** | Clean | Clean | Clean baseline |
| **B** | Augmented (shortcut present) | Augmented (shortcut present) | Shortcut-available accuracy |
| **C** | Augmented (shortcut present) | Shortcut replaced with Gaussian noise | Shortcut-removal accuracy — the simulated deployment-shift condition |
| **D** | Augmented (shortcut present) | Shortcut feature only | Shortcut-only diagnostic |

A CF value close to **1.0** means most shortcut-available performance is retained once the shortcut disappears — the model's accuracy looks causally grounded. A lower CF means the model's performance is sensitive to the shortcut going away. The paper uses **CF = 0.7** as a study-specific threshold for flagging a model for further investigation, and treats CF as a *performance-retention diagnostic*, not a literal proportion of causally-grounded predictions. When B < 0.01, CF-based ranking is excluded because the ratio becomes unstable.

Uncertainty in CF is quantified with a non-parametric bootstrap over the 5 cross-validation folds (1,000 replicates), reporting the 2.5th–97.5th percentile as a 95% confidence interval.

```python
from shortcut_lens import causal_fidelity_score

result = causal_fidelity_score(
    condition_A_acc=0.84,
    condition_B_acc=0.94,
    condition_C_acc=0.71,
    fold_results_B=[0.93, 0.95, 0.94, 0.94, 0.95],
    fold_results_C=[0.70, 0.72, 0.71, 0.70, 0.72],
)
# {'cf_score': 0.755, 'srs': 0.23, 'ci_lower': 0.71, 'ci_upper': 0.79, ...}
```

---

## ⚡ Quickstart

```bash
git clone https://anonymous.4open.science/r/ShortCutLens-153A/
cd ShortCutLens-153A
pip install -e .

python - <<'PY'
from shortcut_lens import ShortcutInjector, run_full_experiment, causal_fidelity_score
from sklearn.ensemble import RandomForestClassifier
from shortcut_lens.utils import load_dataset

X, y = load_dataset("heart_disease")
injector = ShortcutInjector(correlation_strength=0.5)
model = RandomForestClassifier(n_estimators=100, random_state=42)

result = run_full_experiment(X, y, model, injector, shortcut_type="demographic_proxy")
cf = causal_fidelity_score(
    result["condition_A"]["accuracy"]["mean"],
    result["condition_B"]["accuracy"]["mean"],
    result["condition_C"]["accuracy"]["mean"],
)
print(f"Causal Fidelity Score: {cf['cf_score']:.3f}")
PY
```

> For a longer, hand-held walkthrough (including troubleshooting), see [SETUP.md](SETUP.md).

---

## 📦 Installation

**Requirements:** Python 3.9+, Git, and (optionally) LaTeX for compiling the paper.

### Editable install (recommended for development)

```bash
git clone https://anonymous.4open.science/r/ShortCutLens-153A/
cd ShortCutLens-153A

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

## 💻 Usage

### 1. Inject a shortcut

```python
from shortcut_lens import ShortcutInjector

injector = ShortcutInjector(correlation_strength=0.5, random_state=42)
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
    {"knn_5": {"cf_score": [...]}, "xgboost": {"cf_score": [...]}},
    metric="cf_score",
)
```

### Command-line interface

```bash
# Full grid: 6 datasets × 5 shortcuts × 12 strengths × 8 classifiers × 5-fold CV
python experiments/reproduce_all.py

# Fast smoke-test grid (what CI runs)
python experiments/reproduce_all.py --quick

# Subset of datasets
python experiments/reproduce_all.py --datasets heart_disease,mammography

# Regenerate every figure from a results file
python experiments/generate_figures.py --results experiments/results/all_results.json
```

---

## 🧪 The Experimental Protocol

Every (dataset, shortcut mechanism, strength, classifier) combination is evaluated with stratified 5-fold cross-validation across all four conditions above, with preprocessing (standardization, imputation) fit on the training fold only to avoid leakage. The gap between **Condition B** and **Condition C** is the deployment-time performance loss the CF Score is designed to catch before it happens in production.

---

## 🧬 Shortcut Types

| # | Mechanism | Construction |
|---|---|---|
| 1 | `label_proxy` | Mixes a sign-encoded class label with Gaussian noise |
| 2 | `demographic_proxy` | Combines label information with a normalized existing feature |
| 3 | `temporal_shortcut` | A label-related signal that decays with sample order |
| 4 | `selection_bias` | A noisy feature with class-conditional mean shifts controlled by strength |
| 5 | `measurement_artifact` | A signal generated across three collection batches with batch-dependent strength |

Because the five mechanisms differ in construction, the strength parameter `r` is interpreted independently per mechanism rather than as a shared Pearson correlation. Every experiment sweeps **12 strengths**: `r ∈ {0.0, 0.1, 0.2, …, 0.9, 0.95, 0.99}`.

---

## 📊 Datasets

Six real-world datasets spanning healthcare, finance, and agriculture, chosen to vary sample size, dimensionality, class balance, and domain.

| Dataset | Samples | Features | Domain |
|---|---|---|---|
| [Heart Disease (UCI)](https://archive.ics.uci.edu/dataset/45/heart+disease) | 303 | 13 | Healthcare |
| [Mammographic Mass (UCI)](https://archive.ics.uci.edu/dataset/161/mammographic+mass) | 961 | 5 | Healthcare |
| [Adult Income (UCI)](https://archive.ics.uci.edu/dataset/2/adult) | 48,842 | 14 | Finance / fairness |
| [Credit Card Default (UCI)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) | 30,000 | 23 | Finance |
| SPAS-Dataset-BD | ~13,000 | 9 (selected) | Smart precision agriculture, Bangladesh |
| MADELON (UCI) | 2,600 | 500 | Synthetic / high-dimensional feature selection |

`shortcut_lens/utils.py::load_dataset(name)` handles fetching and caching for all six.

---

## 🤖 Benchmarked Models

| Family | Model(s) |
|---|---|
| Linear | Logistic Regression |
| Kernel | SVM (RBF kernel) |
| Instance-based | k-Nearest Neighbors (k = 5) |
| Tree-based | Decision Tree (max depth 10), Random Forest (100 estimators) |
| Boosting | Gradient Boosting (100 estimators), XGBoost |
| Neural | Multi-Layer Perceptron (hidden layers 128, 64; ReLU) |

Hyperparameters are held fixed across all conditions so observed differences reflect the injected shortcut rather than tuning variation. `shortcut_lens/utils.py` exposes a model-suite registry, so the classifier set is easily extensible.

---

## 🗂️ Repository Structure

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
│   ├── reproduce_all.py          # Runs the full experimental grid
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

## 🔁 Reproducing All Results

```bash
# Full grid: 6 datasets × 5 shortcut types × 12 strengths × 8 classifiers × 4 conditions × 5-fold CV
python experiments/reproduce_all.py

# Fast smoke-test grid (what CI runs)
python experiments/reproduce_all.py --quick

# Subset of datasets
python experiments/reproduce_all.py --datasets heart_disease,mammography

# Regenerate every figure from a results file
python experiments/generate_figures.py --results experiments/results/all_results.json
```

All results are written to `experiments/results/all_results.json`; all figures to `paper/figures/*.pdf` (vector graphics, ready for direct LaTeX inclusion).

| Mode | Coverage |
|---|---|
| `--quick` | A reduced subset of datasets, mechanisms, strengths, and models for a fast end-to-end smoke test |
| Full | 8 classifiers × 6 datasets × 5 mechanisms × 12 strengths × 4 conditions × 5 folds (the full paper grid) |

---

## ✅ Testing & CI

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov=shortcut_lens --cov-report=term-missing
```

Every push and pull request runs, via GitHub Actions (`.github/workflows/tests.yml`):

1. `flake8` linting
2. The full unit + integration test suite with coverage reporting
3. A `--quick` end-to-end smoke test of the entire experimental pipeline

```bash
# Format code
black shortcut_lens/ experiments/ tests/

# Type checking
mypy shortcut_lens/
```

---

## 📈 Results Summary

Overall classifier ranking by mean CF Score across all 6 datasets, 5 shortcut mechanisms, and shortcut strengths r ≥ 0.5 (210 CF observations per classifier):

| Rank | Model | Mean CF | Std. Dev. |
|---|---|---|---|
| 1 | k-NN (k = 5) | 0.8197 | 0.089 |
| 2 | SVM (RBF) | 0.7425 | 0.112 |
| 3 | Random Forest | 0.7342 | 0.124 |
| 4 | Logistic Regression | 0.7276 | 0.118 |
| 5 | Gradient Boosting | 0.7196 | 0.134 |
| 6 | MLP | 0.7167 | 0.128 |
| 7 | Decision Tree | 0.7005 | 0.146 |
| 8 | XGBoost | 0.6982 | 0.152 |

At r = 0.5, demographic-proxy injection produces the largest CF degradation across most datasets (CF falling below 0.60 for most classifiers on Adult, Credit Default, Heart Disease, MADELON, and Mammographic Mass), while measurement-artifact and temporal shortcuts leave CF comparatively stable across the board. Full per-dataset, per-mechanism CF values at r = 0.5 are reported in the paper's Table II; pairwise significance tests for the Heart Disease demographic-proxy condition are in Table III.

---

## 📄 Paper

The full write-up follows a standard conference structure: Abstract, Introduction, Related Work, Datasets, Methodology, Results, Discussion, Practical Guidance, Threats to Validity, Conclusion.

```bash
cd paper && latexmk -pdf main.tex
```

---

## 📝 Citing This Work

This repository accompanies a paper currently under anonymous review. A full citation will be added upon publication.

```bibtex
@misc{shortcutlens2026,
  title        = {ShortcutLens: Measuring Spurious Correlation Reliance in Tabular
                  Classifiers with the Causal Fidelity Score},
  year         = {2026},
  howpublished = {\url{https://anonymous.4open.science/r/ShortCutLens-153A/}},
  note         = {Anonymous submission}
}
```

---

## 🤝 Contributing

Issues and pull requests are welcome.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run `pytest` and `flake8` locally — CI will otherwise fail the same checks
5. Commit (`git commit -m "Add amazing feature"`) and push
6. Open a Pull Request

See `.github/workflows/tests.yml` for the exact commands CI runs.

---

## ⚖️ License

Released under the [MIT License](LICENSE).

<div align="center">

⭐ *If this project is useful to you, consider starring the repository.* ⭐

</div>
