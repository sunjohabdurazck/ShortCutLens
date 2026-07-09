"""
metrics.py
----------
The Causal Fidelity (CF) Score and statistical significance testing
utilities used to compare classifiers' shortcut reliance.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


def causal_fidelity_score(
    condition_A_acc: float,
    condition_B_acc: float,
    condition_C_acc: float,
    fold_results_B: Optional[Iterable[float]] = None,
    fold_results_C: Optional[Iterable[float]] = None,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict:
    """Compute the Causal Fidelity (CF) Score.

    CF = 1 - SRS / Acc(B),  where SRS (Shortcut Reliance Score) =
    Acc(B) - Acc(C).

    Interpretation
    --------------
    CF = 1.0  -> model ignores the shortcut entirely (fully causal)
    CF = 0.0  -> model's performance is entirely shortcut-dependent
    CF < 0.0  -> the shortcut actively hurts generalization (rare)

    Parameters
    ----------
    condition_A_acc, condition_B_acc, condition_C_acc : float
        Mean accuracy (or other metric) under conditions A (clean/clean),
        B (shortcut/shortcut) and C (shortcut/OOD) respectively.
    fold_results_B, fold_results_C : sequence of float, optional
        Per-fold scores for B and C. If both are supplied, a
        bootstrap 95% confidence interval is computed.
    n_bootstrap : int, default=1000
        Number of bootstrap resamples.

    Returns
    -------
    dict with keys: cf_score, srs, ci_lower, ci_upper (CI keys are
    only present when fold-level data is supplied).
    """
    srs = condition_B_acc - condition_C_acc
    max_degradation = condition_B_acc

    if max_degradation == 0:
        return {"cf_score": 1.0, "srs": 0.0, "ci_lower": 1.0, "ci_upper": 1.0}

    cf = 1 - (srs / max_degradation)
    result = {"cf_score": cf, "srs": srs, "condition_A": condition_A_acc}

    if fold_results_B is not None and fold_results_C is not None:
        rng = np.random.default_rng(random_state)
        b_arr, c_arr = np.asarray(list(fold_results_B)), np.asarray(list(fold_results_C))
        bootstrap_cfs = []
        for _ in range(n_bootstrap):
            b_sample = rng.choice(b_arr, len(b_arr), replace=True)
            c_sample = rng.choice(c_arr, len(c_arr), replace=True)
            b_mean, c_mean = b_sample.mean(), c_sample.mean()
            if b_mean > 0:
                bootstrap_cfs.append(1 - (b_mean - c_mean) / b_mean)
        if bootstrap_cfs:
            result["ci_lower"] = float(np.percentile(bootstrap_cfs, 2.5))
            result["ci_upper"] = float(np.percentile(bootstrap_cfs, 97.5))

    return result


def compare_models_statistically(
    results_dict: dict, metric: str = "cf_score", alpha: float = 0.05
) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank tests with Benjamini-Hochberg FDR
    correction across all model pairs.

    Parameters
    ----------
    results_dict : dict[str, dict[str, Sequence[float]]]
        Mapping of model name -> {metric_name: [per-fold values]}.
    metric : str
        Which metric to compare (must be a key inside each model's dict).
    alpha : float
        Significance level for the FDR correction.

    Returns
    -------
    pandas.DataFrame sorted by corrected p-value, with columns:
    model_1, model_2, p_value_raw, p_value_corrected, significant.
    """
    model_names = list(results_dict.keys())
    p_values, pairs = [], []

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            scores_1 = np.asarray(results_dict[m1][metric])
            scores_2 = np.asarray(results_dict[m2][metric])
            if len(scores_1) < 5 or np.allclose(scores_1, scores_2):
                p_values.append(1.0)
            else:
                _, p = wilcoxon(scores_1, scores_2, alternative="two-sided")
                p_values.append(p)
            pairs.append((m1, m2))

    if not p_values:
        return pd.DataFrame(
            columns=["model_1", "model_2", "p_value_raw", "p_value_corrected", "significant"]
        )

    rejected, corrected_p, _, _ = multipletests(p_values, method="fdr_bh", alpha=alpha)

    df = pd.DataFrame(
        {
            "model_1": [p[0] for p in pairs],
            "model_2": [p[1] for p in pairs],
            "p_value_raw": p_values,
            "p_value_corrected": corrected_p,
            "significant": rejected,
        }
    )
    return df.sort_values("p_value_corrected").reset_index(drop=True)
