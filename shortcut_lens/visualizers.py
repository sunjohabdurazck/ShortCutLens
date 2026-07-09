"""
visualizers.py
--------------
Plotting utilities for CF-score phase-transition curves, phase
transition heatmaps, and feature-importance audits. All figures are
saved as vector PDFs suitable for direct inclusion in a LaTeX paper.
"""

from __future__ import annotations

import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

DEFAULT_COLORS = {
    "logistic_regression": "#1f77b4",
    "knn_5": "#ff7f0e",
    "decision_tree": "#2ca02c",
    "svm_rbf": "#d62728",
    "random_forest": "#9467bd",
    "gradient_boosting": "#8c564b",
    "xgboost": "#e377c2",
    "mlp": "#7f7f7f",
}


def plot_cf_curves(
    all_results: dict,
    dataset_name: str,
    shortcut_type: str,
    correlation_strengths: Iterable[float],
    model_names: Iterable[str],
    out_dir: str = "paper/figures",
) -> str:
    """Phase-transition plot: CF Score vs. correlation strength for
    every model. This is typically Figure 1 of the paper."""
    correlation_strengths = list(correlation_strengths)
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name in model_names:
        cf_scores, ci_lowers, ci_uppers, xs = [], [], [], []
        for r in correlation_strengths:
            key = f"{dataset_name}__{shortcut_type}__{r}__{model_name}"
            if key in all_results:
                cf_data = all_results[key]["cf_score"]
                cf_scores.append(cf_data["cf_score"])
                ci_lowers.append(cf_data.get("ci_lower", cf_data["cf_score"]))
                ci_uppers.append(cf_data.get("ci_upper", cf_data["cf_score"]))
                xs.append(r)
        if not xs:
            continue
        color = DEFAULT_COLORS.get(model_name, "black")
        ax.plot(
            xs, cf_scores, label=model_name, color=color,
            linewidth=2, marker="o", markersize=4,
        )
        ax.fill_between(xs, ci_lowers, ci_uppers, alpha=0.1, color=color)

    ax.axhline(
        y=0.7, color="red", linestyle="--", alpha=0.7,
        label="CF=0.7 (deployment risk threshold)",
    )
    ax.axvspan(0.7, 1.0, alpha=0.05, color="red")
    ax.set_xlabel("Shortcut Correlation Strength (r)", fontsize=12)
    ax.set_ylabel("Causal Fidelity Score", fontsize=12)
    title = (
        f"Causal Fidelity vs Shortcut Strength\n"
        f"Dataset: {dataset_name}, Shortcut: {shortcut_type}"
    )
    ax.set_title(title, fontsize=13)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"cf_curve_{dataset_name}_{shortcut_type}.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_phase_transition_heatmap(
    all_results: dict,
    shortcut_type: str,
    datasets: Iterable[str],
    model_names: Iterable[str],
    correlation_strengths: Iterable[float],
    out_dir: str = "paper/figures",
    threshold: float = 0.7,
) -> str:
    """Heatmap of models x datasets, colour = correlation strength at
    which CF first drops below ``threshold``. The paper's key summary
    figure."""
    datasets, model_names, correlation_strengths = (
        list(datasets), list(model_names), list(correlation_strengths)
    )
    matrix = np.ones((len(model_names), len(datasets)))

    for i, model_name in enumerate(model_names):
        for j, dataset_name in enumerate(datasets):
            for r in correlation_strengths:
                key = f"{dataset_name}__{shortcut_type}__{r}__{model_name}"
                if key in all_results:
                    if all_results[key]["cf_score"]["cf_score"] < threshold:
                        matrix[i, j] = r
                        break

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        matrix, xticklabels=datasets, yticklabels=model_names, annot=True, fmt=".2f",
        cmap="RdYlGn", vmin=0.0, vmax=1.0, ax=ax,
        cbar_kws={"label": f"r at which CF < {threshold} (higher = more robust)"},
    )
    title = (
        "Phase Transition Heatmap: Shortcut Tolerance by Model and Dataset\n"
        f"(Shortcut type: {shortcut_type})"
    )
    ax.set_title(title)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"heatmap_{shortcut_type}.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_importance_audit(
    audit_results_by_model: dict, r_value: float, shortcut_type: str, out_dir: str = "paper/figures"
) -> str:
    """Bar chart comparing shortcut rank / dominance ratio across models."""
    models = list(audit_results_by_model.keys())
    ranks = [audit_results_by_model[m]["shortcut_rank"] for m in models]
    ratios = [audit_results_by_model[m]["dominance_ratio"] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(models, ranks, color=["#d62728" if r <= 3 else "#2ca02c" for r in ranks])
    ax1.axhline(y=3, color="red", linestyle="--", alpha=0.5, label="Top-3 threshold")
    ax1.set_ylabel("Shortcut Feature Rank (lower = more relied upon)")
    ax1.set_title(f"Shortcut Feature Rank\n(r={r_value}, type={shortcut_type})")
    ax1.legend()
    ax1.tick_params(axis="x", rotation=45)

    ax2.bar(models, ratios, color=["#d62728" if d > 1 else "#2ca02c" for d in ratios])
    ax2.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Dominance threshold")
    ax2.set_ylabel("Shortcut Dominance Ratio")
    ax2.set_title(f"Shortcut vs Real Feature Importance Ratio\n(r={r_value})")
    ax2.legend()
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"importance_audit_r{r_value}_{shortcut_type}.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
