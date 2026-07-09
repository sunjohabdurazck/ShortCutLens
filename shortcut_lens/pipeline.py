"""
pipeline.py
-----------
The four-condition experimental protocol (A/B/C/D) that underlies
every ShortcutLens result, plus evaluation and aggregation helpers.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .injectors import ShortcutInjector


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute accuracy, macro-F1, and AUC for a fitted model."""
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    )
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "auc": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
    }


def aggregate_metrics(fold_results: list[dict]) -> dict:
    """Aggregate per-fold metric dicts into {metric: {mean, std, values}}."""
    keys = fold_results[0].keys()
    return {
        k: {
            "mean": float(np.mean([r[k] for r in fold_results])),
            "std": float(np.std([r[k] for r in fold_results])),
            "values": [r[k] for r in fold_results],
        }
        for k in keys
    }


def run_full_experiment(
    X: np.ndarray,
    y: np.ndarray,
    model,
    injector: ShortcutInjector,
    shortcut_type: str = "label_proxy",
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """Run all four experimental conditions with stratified k-fold CV.

    Condition A: Train CLEAN -> Test CLEAN            (baseline)
    Condition B: Train SHORTCUT -> Test SHORTCUT       (does it exploit it?)
    Condition C: Train SHORTCUT -> Test OOD            (deployment failure -- KEY)
    Condition D: Train SHORTCUT -> Test SHORTCUT-ONLY  (pure shortcut reliance)

    Returns
    -------
    dict mapping condition name -> aggregated metrics (see
    ``aggregate_metrics``).
    """
    X, y = np.asarray(X), np.asarray(y)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    results = {"condition_A": [], "condition_B": [], "condition_C": [], "condition_D": []}

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_s = injector.inject(shortcut_type, X_train, y_train)
        X_test_s = injector.inject(shortcut_type, X_test, y_test)
        X_test_ood = injector.remove_shortcut(X_test_s)
        X_test_shortcut_only = X_test_s.copy()
        X_test_shortcut_only[:, :-1] = 0

        scaler = StandardScaler()
        X_train_clean_scaled = scaler.fit_transform(X_train)
        X_test_clean_scaled = scaler.transform(X_test)

        scaler_s = StandardScaler()
        X_train_s_scaled = scaler_s.fit_transform(X_train_s)
        X_test_s_scaled = scaler_s.transform(X_test_s)
        X_test_ood_scaled = scaler_s.transform(X_test_ood)
        X_test_so_scaled = scaler_s.transform(X_test_shortcut_only)

        # Condition A
        model.fit(X_train_clean_scaled, y_train)
        results["condition_A"].append(evaluate(model, X_test_clean_scaled, y_test))

        # Conditions B, C, D share one shortcut-trained model
        model.fit(X_train_s_scaled, y_train)
        results["condition_B"].append(evaluate(model, X_test_s_scaled, y_test))
        results["condition_C"].append(evaluate(model, X_test_ood_scaled, y_test))
        results["condition_D"].append(evaluate(model, X_test_so_scaled, y_test))

    return {k: aggregate_metrics(v) for k, v in results.items()}
