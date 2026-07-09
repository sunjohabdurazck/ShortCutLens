"""
audit.py
--------
Feature-importance auditing: quantify how strongly a trained model
"trusts" the injected shortcut column relative to genuine features.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

TREE_BASED = {"random_forest", "gradient_boosting", "decision_tree", "xgboost"}
LINEAR = {"logistic_regression"}


def audit_shortcut_reliance(
    trained_model,
    model_name: str,
    feature_names: Sequence[str],
    shortcut_col_name: str = "SHORTCUT",
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
) -> Optional[dict]:
    """Extract feature importances and compute the shortcut dominance
    ratio for a trained model.

    Supports tree ensembles, logistic regression directly, and any
    scikit-learn estimator via permutation importance when ``X_test``
    and ``y_test`` are supplied (required for kernel SVM, MLP, k-NN).

    Returns
    -------
    dict with keys: importances, shortcut_rank (1 = most important,
    i.e. worst case), shortcut_importance, real_feature_mean,
    dominance_ratio (>1 means the shortcut dominates real features).
    Returns None if importances cannot be computed.
    """
    all_features = list(feature_names) + [shortcut_col_name]
    n_features = len(all_features)

    if model_name in TREE_BASED:
        importances = np.asarray(trained_model.feature_importances_)
    elif model_name in LINEAR:
        importances = np.abs(np.asarray(trained_model.coef_[0]))
    elif X_test is not None and y_test is not None:
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            trained_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        importances = result.importances_mean
    else:
        return None

    shortcut_rank = int(np.argsort(importances)[::-1].tolist().index(n_features - 1)) + 1
    shortcut_importance = float(importances[-1])
    real_feature_mean = float(np.mean(importances[:-1]))
    dominance_ratio = shortcut_importance / (real_feature_mean + 1e-8)

    return {
        "importances": dict(zip(all_features, importances.tolist())),
        "shortcut_rank": shortcut_rank,
        "shortcut_importance": shortcut_importance,
        "dominance_ratio": dominance_ratio,
        "real_feature_mean": real_feature_mean,
    }
