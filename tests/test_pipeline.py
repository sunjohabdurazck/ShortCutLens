import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from shortcut_lens import ShortcutInjector, causal_fidelity_score, run_full_experiment


def _toy_classification_data(n=300, n_features=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    true_weights = rng.normal(size=n_features)
    logits = X @ true_weights
    y = (logits > np.median(logits)).astype(int)
    return X, y


def test_full_pipeline_end_to_end_logistic_regression():
    X, y = _toy_classification_data()
    injector = ShortcutInjector(correlation_strength=0.9, random_state=42)
    model = LogisticRegression(max_iter=500, random_state=42)

    result = run_full_experiment(X, y, model, injector, shortcut_type="label_proxy", n_folds=3)

    for cond in ["condition_A", "condition_B", "condition_C", "condition_D"]:
        assert cond in result
        assert 0.0 <= result[cond]["accuracy"]["mean"] <= 1.0

    cf = causal_fidelity_score(
        result["condition_A"]["accuracy"]["mean"],
        result["condition_B"]["accuracy"]["mean"],
        result["condition_C"]["accuracy"]["mean"],
    )
    assert -1.0 <= cf["cf_score"] <= 1.0 + 1e-9


def test_high_correlation_label_proxy_creates_shortcut_gap_for_rf():
    """With a near-perfect label-proxy shortcut, a Random Forest should
    show materially inflated shortcut-test accuracy (condition B) versus
    OOD accuracy (condition C) -- i.e. it should exploit the shortcut."""
    X, y = _toy_classification_data(n=400, seed=1)
    injector = ShortcutInjector(correlation_strength=0.99, random_state=1)
    model = RandomForestClassifier(n_estimators=50, random_state=1)

    result = run_full_experiment(X, y, model, injector, shortcut_type="label_proxy", n_folds=3)
    acc_B = result["condition_B"]["accuracy"]["mean"]
    acc_C = result["condition_C"]["accuracy"]["mean"]

    assert acc_B >= acc_C  # shortcut inflates in-distribution performance


def test_zero_correlation_shortcut_has_no_effect():
    """With correlation_strength=0, the shortcut is pure noise, so
    condition B and condition C accuracy should be close."""
    X, y = _toy_classification_data(n=300, seed=2)
    injector = ShortcutInjector(correlation_strength=0.0, random_state=2)
    model = LogisticRegression(max_iter=500, random_state=2)

    result = run_full_experiment(X, y, model, injector, shortcut_type="label_proxy", n_folds=3)
    acc_B = result["condition_B"]["accuracy"]["mean"]
    acc_C = result["condition_C"]["accuracy"]["mean"]

    assert abs(acc_B - acc_C) < 0.15
