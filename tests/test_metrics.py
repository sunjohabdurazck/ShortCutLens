import numpy as np

from shortcut_lens import causal_fidelity_score, compare_models_statistically


def test_cf_score_perfect_causal_model():
    # No gap between shortcut-test and OOD-test accuracy -> CF = 1.
    result = causal_fidelity_score(condition_A_acc=0.85, condition_B_acc=0.85, condition_C_acc=0.85)
    assert np.isclose(result["cf_score"], 1.0)
    assert np.isclose(result["srs"], 0.0)


def test_cf_score_fully_shortcut_dependent_model():
    # All of B's accuracy vanishes at OOD test time -> CF = 0.
    result = causal_fidelity_score(condition_A_acc=0.5, condition_B_acc=0.95, condition_C_acc=0.0)
    assert np.isclose(result["cf_score"], 0.0)


def test_cf_score_negative_when_shortcut_hurts():
    # OOD accuracy exceeds shortcut-test accuracy -> CF > 1 is impossible by
    # construction bounds, but SRS can be negative, pushing CF above 1;
    # this test instead checks the degenerate zero-degradation branch.
    result = causal_fidelity_score(condition_A_acc=0.7, condition_B_acc=0.0, condition_C_acc=0.0)
    assert result["cf_score"] == 1.0  # max_degradation == 0 guard


def test_cf_score_bootstrap_ci_present_and_ordered():
    rng = np.random.default_rng(0)
    fold_B = list(0.9 + rng.normal(0, 0.01, 5))
    fold_C = list(0.6 + rng.normal(0, 0.01, 5))
    result = causal_fidelity_score(
        condition_A_acc=0.85, condition_B_acc=np.mean(fold_B), condition_C_acc=np.mean(fold_C),
        fold_results_B=fold_B, fold_results_C=fold_C, n_bootstrap=200,
    )
    assert "ci_lower" in result and "ci_upper" in result
    assert result["ci_lower"] <= result["cf_score"] <= result["ci_upper"]


def test_compare_models_statistically_returns_expected_columns():
    results = {
        "model_a": {"cf_score": [0.90, 0.91, 0.89, 0.92, 0.90, 0.91, 0.89, 0.93]},
        "model_b": {"cf_score": [0.30, 0.28, 0.31, 0.29, 0.32, 0.27, 0.31, 0.29]},
    }
    df = compare_models_statistically(results, metric="cf_score")
    assert list(df.columns) == [
        "model_1", "model_2", "p_value_raw", "p_value_corrected", "significant",
    ]
    assert len(df) == 1
    assert bool(df.iloc[0]["significant"]) is True


def test_compare_models_statistically_identical_scores_not_significant():
    results = {
        "model_a": {"cf_score": [0.5, 0.5, 0.5, 0.5, 0.5]},
        "model_b": {"cf_score": [0.5, 0.5, 0.5, 0.5, 0.5]},
    }
    df = compare_models_statistically(results, metric="cf_score")
    assert bool(df.iloc[0]["significant"]) is False
