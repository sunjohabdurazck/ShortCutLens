import numpy as np
import pytest

from shortcut_lens import ShortcutInjector


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = rng.integers(0, 2, size=200)
    return X, y


@pytest.mark.parametrize(
    "shortcut_type",
    [
        "label_proxy",
        "demographic_proxy",
        "temporal_shortcut",
        "selection_bias",
        "measurement_artifact",
    ],
)
def test_injection_adds_one_column(toy_data, shortcut_type):
    X, y = toy_data
    injector = ShortcutInjector(correlation_strength=0.7, random_state=1)
    X_out = injector.inject(shortcut_type, X, y)
    assert X_out.shape == (X.shape[0], X.shape[1] + 1)
    # Original columns must be untouched.
    np.testing.assert_allclose(X_out[:, :-1], X)


def test_label_proxy_correlation_increases_with_r(toy_data):
    X, y = toy_data
    low = ShortcutInjector(correlation_strength=0.0, random_state=1).inject_label_proxy(X, y)
    high = ShortcutInjector(correlation_strength=0.99, random_state=1).inject_label_proxy(X, y)
    corr_low = abs(np.corrcoef(low[:, -1], y)[0, 1])
    corr_high = abs(np.corrcoef(high[:, -1], y)[0, 1])
    assert corr_high > corr_low


def test_remove_shortcut_destroys_correlation(toy_data):
    X, y = toy_data
    injector = ShortcutInjector(correlation_strength=0.99, random_state=1)
    X_shortcut = injector.inject_label_proxy(X, y)
    X_ood = injector.remove_shortcut(X_shortcut)
    corr_before = abs(np.corrcoef(X_shortcut[:, -1], y)[0, 1])
    corr_after = abs(np.corrcoef(X_ood[:, -1], y)[0, 1])
    assert corr_after < corr_before


def test_zero_out_shortcut_sets_column_to_zero(toy_data):
    X, y = toy_data
    injector = ShortcutInjector(correlation_strength=0.7, random_state=1)
    X_shortcut = injector.inject_label_proxy(X, y)
    X_ablated = injector.zero_out_shortcut(X_shortcut)
    assert np.all(X_ablated[:, -1] == 0.0)


def test_invalid_correlation_strength_raises():
    with pytest.raises(ValueError):
        ShortcutInjector(correlation_strength=1.5)


def test_unknown_shortcut_type_raises(toy_data):
    X, y = toy_data
    injector = ShortcutInjector()
    with pytest.raises(ValueError):
        injector.inject("not_a_real_type", X, y)


def test_reproducibility_with_fixed_seed(toy_data):
    X, y = toy_data
    a = ShortcutInjector(correlation_strength=0.5, random_state=7).inject_label_proxy(X, y)
    b = ShortcutInjector(correlation_strength=0.5, random_state=7).inject_label_proxy(X, y)
    np.testing.assert_allclose(a, b)
