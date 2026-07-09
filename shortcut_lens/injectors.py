"""
injectors.py
------------
Controlled injection of synthetic spurious correlations ("shortcuts")
into tabular datasets, and utilities to simulate their removal at
deployment time (out-of-distribution / OOD testing).

Each `inject_*` method appends exactly one synthetic shortcut column
to the feature matrix `X`. The strength of the correlation between
that column and the label `y` is controlled by `correlation_strength`
(0.0 = pure noise, 1.0 = near-perfect proxy for the label).
"""

from __future__ import annotations

import numpy as np


class ShortcutInjector:
    """Injects controlled spurious correlations into tabular datasets.

    Parameters
    ----------
    correlation_strength : float, default=0.7
        Value in [0, 1]. 0.0 yields a shortcut column that is pure
        noise; 1.0 yields a column that is (almost) a perfect proxy
        for the label.
    random_state : int, default=42
        Seed for the internal NumPy random generator, for
        reproducibility.

    Notes
    -----
    All methods take ``X`` (n_samples, n_features) and ``y``
    (n_samples,) and return a new array with one additional column
    appended (the shortcut). The original arrays are never mutated.
    """

    VALID_TYPES = (
        "label_proxy",
        "demographic_proxy",
        "temporal_shortcut",
        "selection_bias",
        "measurement_artifact",
    )

    def __init__(self, correlation_strength: float = 0.7, random_state: int = 42):
        if not 0.0 <= correlation_strength <= 1.0:
            raise ValueError("correlation_strength must be in [0, 1]")
        self.r = correlation_strength
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Shortcut types
    # ------------------------------------------------------------------
    def inject_label_proxy(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Type 1 — Label Proxy.

        A feature that directly mirrors the label in training, e.g. a
        data-collection artifact that leaks the outcome (patient ID
        range correlated with diagnosis at a single hospital).

        Implemented as a convex blend of the (sign-encoded) label and
        pure Gaussian noise, so correlation with ``y`` increases
        monotonically with ``r``: r=0 is indistinguishable from
        noise, r->1 approaches a near-perfect proxy.
        """
        X, y = np.asarray(X), np.asarray(y)
        label_signal = y.astype(float) * 2 - 1  # {0,1} -> {-1,+1}
        noise = self.rng.normal(0, 1, len(y))
        shortcut = self.r * label_signal + (1 - self.r) * noise
        return np.column_stack([X, shortcut])

    def inject_demographic_proxy(
        self, X: np.ndarray, y: np.ndarray, proxy_col_idx: int = 0
    ) -> np.ndarray:
        """Type 2 — Demographic Proxy.

        An existing feature (e.g. zip code, occupation) is blended
        with the label to simulate a protected-attribute proxy.
        """
        X, y = np.asarray(X), np.asarray(y)
        proxy = X[:, proxy_col_idx].astype(float).copy()
        proxy_normalized = (proxy - proxy.mean()) / (proxy.std() + 1e-8)
        label_signal = y.astype(float) * 2 - 1  # map {0,1} -> {-1,+1}
        shortcut = self.r * label_signal + (1 - self.r) * proxy_normalized
        return np.column_stack([X, shortcut])

    def inject_temporal_shortcut(
        self, X: np.ndarray, y: np.ndarray, decay_rate: float = 0.1
    ) -> np.ndarray:
        """Type 3 — Temporal Shortcut.

        Valid early in the data stream, decays over "time" (row
        order), simulating seasonal effects, policy changes, or
        sensor drift.
        """
        X, y = np.asarray(X), np.asarray(y)
        n = len(y)
        time_weights = np.exp(-decay_rate * np.arange(n))
        noise = self.rng.normal(0, 1, n)
        shortcut = self.r * time_weights * y.astype(float) + (1 - self.r) * noise
        return np.column_stack([X, shortcut])

    def inject_selection_bias(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Type 4 — Selection Bias Shortcut.

        Simulates non-random sampling (e.g. training data pooled from
        specific hospitals/regions) by shifting the shortcut's mean
        per class.
        """
        X, y = np.asarray(X), np.asarray(y)
        positive_idx = np.where(y == 1)[0]
        negative_idx = np.where(y == 0)[0]
        shortcut = self.rng.normal(0, 1, len(y))
        shortcut[positive_idx] += self.r * 2
        shortcut[negative_idx] -= self.r * 2
        shortcut += self.rng.normal(0, max(1 - self.r, 1e-6), len(y))
        return np.column_stack([X, shortcut])

    def inject_measurement_artifact(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Type 5 — Measurement Artifact.

        Systematic bias from the data-collection instrument/process
        (e.g. scanner ID in medical imaging), where the artifact only
        correlates with the label within certain collection batches.
        """
        X, y = np.asarray(X), np.asarray(y)
        n = len(y)
        batch_ids = self.rng.integers(0, 3, n)
        shortcut = np.zeros(n)
        for batch in range(3):
            mask = batch_ids == batch
            batch_r = self.r * (batch / 2)
            shortcut[mask] = batch_r * y[mask].astype(float) + (
                1 - batch_r
            ) * self.rng.normal(0, 1, mask.sum())
        return np.column_stack([X, shortcut])

    def inject(self, shortcut_type: str, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """Dispatch to the appropriate ``inject_*`` method by name."""
        if shortcut_type not in self.VALID_TYPES:
            raise ValueError(
                f"Unknown shortcut_type '{shortcut_type}'. Valid options: {self.VALID_TYPES}"
            )
        return getattr(self, f"inject_{shortcut_type}")(X, y, **kwargs)

    # ------------------------------------------------------------------
    # Deployment simulation
    # ------------------------------------------------------------------
    def remove_shortcut(self, X_with_shortcut: np.ndarray) -> np.ndarray:
        """Simulate deployment by replacing the shortcut column with
        pure Gaussian noise (out-of-distribution test set)."""
        X_ood = np.asarray(X_with_shortcut).copy()
        X_ood[:, -1] = self.rng.normal(0, 1, len(X_ood))
        return X_ood

    def zero_out_shortcut(self, X_with_shortcut: np.ndarray) -> np.ndarray:
        """Alternative ablation: zero the shortcut column instead of
        replacing it with noise, for ablation comparisons."""
        X_ablated = np.asarray(X_with_shortcut).copy()
        X_ablated[:, -1] = 0.0
        return X_ablated
