"""
utils.py
--------
Dataset loaders and the model suite registry used across
ShortcutLens experiments.

All loaders return ``(X, y)`` as NumPy arrays with ``y`` binarized to
{0, 1}. Loaders that hit the network (UCI / OpenML / Kaggle) cache
results under ``~/.shortcut_lens_cache`` to keep repeat runs fast and
offline-friendly.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

CACHE_DIR = os.path.expanduser("~/.shortcut_lens_cache")


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}.csv")


def load_heart_disease() -> tuple[np.ndarray, np.ndarray]:
    """UCI Heart Disease (Cleveland) -- 303 samples, 13 features, binary."""
    path = _cache_path("heart_disease")
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "heart-disease/processed.cleveland.data"
        )
        cols = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
        ]
        df = pd.read_csv(url, header=None, names=cols)
        df = df.replace("?", np.nan).dropna()
        df.to_csv(path, index=False)
    y = (df["target"].astype(float) > 0).astype(int).to_numpy()
    X = df.drop(columns=["target"]).astype(float).to_numpy()
    return X, y


def load_adult_income() -> tuple[np.ndarray, np.ndarray]:
    """UCI/OpenML Adult Income -- ~48.8k samples, 14 features, binary."""
    from sklearn.datasets import fetch_openml

    data = fetch_openml(name="adult", version=2, as_frame=True)
    df = data.data.copy()
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].cat.codes
    df = df.fillna(df.median(numeric_only=True))
    X = df.to_numpy(dtype=float)
    y = (data.target.astype(str) == ">50K").astype(int).to_numpy()
    return X, y


def load_credit_default() -> tuple[np.ndarray, np.ndarray]:
    """UCI Default of Credit Card Clients -- 30k samples, 23 features."""
    path = _cache_path("credit_default")
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
        "default%20of%20credit%20card%20clients.xls"
    )
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(url, header=1)
        df.to_csv(path, index=False)
    y = df["default payment next month"].astype(int).to_numpy()
    X = df.drop(columns=["ID", "default payment next month"]).astype(float).to_numpy()
    return X, y


def load_mammography() -> tuple[np.ndarray, np.ndarray]:
    """UCI Mammographic Mass -- 961 samples, 5 features, imbalanced."""
    path = _cache_path("mammography")
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        base = "https://archive.ics.uci.edu/ml/machine-learning-databases"
        url = f"{base}/mammographic-masses/mammographic_masses.data"
        cols = ["BI-RADS", "age", "shape", "margin", "density", "severity"]
        df = pd.read_csv(url, header=None, names=cols, na_values="?").dropna()
        df.to_csv(path, index=False)
    y = df["severity"].astype(int).to_numpy()
    X = df.drop(columns=["severity"]).astype(float).to_numpy()
    return X, y


def load_madelon() -> tuple[np.ndarray, np.ndarray]:
    """UCI MADELON -- 2600 samples, 500 features (420 designed noise)."""
    from sklearn.datasets import fetch_openml

    data = fetch_openml(name="madelon", version=1, as_frame=True)
    X = data.data.to_numpy(dtype=float)
    y = (data.target.astype(str) == "1").astype(int).to_numpy()
    return X, y


def load_bangla_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Local/regional dataset placeholder.

    Replace this with a Kaggle Bangladesh health/agriculture tabular
    dataset (see README "Datasets" section for candidates). Falls
    back to the UCI Statlog German Credit dataset if no local
    dataset has been configured, so the pipeline remains runnable
    end-to-end out of the box.
    """
    path = os.path.join(os.path.dirname(__file__), "..", "data", "bangla_dataset.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        y = df.iloc[:, -1].astype(int).to_numpy()
        X = df.iloc[:, :-1].astype(float).to_numpy()
        return X, y

    # Fallback: UCI Statlog German Credit
    from sklearn.datasets import fetch_openml

    data = fetch_openml(name="credit-g", version=1, as_frame=True)
    df = data.data.copy()
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].cat.codes
    X = df.to_numpy(dtype=float)
    y = (data.target.astype(str) == "bad").astype(int).to_numpy()
    return X, y


DATASET_LOADERS = {
    "heart_disease": load_heart_disease,
    "adult_income": load_adult_income,
    "credit_default": load_credit_default,
    "mammography": load_mammography,
    "madelon": load_madelon,
    "bangla_dataset": load_bangla_dataset,
}


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one of the six benchmark datasets by name."""
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Options: {list(DATASET_LOADERS)}")
    return DATASET_LOADERS[name]()


def preprocess(X: np.ndarray) -> np.ndarray:
    """Median-impute any remaining NaNs (defensive; loaders already
    drop/impute in most cases)."""
    X = np.asarray(X, dtype=float)
    if np.isnan(X).any():
        col_medians = np.nanmedian(X, axis=0)
        idx = np.where(np.isnan(X))
        X[idx] = np.take(col_medians, idx[1])
    return X


def get_model_suite() -> dict:
    """Return the full registry of classifiers used in ShortcutLens,
    including required baselines and the course-curriculum models."""
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    models = {
        "majority_class": DummyClassifier(strategy="most_frequent"),
        "random": DummyClassifier(strategy="uniform"),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "knn_5": KNeighborsClassifier(n_neighbors=5),
        "decision_tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "svm_rbf": SVC(kernel="rbf", probability=True, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "mlp": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0
        )
    except ImportError:
        pass  # xgboost is an optional dependency; skip if not installed

    return models
