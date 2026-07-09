"""
ShortcutLens
============

A robustness-auditing framework for measuring spurious-correlation
("shortcut") reliance in classical and modern tabular ML classifiers
under distribution shift.

Public API
----------
ShortcutInjector        -- inject controlled synthetic shortcuts into tabular data
causal_fidelity_score    -- compute the Causal Fidelity (CF) Score with bootstrap CIs
compare_models_statistically -- paired Wilcoxon + Benjamini-Hochberg testing
audit_shortcut_reliance -- feature-importance audit of shortcut dominance
run_full_experiment      -- run the 4-condition experimental protocol for one config

See https://github.com/<username>/shortcut-lens for full documentation.
"""

from .injectors import ShortcutInjector
from .metrics import causal_fidelity_score, compare_models_statistically
from .audit import audit_shortcut_reliance
from .pipeline import run_full_experiment, evaluate, aggregate_metrics

__version__ = "0.1.0"

__all__ = [
    "ShortcutInjector",
    "causal_fidelity_score",
    "compare_models_statistically",
    "audit_shortcut_reliance",
    "run_full_experiment",
    "evaluate",
    "aggregate_metrics",
]
