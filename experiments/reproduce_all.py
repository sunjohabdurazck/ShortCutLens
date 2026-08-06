#!/usr/bin/env python
"""
reproduce_all.py
-----------------
Single-command reproduction of every ShortcutLens result.

USAGE:
    # Full run (all 6 datasets, full grid - takes hours)
    python experiments/reproduce_all.py

    # Quick smoke test (2 datasets, 2 shortcut types, 3 strengths - takes minutes)
    python experiments/reproduce_all.py --quick

    # Run only remaining 3 large datasets with speed optimizations (takes ~15 min)
    python experiments/reproduce_all.py --remaining

    # Run specific datasets only
    python experiments/reproduce_all.py --datasets heart_disease,mammography

    # Combine: quick + specific datasets
    python experiments/reproduce_all.py --quick --datasets heart_disease
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shortcut_lens import ShortcutInjector, causal_fidelity_score, run_full_experiment
from shortcut_lens.utils import get_model_suite, load_dataset

# ============================================================
# CONFIGURATIONS
# ============================================================

FULL_DATASETS = [
    "heart_disease", "adult_income", "credit_default",
    "mammography", "madelon", "bangla_dataset",
]
FULL_SHORTCUT_TYPES = [
    "label_proxy", "demographic_proxy", "temporal_shortcut",
    "selection_bias", "measurement_artifact",
]
FULL_CORRELATION_STRENGTHS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
FULL_MODELS = None  # All models from get_model_suite()

QUICK_DATASETS = ["heart_disease", "mammography"]
QUICK_SHORTCUT_TYPES = ["label_proxy", "demographic_proxy"]
QUICK_CORRELATION_STRENGTHS = [0.0, 0.5, 0.9]
QUICK_MODELS = ["logistic_regression", "random_forest", "knn_5"]

# Remaining mode - optimized for the 3 large datasets
REMAINING_DATASETS = ["credit_default", "adult_income", "madelon"]
REMAINING_SHORTCUT_TYPES = FULL_SHORTCUT_TYPES  # All 5
REMAINING_CORRELATION_STRENGTHS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]  # 7 strengths
REMAINING_MODELS = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
]  # Skip slow models

# ============================================================
# OPTIMIZATION FUNCTIONS (for remaining mode)
# ============================================================

def optimize_dataset(dataset_name: str, X, y):
    """Apply dataset-specific optimizations for speed."""
    import numpy as np
    from sklearn.feature_selection import SelectKBest, f_classif
    
    if dataset_name == "credit_default":
        if len(X) > 5000:
            idx = np.random.choice(len(X), 5000, replace=False)
            X, y = X[idx], y[idx]
            print(f"  Subsampled credit_default to 5000 samples")
    
    elif dataset_name == "adult_income":
        if len(X) > 5000:
            idx = np.random.choice(len(X), 5000, replace=False)
            X, y = X[idx], y[idx]
            print(f"  Subsampled adult_income to 5000 samples")
    
    elif dataset_name == "madelon":
        if X.shape[1] > 100:
            selector = SelectKBest(f_classif, k=80)
            X = selector.fit_transform(X, y)
            print(f"  Reduced madelon to 80 features (from {X.shape[1]})")
    
    return X, y

# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Reproduce ShortcutLens experiments.",
        epilog="""
Examples:
  python reproduce_all.py                    # Full run (overnight)
  python reproduce_all.py --quick            # Quick smoke test (minutes)
  python reproduce_all.py --remaining        # Run only remaining 3 datasets fast
  python reproduce_all.py --datasets heart_disease,mammography  # Specific datasets
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--quick", action="store_true", help="Run a fast smoke-test grid (2 datasets, 2 shortcut types, 3 strengths)")
    p.add_argument("--remaining", action="store_true", help="Run only remaining 3 large datasets with speed optimizations (~15 min)")
    p.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset subset")
    p.add_argument("--n-folds", type=int, default=None, help="Number of CV folds (default: 5, or 3 for --remaining)")
    p.add_argument("--out", type=str, default="experiments/results/all_results.json")
    return p.parse_args()


def main():
    args = parse_args()
    
    # ============================================================
    # DETERMINE CONFIGURATION
    # ============================================================
    
    if args.quick:
        datasets = QUICK_DATASETS
        shortcut_types = QUICK_SHORTCUT_TYPES
        strengths = QUICK_CORRELATION_STRENGTHS
        model_filter = QUICK_MODELS
        n_folds = args.n_folds if args.n_folds is not None else 5
        mode = "QUICK"
    
    elif args.remaining:
        datasets = REMAINING_DATASETS
        shortcut_types = REMAINING_SHORTCUT_TYPES
        strengths = REMAINING_CORRELATION_STRENGTHS
        model_filter = REMAINING_MODELS
        n_folds = args.n_folds if args.n_folds is not None else 3
        mode = "REMAINING (optimized for speed)"
        use_optimization = True
    
    else:
        datasets = FULL_DATASETS
        shortcut_types = FULL_SHORTCUT_TYPES
        strengths = FULL_CORRELATION_STRENGTHS
        model_filter = None  # All models
        n_folds = args.n_folds if args.n_folds is not None else 5
        mode = "FULL"
        use_optimization = False
    
    # Override with --datasets if provided
    if args.datasets:
        requested = set(args.datasets.split(","))
        datasets = [d for d in datasets if d in requested] or list(requested)
    
    # ============================================================
    # LOAD MODELS
    # ============================================================
    
    models = get_model_suite()
    active_models = {k: v for k, v in models.items() 
                    if k not in ("majority_class", "random")}
    
    # Filter models if specified
    if model_filter:
        active_models = {k: v for k, v in active_models.items() 
                        if k in model_filter}
    
    # ============================================================
    # LOAD EXISTING RESULTS
    # ============================================================
    
    all_results = {}
    out_path = Path(args.out)
    if out_path.exists():
        with open(out_path, "r") as f:
            all_results = json.load(f)
        print(f"✅ Loaded {len(all_results)} existing results")
    else:
        print(f"⚠️  No existing results found. Starting fresh.")
    
    print(f"\n{'='*60}")
    print(f"MODE: {mode}")
    print(f"Datasets: {datasets}")
    print(f"Shortcut types: {shortcut_types}")
    print(f"Correlation strengths: {strengths}")
    print(f"Models: {list(active_models.keys())}")
    print(f"CV folds: {n_folds}")
    print(f"{'='*60}")
    
    # ============================================================
    # RUN EXPERIMENTS
    # ============================================================
    
    t0 = time.time()
    total_computed = 0
    total_skipped = 0
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")
        
        X, y = load_dataset(dataset_name)
        
        # Apply optimizations for remaining mode
        if args.remaining:
            X, y = optimize_dataset(dataset_name, X, y)
        
        print(f"  Samples: {len(X)}, Features: {X.shape[1]}")
        
        for shortcut_type in shortcut_types:
            for r in strengths:
                injector = ShortcutInjector(correlation_strength=r)
                for model_name, model in active_models.items():
                    key = f"{dataset_name}__{shortcut_type}__{r}__{model_name}"
                    
                    # Skip if already computed
                    if key in all_results:
                        print(f"  ⏭️  SKIP {model_name:22s} {shortcut_type:22s} r={r:.2f}")
                        total_skipped += 1
                        continue
                    
                    # Run the experiment
                    result = run_full_experiment(
                        X, y, model, injector, 
                        shortcut_type=shortcut_type, 
                        n_folds=n_folds
                    )
                    cf = causal_fidelity_score(
                        result["condition_A"]["accuracy"]["mean"],
                        result["condition_B"]["accuracy"]["mean"],
                        result["condition_C"]["accuracy"]["mean"],
                        fold_results_B=result["condition_B"]["accuracy"]["values"],
                        fold_results_C=result["condition_C"]["accuracy"]["values"],
                    )
                    all_results[key] = {**result, "cf_score": cf}
                    total_computed += 1
                    
                    elapsed = time.time() - t0
                    print(f"  ✅ {model_name:22s} {shortcut_type:22s} r={r:.2f}  CF={cf['cf_score']:.3f}  [{elapsed/60:.1f} min]")
                    
                    # Save incrementally
                    with open(out_path, "w") as f:
                        json.dump(all_results, f, indent=2)
    
    # ============================================================
    # SUMMARY
    # ============================================================
    
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  ✅ Computed:  {total_computed} new results")
    print(f"  ⏭️  Skipped:   {total_skipped} existing results")
    print(f"  📊 Total:     {len(all_results)} results in all_results.json")
    print(f"  ⏱️  Time:      {elapsed / 60:.1f} minutes")
    print(f"  📁 Output:    {out_path}")
    
    # Show which datasets are now complete
    datasets_in_results = set()
    for key in all_results:
        datasets_in_results.add(key.split("__")[0])
    print(f"\n  📋 Datasets in results:")
    for d in sorted(datasets_in_results):
        count = sum(1 for k in all_results if k.startswith(d))
        print(f"     - {d}: {count} results")


if __name__ == "__main__":
    main()