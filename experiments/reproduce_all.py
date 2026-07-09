#!/usr/bin/env python
"""
reproduce_all.py
-----------------
Single-command reproduction of every ShortcutLens result.

    python experiments/reproduce_all.py [--quick] [--datasets d1,d2] [--out results.json]

By default this runs the full grid: 6 datasets x 5 shortcut types x
12 correlation strengths x ~9 classifiers x 5-fold CV. That is a lot
of model fits (thousands) and can take a few hours on a laptop CPU.

Use --quick for a fast smoke-test grid (2 datasets, 2 shortcut types,
3 correlation strengths, 5 folds) that finishes in a couple of
minutes and is what CI runs.
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

FULL_DATASETS = [
    "heart_disease", "adult_income", "credit_default",
    "mammography", "madelon", "bangla_dataset",
]
FULL_SHORTCUT_TYPES = [
    "label_proxy", "demographic_proxy", "temporal_shortcut",
    "selection_bias", "measurement_artifact",
]
FULL_CORRELATION_STRENGTHS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

QUICK_DATASETS = ["heart_disease", "mammography"]
QUICK_SHORTCUT_TYPES = ["label_proxy", "demographic_proxy"]
QUICK_CORRELATION_STRENGTHS = [0.0, 0.5, 0.9]


def parse_args():
    p = argparse.ArgumentParser(description="Reproduce ShortcutLens experiments.")
    p.add_argument("--quick", action="store_true", help="Run a fast smoke-test grid.")
    p.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset subset.")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--out", type=str, default="experiments/results/all_results.json")
    return p.parse_args()


def main():
    args = parse_args()

    datasets = QUICK_DATASETS if args.quick else FULL_DATASETS
    shortcut_types = QUICK_SHORTCUT_TYPES if args.quick else FULL_SHORTCUT_TYPES
    strengths = QUICK_CORRELATION_STRENGTHS if args.quick else FULL_CORRELATION_STRENGTHS

    if args.datasets:
        requested = set(args.datasets.split(","))
        datasets = [d for d in datasets if d in requested] or list(requested)

    models = get_model_suite()
    # Baselines don't have a meaningful shortcut-reliance story; skip
    # them in the main grid but keep them available for reference.
    active_models = {k: v for k, v in models.items() if k not in ("majority_class", "random")}

    all_results = {}
    t0 = time.time()

    for dataset_name in datasets:
        print(f"\n{'=' * 60}\nDataset: {dataset_name}\n{'=' * 60}")
        X, y = load_dataset(dataset_name)

        for shortcut_type in shortcut_types:
            for r in strengths:
                injector = ShortcutInjector(correlation_strength=r)
                for model_name, model in active_models.items():
                    result = run_full_experiment(
                        X, y, model, injector, shortcut_type=shortcut_type, n_folds=args.n_folds
                    )
                    cf = causal_fidelity_score(
                        result["condition_A"]["accuracy"]["mean"],
                        result["condition_B"]["accuracy"]["mean"],
                        result["condition_C"]["accuracy"]["mean"],
                        fold_results_B=result["condition_B"]["accuracy"]["values"],
                        fold_results_C=result["condition_C"]["accuracy"]["values"],
                    )
                    key = f"{dataset_name}__{shortcut_type}__{r}__{model_name}"
                    all_results[key] = {**result, "cf_score": cf}
                    print(f"  {model_name:22s} shortcut={shortcut_type:22s} r={r:.2f}  CF={cf['cf_score']:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone. {len(all_results)} configurations in {elapsed / 60:.1f} min.")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
