#!/usr/bin/env python
"""
generate_figures.py
--------------------
Regenerates every figure used in the paper from a results JSON file
produced by ``reproduce_all.py``.

    python experiments/generate_figures.py --results experiments/results/all_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shortcut_lens.visualizers import plot_cf_curves, plot_phase_transition_heatmap


def infer_axes(all_results: dict):
    datasets, shortcut_types, strengths, models = set(), set(), set(), set()
    for key in all_results:
        dataset_name, shortcut_type, r, model_name = key.split("__")
        datasets.add(dataset_name)
        shortcut_types.add(shortcut_type)
        strengths.add(float(r))
        models.add(model_name)
    return sorted(datasets), sorted(shortcut_types), sorted(strengths), sorted(models)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="experiments/results/all_results.json")
    p.add_argument("--out-dir", default="paper/figures")
    args = p.parse_args()

    with open(args.results) as f:
        all_results = json.load(f)

    datasets, shortcut_types, strengths, models = infer_axes(all_results)

    for shortcut_type in shortcut_types:
        for dataset_name in datasets:
            path = plot_cf_curves(
                all_results, dataset_name, shortcut_type, strengths, models, out_dir=args.out_dir
            )
            print(f"Wrote {path}")
        heatmap_path = plot_phase_transition_heatmap(
            all_results, shortcut_type, datasets, models, strengths, out_dir=args.out_dir
        )
        print(f"Wrote {heatmap_path}")


if __name__ == "__main__":
    main()
