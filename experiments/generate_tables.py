#!/usr/bin/env python
"""
generate_tables.py
------------------
Generate summary tables from all_results.json for the paper.

Usage:
    python experiments/generate_tables.py
    python experiments/generate_tables.py --csv
    python experiments/generate_tables.py --output paper/tables/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_results(results_path: str = "experiments/results/all_results.json"):
    with open(results_path, "r") as f:
        return json.load(f)


def infer_axes(all_results):
    datasets, shortcut_types, strengths, models = set(), set(), set(), set()
    for key in all_results:
        dataset_name, shortcut_type, r, model_name = key.split("__")
        datasets.add(dataset_name)
        shortcut_types.add(shortcut_type)
        strengths.add(float(r))
        models.add(model_name)
    return sorted(datasets), sorted(shortcut_types), sorted(strengths), sorted(models)


def generate_cf_table(all_results):
    """Generate CF score table by dataset and shortcut type (at r=0.5)."""
    datasets, shortcut_types, strengths, models = infer_axes(all_results)
    
    r_target = 0.5
    
    print("\n" + "="*100)
    print(f"TABLE 1: Causal Fidelity Scores at r={r_target} by Dataset and Shortcut Type")
    print("="*100)
    
    header = f"{'Dataset':<20} {'Shortcut':<22} " + " ".join(f"{m[:12]:<12}" for m in ["Logistic", "RandomForest", "GradientBoost", "KNN", "SVM", "MLP", "DecisionTree"])
    print(header)
    print("-"*100)
    
    for ds in datasets:
        for st in shortcut_types:
            row = f"{ds:<20} {st:<22} "
            for model in ["logistic_regression", "random_forest", "gradient_boosting", "knn_5", "svm_rbf", "mlp", "decision_tree"]:
                key = f"{ds}__{st}__{r_target}__{model}"
                if key in all_results:
                    cf = all_results[key].get("cf_score", {})
                    if isinstance(cf, dict):
                        cf_val = cf.get("cf_score", 0)
                    else:
                        cf_val = cf
                    row += f"{cf_val:<12.3f}"
                else:
                    row += f"{'---':<12}"
            print(row)
        print("-"*100)


def generate_model_ranking_table(all_results):
    """Generate model ranking by robustness (average CF across all shortcuts)."""
    datasets, shortcut_types, strengths, models = infer_axes(all_results)
    
    model_scores = {m: [] for m in models}
    
    for key, value in all_results.items():
        parts = key.split("__")
        if len(parts) == 4:
            ds, st, r, model = parts
            r_float = float(r)
            if r_float >= 0.5:
                cf = value.get("cf_score", {})
                if isinstance(cf, dict):
                    cf_val = cf.get("cf_score", 0)
                else:
                    cf_val = cf
                model_scores[model].append(cf_val)
    
    print("\n" + "="*80)
    print("TABLE 2: Model Robustness Ranking (avg CF across r>=0.5)")
    print("="*80)
    
    avg_scores = {}
    for model, scores in model_scores.items():
        if scores:
            avg_scores[model] = sum(scores) / len(scores)
    
    sorted_models = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'Model':<25} {'Avg CF (r>=0.5)':<20} {'Rank':<10}")
    print("-"*55)
    for i, (model, avg) in enumerate(sorted_models, 1):
        print(f"{model:<25} {avg:<20.4f} {i:<10}")
    print("="*80)
    
    return sorted_models


def generate_shortcut_severity_table(all_results):
    """Rank shortcut types by severity."""
    datasets, shortcut_types, strengths, models = infer_axes(all_results)
    
    shortcut_scores = {st: [] for st in shortcut_types}
    
    for key, value in all_results.items():
        parts = key.split("__")
        if len(parts) == 4:
            ds, st, r, model = parts
            r_float = float(r)
            if r_float >= 0.5:
                cf = value.get("cf_score", {})
                if isinstance(cf, dict):
                    cf_val = cf.get("cf_score", 0)
                else:
                    cf_val = cf
                shortcut_scores[st].append(cf_val)
    
    print("\n" + "="*80)
    print("TABLE 3: Shortcut Type Severity Ranking")
    print("="*80)
    
    avg_scores = {}
    for st, scores in shortcut_scores.items():
        if scores:
            avg_scores[st] = sum(scores) / len(scores)
    
    sorted_st = sorted(avg_scores.items(), key=lambda x: x[1])
    
    severity_labels = ["Most Severe", "Severe", "Moderate", "Benign", "Most Benign"]
    
    print(f"{'Shortcut Type':<25} {'Avg CF (r>=0.5)':<20} {'Severity':<15}")
    print("-"*60)
    for i, (st, avg) in enumerate(sorted_st, 1):
        severity = severity_labels[i-1] if i <= len(severity_labels) else ""
        print(f"{st:<25} {avg:<20.4f} {severity:<15}")
    print("="*80)
    
    return sorted_st


def generate_dataset_vulnerability_table(all_results):
    """Rank datasets by vulnerability."""
    datasets, shortcut_types, strengths, models = infer_axes(all_results)
    
    dataset_scores = {ds: [] for ds in datasets}
    
    for key, value in all_results.items():
        parts = key.split("__")
        if len(parts) == 4:
            ds, st, r, model = parts
            r_float = float(r)
            if r_float >= 0.5:
                cf = value.get("cf_score", {})
                if isinstance(cf, dict):
                    cf_val = cf.get("cf_score", 0)
                else:
                    cf_val = cf
                dataset_scores[ds].append(cf_val)
    
    print("\n" + "="*80)
    print("TABLE 4: Dataset Vulnerability Ranking (avg CF across r>=0.5)")
    print("="*80)
    
    avg_scores = {}
    for ds, scores in dataset_scores.items():
        if scores:
            avg_scores[ds] = sum(scores) / len(scores)
    
    sorted_ds = sorted(avg_scores.items(), key=lambda x: x[1])
    
    vuln_labels = ["Most Vulnerable", "Vulnerable", "Moderate", "Robust", "Most Robust"]
    
    print(f"{'Dataset':<20} {'Avg CF (r>=0.5)':<20} {'Vulnerability':<15}")
    print("-"*55)
    for i, (ds, avg) in enumerate(sorted_ds, 1):
        vuln = vuln_labels[i-1] if i <= len(vuln_labels) else ""
        print(f"{ds:<20} {avg:<20.4f} {vuln:<15}")
    print("="*80)
    
    return sorted_ds


def generate_phase_transition_table(all_results):
    """Find the correlation strength where CF drops below 0.7 for each dataset-shortcut."""
    datasets, shortcut_types, strengths, models = infer_axes(all_results)
    strengths_sorted = sorted(strengths)
    
    print("\n" + "="*80)
    print("TABLE 5: Phase Transition Thresholds (CF < 0.7)")
    print("="*80)
    
    # Use random_forest as the representative model (or best available)
    best_model = "random_forest"
    if best_model not in models:
        best_model = list(models)[0] if models else "logistic_regression"
    
    print(f"{'Dataset':<20} {'Shortcut':<22} {'Threshold (r)':<15} {'CF at threshold':<15}")
    print("-"*75)
    
    results = []
    for ds in datasets:
        for st in shortcut_types:
            threshold = None
            cf_at_threshold = None
            
            for r in strengths_sorted:
                key = f"{ds}__{st}__{r}__{best_model}"
                if key in all_results:
                    cf = all_results[key].get("cf_score", {})
                    if isinstance(cf, dict):
                        cf_val = cf.get("cf_score", 0)
                    else:
                        cf_val = cf
                    
                    if cf_val < 0.7:
                        threshold = r
                        cf_at_threshold = cf_val
                        break
            
            if threshold is not None:
                print(f"{ds:<20} {st:<22} {threshold:<15.2f} {cf_at_threshold:<15.3f}")
                results.append({"dataset": ds, "shortcut": st, "threshold": threshold, "cf": cf_at_threshold})
            else:
                print(f"{ds:<20} {st:<22} {'>0.99':<15} {'---':<15}")
                results.append({"dataset": ds, "shortcut": st, "threshold": ">0.99", "cf": "---"})
        print("-"*75)
    
    return results


def generate_summary_statistics(all_results):
    """Generate summary statistics for each dataset."""
    datasets, shortcut_types, strengths, models = infer_axes(all_results)
    
    print("\n" + "="*80)
    print("TABLE 6: Dataset Summary Statistics")
    print("="*80)
    
    print(f"{'Dataset':<20} {'# Results':<12} {'Models':<12} {'Shortcuts':<12} {'Strengths':<12}")
    print("-"*70)
    
    stats = []
    for ds in datasets:
        count = sum(1 for k in all_results if k.startswith(ds))
        models_ds = set()
        shortcuts_ds = set()
        strengths_ds = set()
        for key in all_results:
            if key.startswith(ds):
                parts = key.split("__")
                if len(parts) == 4:
                    models_ds.add(parts[3])
                    shortcuts_ds.add(parts[1])
                    strengths_ds.add(float(parts[2]))
        print(f"{ds:<20} {count:<12} {len(models_ds):<12} {len(shortcuts_ds):<12} {len(strengths_ds):<12}")
        stats.append({"dataset": ds, "results": count, "models": len(models_ds), "shortcuts": len(shortcuts_ds), "strengths": len(strengths_ds)})
    print("="*80)
    
    return stats


def export_tables_to_csv(all_results, output_dir="paper/tables"):
    """Export all tables as CSV files for Excel/paper."""
    datasets, shortcut_types, strengths, models = infer_axes(all_results)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Table 1: CF scores at r=0.5
    r_target = 0.5
    with open(output_dir / "table1_cf_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Dataset", "Shortcut"] + sorted(models)
        writer.writerow(header)
        
        for ds in datasets:
            for st in shortcut_types:
                row = [ds, st]
                for model in sorted(models):
                    key = f"{ds}__{st}__{r_target}__{model}"
                    if key in all_results:
                        cf = all_results[key].get("cf_score", {})
                        cf_val = cf.get("cf_score", 0) if isinstance(cf, dict) else cf
                        row.append(f"{cf_val:.3f}")
                    else:
                        row.append("")
                writer.writerow(row)
    
    # Table 2: Model ranking
    with open(output_dir / "table2_model_ranking.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Avg_CF_r_ge_0.5", "Rank"])
        
        model_scores = {m: [] for m in models}
        for key, value in all_results.items():
            parts = key.split("__")
            if len(parts) == 4:
                ds, st, r, model = parts
                r_float = float(r)
                if r_float >= 0.5:
                    cf = value.get("cf_score", {})
                    cf_val = cf.get("cf_score", 0) if isinstance(cf, dict) else cf
                    model_scores[model].append(cf_val)
        
        avg_scores = {}
        for model, scores in model_scores.items():
            if scores:
                avg_scores[model] = sum(scores) / len(scores)
        
        sorted_models = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (model, avg) in enumerate(sorted_models, 1):
            writer.writerow([model, f"{avg:.4f}", i])
    
    print(f"✅ Tables exported to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Generate tables from all_results.json")
    parser.add_argument("--csv", action="store_true", help="Export tables as CSV files")
    parser.add_argument("--output", type=str, default="paper/tables", help="Output directory for CSV files")
    parser.add_argument("--results", type=str, default="experiments/results/all_results.json", help="Path to results JSON")
    args = parser.parse_args()
    
    # Load results
    all_results = load_results(args.results)
    print(f"✅ Loaded {len(all_results)} results from {args.results}")
    
    # Generate all tables
    generate_cf_table(all_results)
    generate_model_ranking_table(all_results)
    generate_shortcut_severity_table(all_results)
    generate_dataset_vulnerability_table(all_results)
    generate_phase_transition_table(all_results)
    generate_summary_statistics(all_results)
    
    # Export CSV if requested
    if args.csv:
        export_tables_to_csv(all_results, args.output)


if __name__ == "__main__":
    main()