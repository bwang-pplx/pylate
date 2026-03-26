"""Collect all BEIR results into a single summary table.

Usage:
    python scripts/ood_study/collect_results.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

OUTPUT_DIR = "output/ood_study"

# Map model dirs to display names
MODELS = {
    # Core spectrum
    "m5-colbert": ("M5: ColBERT-Full", "colbert"),
    "m2-dense": ("M2: Dense", "dense"),
    "a2nat-meansim": ("A2-nat: MeanSim", "colbert"),
    "d2-dim64": ("D2: dim=64", "colbert"),
    "m5-colbert-hn7": ("M5-HN7: ColBERT", "colbert"),
    "m2-dense-hn7": ("M2-HN7: Dense", "dense"),
    "ce-modernbert": ("CE: Cross-Encoder", "ce"),
    # Inference-only variants (results stored in M5 dirs with suffix)
}

# Inference-only variants stored under m5 checkpoints
# M6/D4 are identical (50% IDF pruning on M5), M7/D5 are identical (25%)
INFERENCE_VARIANTS = {
    "a2zs-meansim": ("A2-zs: MeanSim (zero-shot)", "_meansim"),
    "c2-dropout30": ("C2: 30% dropout", "_dropout30"),
    "m6-prune50": ("M6/D4: pruned 50%", "_prune50"),
    "m7-prune75": ("M7/D5: pruned 75%", "_prune25"),
}

BEIR_DATASETS = [
    "nfcorpus", "fiqa", "scifact", "nq", "msmarco", "hotpotqa",
    "arguana", "quora", "scidocs", "dbpedia-entity", "fever",
    "climate-fever", "webis-touche2020", "trec-covid",
]


def load_results(results_path: str) -> dict | None:
    """Load all_results.json from a results directory."""
    path = os.path.join(results_path, "all_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def collect_all():
    all_data = {}

    # Trained models (3 seeds each, except CE with 1 seed)
    for model_prefix, (display_name, _) in MODELS.items():
        seeds = [1] if "ce-" in model_prefix else [1, 2, 3]

        for seed in seeds:
            model_dir = f"{model_prefix}-seed{seed}"
            results_path = os.path.join(OUTPUT_DIR, model_dir, "final", "beir_results")
            results = load_results(results_path)
            if results:
                key = f"{display_name} (s{seed})"
                all_data[key] = results

    # Inference-only variants (from M5 seeds)
    for variant_key, (display_name, suffix) in INFERENCE_VARIANTS.items():
        for seed in [1, 2, 3]:
            model_dir = f"m5-colbert-seed{seed}"
            results_path = os.path.join(
                OUTPUT_DIR, model_dir, "final", f"beir_results{suffix}"
            )
            results = load_results(results_path)
            if results:
                key = f"{display_name} (s{seed})"
                all_data[key] = results

    return all_data


def print_table(all_data: dict):
    if not all_data:
        print("No results found.")
        return

    # Header
    header = ["Model"] + BEIR_DATASETS + ["Average"]
    print("\t".join(header))

    for model_name, results in sorted(all_data.items()):
        row = [model_name]
        values = []
        for ds in BEIR_DATASETS:
            if ds in results and "ndcg@10" in results[ds]:
                val = results[ds]["ndcg@10"]
                row.append(f"{val:.4f}")
                values.append(val)
            else:
                row.append("-")

        if values:
            row.append(f"{sum(values)/len(values):.4f}")
        else:
            row.append("-")

        print("\t".join(row))


def save_csv(all_data: dict, path: str):
    """Save results as CSV."""
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Model"] + BEIR_DATASETS + ["Average"]
        writer.writerow(header)

        for model_name, results in sorted(all_data.items()):
            row = [model_name]
            values = []
            for ds in BEIR_DATASETS:
                if ds in results and "ndcg@10" in results[ds]:
                    val = results[ds]["ndcg@10"]
                    row.append(f"{val:.4f}")
                    values.append(val)
                else:
                    row.append("")

            if values:
                row.append(f"{sum(values)/len(values):.4f}")
            else:
                row.append("")

            writer.writerow(row)

    print(f"\nCSV saved to {path}")


def main():
    all_data = collect_all()

    print(f"\nFound results for {len(all_data)} model configurations.\n")
    print_table(all_data)

    csv_path = os.path.join(OUTPUT_DIR, "beir_summary.csv")
    save_csv(all_data, csv_path)


if __name__ == "__main__":
    main()
