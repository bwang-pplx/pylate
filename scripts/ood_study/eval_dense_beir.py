"""Evaluate a dense (sentence-transformers) model on BEIR datasets via MTEB.

Usage:
    python scripts/ood_study/eval_dense_beir.py --model output/ood_study/m2-dense-seed1/final --dataset all
"""

from __future__ import annotations

import argparse
import json
import os

import mteb
from sentence_transformers import SentenceTransformer

ALL_DATASETS = [
    "NFCorpus", "FiQA2018", "SciFact", "NQ", "MSMARCO", "HotpotQA",
    "ArguAna", "QuoraRetrieval", "SCIDOCS", "DBPedia", "FEVER",
    "ClimateFEVER", "Touche2020", "TRECCOVID",
]

# Map MTEB names to short names for output
SHORT_NAMES = {
    "NFCorpus": "nfcorpus", "FiQA2018": "fiqa", "SciFact": "scifact",
    "NQ": "nq", "MSMARCO": "msmarco", "HotpotQA": "hotpotqa",
    "ArguAna": "arguana", "QuoraRetrieval": "quora", "SCIDOCS": "scidocs",
    "DBPedia": "dbpedia-entity", "FEVER": "fever",
    "ClimateFEVER": "climate-fever", "Touche2020": "webis-touche2020",
    "TRECCOVID": "trec-covid",
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate dense model on BEIR via MTEB")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, nargs="+", default=["NFCorpus"])
    args = parser.parse_args()

    datasets = ALL_DATASETS if "all" in args.dataset else args.dataset

    model = SentenceTransformer(args.model)

    results_dir = os.path.join(args.model, "beir_results")
    os.makedirs(results_dir, exist_ok=True)

    all_results = {}
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset_name}")
        print(f"{'='*60}")

        tasks = mteb.get_tasks(tasks=[dataset_name])
        eval_results = mteb.MTEB(tasks=tasks).run(
            model,
            output_folder=os.path.join(results_dir, "mteb_raw"),
            eval_splits=["dev"] if dataset_name == "MSMARCO" else ["test"],
        )

        # Extract nDCG@10 from MTEB results
        for r in eval_results:
            split = "dev" if dataset_name == "MSMARCO" else "test"
            scores = r.scores.get(split, [{}])[0]
            short = SHORT_NAMES.get(dataset_name, dataset_name)
            result = {
                "ndcg@10": scores.get("ndcg_at_10", None),
                "ndcg@100": scores.get("ndcg_at_100", None),
                "recall@10": scores.get("recall_at_10", None),
                "recall@100": scores.get("recall_at_100", None),
                "map": scores.get("map_at_100", None),
            }
            all_results[short] = result

            with open(os.path.join(results_dir, f"{short}.json"), "w") as f:
                json.dump(result, f, indent=2)

    with open(os.path.join(results_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for name, result in all_results.items():
        print(f"  {name}: ndcg@10={result.get('ndcg@10', 'N/A')}")
    if len(all_results) > 1:
        vals = [r["ndcg@10"] for r in all_results.values() if r.get("ndcg@10")]
        if vals:
            print(f"\n  Average ndcg@10: {sum(vals)/len(vals):.4f}")
    print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()
