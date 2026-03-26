"""Evaluate a dense (sentence-transformers) model on BEIR datasets.

Uses brute-force cosine similarity — no index needed.

Usage:
    python scripts/ood_study/eval_dense_beir.py --model output/ood_study/m2-dense-seed1/final --dataset all
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from sentence_transformers import SentenceTransformer

from pylate import evaluation

ALL_DATASETS = [
    "nfcorpus", "fiqa", "scifact", "nq", "msmarco", "hotpotqa",
    "arguana", "quora", "scidocs", "dbpedia-entity", "fever",
    "climate-fever", "webis-touche2020", "trec-covid",
]


def eval_dataset(model, dataset_name: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Evaluating on {dataset_name}")
    print(f"{'='*60}")

    documents, queries, qrels = evaluation.load_beir(
        dataset_name=dataset_name,
        split="dev" if "msmarco" in dataset_name else "test",
    )

    doc_texts = [doc["text"] for doc in documents]
    doc_ids = [doc["id"] for doc in documents]
    query_texts = list(queries.values())
    query_ids = list(queries.keys())

    # Encode
    doc_embs = model.encode(doc_texts, batch_size=512, show_progress_bar=True,
                            convert_to_tensor=True, normalize_embeddings=True)
    query_embs = model.encode(query_texts, batch_size=512, show_progress_bar=True,
                              convert_to_tensor=True, normalize_embeddings=True)

    # Brute-force cosine similarity, get top-100
    sim = query_embs @ doc_embs.T
    topk = torch.topk(sim, k=min(100, len(doc_ids)), dim=1)

    scores = []
    for i in range(len(query_texts)):
        query_scores = []
        for score, idx in zip(topk.values[i].tolist(), topk.indices[i].tolist()):
            if doc_ids[idx] != query_ids[i]:  # remove self-matches
                query_scores.append({"id": doc_ids[idx], "score": score})
        scores.append(query_scores)

    result = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=query_ids,
        metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
    )

    print(f"{dataset_name}: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate dense model on BEIR")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, nargs="+", default=["nfcorpus"])
    args = parser.parse_args()

    datasets = ALL_DATASETS if "all" in args.dataset else args.dataset

    model = SentenceTransformer(args.model)

    results_dir = f"{args.model}/beir_results"
    os.makedirs(results_dir, exist_ok=True)

    all_results = {}
    for dataset_name in datasets:
        result = eval_dataset(model, dataset_name)
        all_results[dataset_name] = result

        with open(f"{results_dir}/{dataset_name}.json", "w") as f:
            json.dump(result, f, indent=2)
        with open(f"{results_dir}/all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for name, result in all_results.items():
        print(f"  {name}: ndcg@10={result.get('ndcg@10', 'N/A')}")
    if len(all_results) > 1:
        avg = sum(r["ndcg@10"] for r in all_results.values()) / len(all_results)
        print(f"\n  Average ndcg@10: {avg:.4f}")
    print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()
