"""Evaluate a cross-encoder by reranking BM25 top-1000 on BEIR datasets.

Usage:
    python scripts/ood_study/eval_ce_beir.py --model output/ood_study/ce-modernbert-seed1/final --dataset all
"""

from __future__ import annotations

import argparse
import json
import os

from sentence_transformers.cross_encoder import CrossEncoder

from pylate import evaluation

ALL_DATASETS = [
    "nfcorpus", "fiqa", "scifact", "nq", "msmarco", "hotpotqa",
    "arguana", "quora", "scidocs", "dbpedia-entity", "fever",
    "climate-fever", "webis-touche2020", "trec-covid",
]


def get_bm25_top_k(documents, queries, qrels, dataset_name, k=1000):
    """Get BM25 top-k candidates using pyserini or rank_bm25."""
    from rank_bm25 import BM25Okapi

    doc_texts = [doc["text"] for doc in documents]
    doc_ids = [doc["id"] for doc in documents]

    # Tokenize for BM25
    tokenized_docs = [text.lower().split() for text in doc_texts]
    bm25 = BM25Okapi(tokenized_docs)

    query_texts = list(queries.values())
    query_ids = list(queries.keys())

    bm25_results = {}
    for qid, qtext in zip(query_ids, query_texts):
        tokenized_query = qtext.lower().split()
        scores = bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = scores.argsort()[-k:][::-1]
        bm25_results[qid] = [
            (doc_ids[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]

    return bm25_results, doc_texts, doc_ids


def eval_dataset(model, dataset_name: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Evaluating on {dataset_name}")
    print(f"{'='*60}")

    documents, queries, qrels = evaluation.load_beir(
        dataset_name=dataset_name,
        split="dev" if "msmarco" in dataset_name else "test",
    )

    doc_texts = {doc["id"]: doc["text"] for doc in documents}
    query_texts = queries
    query_ids = list(queries.keys())

    # Get BM25 top-1000
    print("  Running BM25...")
    bm25_results, _, _ = get_bm25_top_k(documents, queries, qrels, dataset_name, k=1000)

    # Rerank with cross-encoder
    print("  Reranking with cross-encoder...")
    scores = []
    for qid in query_ids:
        if qid not in bm25_results or len(bm25_results[qid]) == 0:
            scores.append([])
            continue

        candidates = bm25_results[qid]
        pairs = [(query_texts[qid], doc_texts[did]) for did, _ in candidates]

        ce_scores = model.predict(pairs, batch_size=256, show_progress_bar=False)

        query_scores = []
        for (did, _), score in zip(candidates, ce_scores):
            if did != qid:  # remove self-matches
                query_scores.append({"id": did, "score": float(score)})

        # Sort by CE score
        query_scores.sort(key=lambda x: x["score"], reverse=True)
        scores.append(query_scores[:100])

    result = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=query_ids,
        metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
    )

    # Also compute BM25 recall@1000 for context
    bm25_recall = 0
    total = 0
    for qid, rels in qrels.items():
        bm25_docs = {did for did, _ in bm25_results.get(qid, [])}
        for did, score in rels.items():
            if int(score) > 0:
                total += 1
                if did in bm25_docs:
                    bm25_recall += 1
    if total > 0:
        result["bm25_recall@1000"] = bm25_recall / total

    print(f"{dataset_name}: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate CE on BEIR (rerank BM25 top-1000)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, nargs="+", default=["nfcorpus"])
    args = parser.parse_args()

    datasets = ALL_DATASETS if "all" in args.dataset else args.dataset

    model = CrossEncoder(args.model)

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
        r1k = result.get("bm25_recall@1000", "N/A")
        print(f"  {name}: ndcg@10={result.get('ndcg@10', 'N/A')}, bm25_r@1000={r1k}")
    if len(all_results) > 1:
        avg = sum(r["ndcg@10"] for r in all_results.values()) / len(all_results)
        print(f"\n  Average ndcg@10: {avg:.4f}")
    print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()
