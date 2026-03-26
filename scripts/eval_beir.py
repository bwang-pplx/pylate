"""Evaluate a trained ColBERT model on BEIR datasets.

Usage:
    # Single dataset
    python scripts/eval_beir.py --model output/pplx-colbert/final --dataset nfcorpus

    # Multiple datasets
    python scripts/eval_beir.py --model output/pplx-colbert/final --dataset nfcorpus fiqa scifact nq msmarco hotpotqa arguana

    # All standard BEIR datasets
    python scripts/eval_beir.py --model output/pplx-colbert/final --dataset all
"""

from __future__ import annotations

import argparse
import json

from pylate import evaluation, indexes, models, retrieve

ALL_DATASETS = [
    "nfcorpus", "fiqa", "scifact", "nq", "msmarco", "hotpotqa",
    "arguana", "quora", "scidocs", "dbpedia-entity", "fever",
    "climate-fever", "webis-touche2020", "trec-covid",
]

QUERY_LENGTHS = {
    "quora": 32, "climate-fever": 64, "nq": 32, "msmarco": 32,
    "hotpotqa": 32, "nfcorpus": 32, "scifact": 48, "trec-covid": 48,
    "fiqa": 32, "arguana": 64, "scidocs": 48, "dbpedia-entity": 32,
    "webis-touche2020": 32, "fever": 32,
}


def eval_dataset(model, dataset_name: str, args_model: str = "") -> dict:
    print(f"\n{'='*60}")
    print(f"Evaluating on {dataset_name}")
    print(f"{'='*60}")

    documents, queries, qrels = evaluation.load_beir(
        dataset_name=dataset_name,
        split="dev" if "msmarco" in dataset_name else "test",
    )

    model_short = args_model.rstrip("/").split("/")[-1]
    index_name = f"{dataset_name}_{model_short}"
    index = indexes.PLAID(
        override=True,
        index_folder=f"eval_indexes/{model_short}",
        index_name=index_name,
    )

    retriever = retrieve.ColBERT(index=index)

    documents_embeddings = model.encode(
        sentences=[doc["text"] for doc in documents],
        batch_size=2000,
        is_query=False,
        show_progress_bar=True,
    )

    index.add_documents(
        documents_ids=[doc["id"] for doc in documents],
        documents_embeddings=documents_embeddings,
    )

    queries_embeddings = model.encode(
        sentences=list(queries.values()),
        is_query=True,
        show_progress_bar=True,
        batch_size=32,
    )

    scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=100)

    # Remove self-matches
    for (query_id, _), query_scores in zip(queries.items(), scores):
        for score in query_scores:
            if score["id"] == query_id:
                query_scores.remove(score)

    result = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=list(queries.keys()),
        metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
    )

    print(f"{dataset_name}: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate ColBERT on BEIR")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--dataset", type=str, nargs="+", default=["nfcorpus"],
                        help="Dataset name(s), or 'all' for all BEIR datasets")
    parser.add_argument("--document_length", type=int, default=512)
    args = parser.parse_args()

    datasets = ALL_DATASETS if "all" in args.dataset else args.dataset

    model = models.ColBERT(
        model_name_or_path=args.model,
        document_length=args.document_length,
        trust_remote_code=True,
    )

    import os
    results_dir = f"{args.model}/beir_results"
    os.makedirs(results_dir, exist_ok=True)

    all_results = {}
    for dataset_name in datasets:
        model.query_length = QUERY_LENGTHS.get(dataset_name, 32)
        result = eval_dataset(model, dataset_name, args_model=args.model)
        all_results[dataset_name] = result

        # Save immediately after each dataset
        with open(f"{results_dir}/{dataset_name}.json", "w") as f:
            json.dump(result, f, indent=2)
        with open(f"{results_dir}/all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for name, result in all_results.items():
        ndcg10 = result.get("ndcg@10", "N/A")
        print(f"  {name}: ndcg@10={ndcg10}")

    if len(all_results) > 1:
        avg_ndcg = sum(r["ndcg@10"] for r in all_results.values()) / len(all_results)
        print(f"\n  Average ndcg@10: {avg_ndcg:.4f}")

    print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()
