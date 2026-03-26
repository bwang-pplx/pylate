"""Evaluate a ColBERT model on BEIR datasets.

For standard MaxSim eval (no ablations), prefer using scripts/eval_beir.py
which uses PLAID indexing and is much faster.

This script uses brute-force scoring and supports ablation variants:
- MeanSim aggregation (A2-zs)
- Token dropout (C2)
- Post-hoc IDF pruning (M6/M7/D4/D5)

Usage:
    # A2-zs: MeanSim on MaxSim-trained model
    python scripts/ood_study/eval_colbert_beir.py --model output/ood_study/m5-colbert-seed1/final --aggregation mean --dataset all

    # C2: random 30% token dropout
    python scripts/ood_study/eval_colbert_beir.py --model output/ood_study/m5-colbert-seed1/final --token_dropout 0.3 --dataset all

    # M6: post-hoc 50% IDF pruning
    python scripts/ood_study/eval_colbert_beir.py --model output/ood_study/m5-colbert-seed1/final --idf_prune 0.5 --dataset all
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
import torch

from pylate import evaluation, models
from pylate.scores import colbert_scores

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


def compute_idf(documents: list[str], tokenizer, max_length: int = 256) -> dict[int, float]:
    """Compute IDF scores for each token ID across the corpus."""
    doc_freq = Counter()
    n_docs = len(documents)

    for doc in documents:
        tokens = tokenizer(doc, truncation=True, max_length=max_length)
        unique_ids = set(tokens["input_ids"])
        for tid in unique_ids:
            doc_freq[tid] += 1

    return {tid: np.log((n_docs + 1) / (df + 1)) for tid, df in doc_freq.items()}


def prune_embeddings_by_idf(embeddings, input_ids, idf, keep_fraction):
    """Remove lowest-IDF token embeddings from each document."""
    pruned = []
    for emb, ids in zip(embeddings, input_ids):
        if isinstance(emb, np.ndarray):
            emb = torch.tensor(emb)
        n_tokens = len(emb)
        n_keep = max(1, int(n_tokens * keep_fraction))

        token_scores = [idf.get(int(tid), 0.0) for tid in ids[:n_tokens]]
        keep_indices = sorted(
            sorted(range(len(token_scores)), key=lambda i: token_scores[i], reverse=True)[:n_keep]
        )
        pruned.append(emb[keep_indices])
    return pruned


def apply_token_dropout(embeddings, dropout_rate, seed=42):
    """Randomly drop a fraction of token embeddings from each document."""
    rng = np.random.RandomState(seed)
    dropped = []
    for emb in embeddings:
        if isinstance(emb, np.ndarray):
            emb = torch.tensor(emb)
        n_tokens = len(emb)
        n_keep = max(1, int(n_tokens * (1 - dropout_rate)))
        keep_indices = sorted(rng.choice(n_tokens, size=n_keep, replace=False))
        dropped.append(emb[keep_indices])
    return dropped


def brute_force_score(query_embs, doc_embs, doc_ids, query_ids, aggregation, device, doc_batch_size=512):
    """Score all query-doc pairs using brute-force ColBERT scoring on GPU."""
    scores = []
    n_queries = len(query_embs)

    for i, q_emb in enumerate(query_embs):
        if isinstance(q_emb, np.ndarray):
            q_emb = torch.tensor(q_emb)
        q_emb_t = q_emb.unsqueeze(0).to(device)

        query_scores = []
        for j in range(0, len(doc_embs), doc_batch_size):
            batch_embs = doc_embs[j:j + doc_batch_size]

            # Pad to same length
            max_len = max(len(e) for e in batch_embs)
            dim = q_emb_t.shape[-1]
            padded = torch.zeros(len(batch_embs), max_len, dim, device=device)
            mask = torch.zeros(len(batch_embs), max_len, device=device)
            for k, emb in enumerate(batch_embs):
                if isinstance(emb, np.ndarray):
                    emb = torch.tensor(emb)
                emb_len = len(emb)
                padded[k, :emb_len] = emb.to(device)
                mask[k, :emb_len] = 1.0

            batch_scores = colbert_scores(
                q_emb_t, padded,
                documents_mask=mask,
                aggregation=aggregation,
            )[0]

            for k, s in enumerate(batch_scores.tolist()):
                doc_idx = j + k
                if doc_ids[doc_idx] != query_ids[i]:
                    query_scores.append({"id": doc_ids[doc_idx], "score": s})

        query_scores.sort(key=lambda x: x["score"], reverse=True)
        scores.append(query_scores[:100])

        if (i + 1) % 50 == 0:
            print(f"    Scored {i+1}/{n_queries} queries")

    return scores


def eval_dataset(model, dataset_name: str, args, device) -> dict:
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
    doc_embs = model.encode(doc_texts, batch_size=128, is_query=False, show_progress_bar=True)
    query_embs = model.encode(query_texts, batch_size=128, is_query=True, show_progress_bar=True)

    # Apply post-hoc modifications
    if args.idf_prune > 0:
        print(f"  Applying IDF pruning (keep {args.idf_prune*100:.0f}%)...")
        tokenizer = model.tokenizer
        doc_input_ids = [
            tokenizer(text, truncation=True, max_length=args.document_length)["input_ids"]
            for text in doc_texts
        ]
        idf = compute_idf(doc_texts, tokenizer, max_length=args.document_length)
        doc_embs = prune_embeddings_by_idf(doc_embs, doc_input_ids, idf, args.idf_prune)

    if args.token_dropout > 0:
        print(f"  Applying token dropout ({args.token_dropout*100:.0f}%)...")
        doc_embs = apply_token_dropout(doc_embs, args.token_dropout)

    # Brute-force scoring on GPU
    print("  Computing scores (brute-force)...")
    result_scores = brute_force_score(
        query_embs, doc_embs, doc_ids, query_ids,
        aggregation=args.aggregation,
        device=device,
    )

    result = evaluation.evaluate(
        scores=result_scores,
        qrels=qrels,
        queries=query_ids,
        metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
    )

    print(f"{dataset_name}: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate ColBERT on BEIR (ablation variants)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, nargs="+", default=["nfcorpus"])
    parser.add_argument("--aggregation", type=str, default="max", choices=["max", "mean"])
    parser.add_argument("--token_dropout", type=float, default=0.0,
                        help="Fraction of doc tokens to randomly drop (C2)")
    parser.add_argument("--idf_prune", type=float, default=0.0,
                        help="Fraction of doc tokens to KEEP by IDF (e.g., 0.5 for M6)")
    parser.add_argument("--document_length", type=int, default=256)
    parser.add_argument("--results_suffix", type=str, default="")
    args = parser.parse_args()

    datasets = ALL_DATASETS if "all" in args.dataset else args.dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.ColBERT(
        model_name_or_path=args.model,
        document_length=args.document_length,
        trust_remote_code=True,
    )

    suffix = args.results_suffix
    if not suffix:
        if args.aggregation == "mean":
            suffix = "_meansim"
        elif args.token_dropout > 0:
            suffix = f"_dropout{int(args.token_dropout*100)}"
        elif args.idf_prune > 0:
            suffix = f"_prune{int(args.idf_prune*100)}"

    results_dir = f"{args.model}/beir_results{suffix}"
    os.makedirs(results_dir, exist_ok=True)

    all_results = {}
    for dataset_name in datasets:
        model.query_length = QUERY_LENGTHS.get(dataset_name, 32)
        result = eval_dataset(model, dataset_name, args, device)
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
