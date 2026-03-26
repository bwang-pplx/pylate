"""Evaluate a ColBERT model on BEIR datasets.

Supports standard MaxSim, MeanSim (A2-zs), token dropout (C2),
and post-hoc IDF pruning (M6/M7/D4/D5).

Usage:
    # Standard ColBERT (M5/A1)
    python scripts/ood_study/eval_colbert_beir.py --model output/ood_study/m5-colbert-seed1/final --dataset all

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


def compute_idf(documents: list[str], tokenizer) -> dict[int, float]:
    """Compute IDF scores for each token ID across the corpus."""
    doc_freq = Counter()
    n_docs = len(documents)

    for doc in documents:
        tokens = tokenizer(doc, truncation=True, max_length=256)
        unique_ids = set(tokens["input_ids"])
        for tid in unique_ids:
            doc_freq[tid] += 1

    idf = {}
    for tid, df in doc_freq.items():
        idf[tid] = np.log((n_docs + 1) / (df + 1))
    return idf


def prune_embeddings_by_idf(embeddings, input_ids, idf, keep_fraction):
    """Remove lowest-IDF token embeddings from each document."""
    pruned = []
    for emb, ids in zip(embeddings, input_ids):
        if isinstance(emb, np.ndarray):
            emb = torch.tensor(emb)
        n_tokens = len(emb)
        n_keep = max(1, int(n_tokens * keep_fraction))

        # Score each token by IDF
        token_scores = [idf.get(int(tid), 0.0) for tid in ids[:n_tokens]]
        keep_indices = sorted(range(len(token_scores)),
                              key=lambda i: token_scores[i], reverse=True)[:n_keep]
        keep_indices.sort()  # maintain order

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


def eval_dataset(model, dataset_name: str, args) -> dict:
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

    # Encode documents
    doc_embs = model.encode(
        doc_texts, batch_size=128, is_query=False, show_progress_bar=True
    )

    # IDF pruning
    if args.idf_prune > 0:
        print(f"  Applying IDF pruning (keep {args.idf_prune*100:.0f}%)...")
        tokenizer = model.tokenizer
        doc_input_ids = [
            tokenizer(text, truncation=True, max_length=256)["input_ids"]
            for text in doc_texts
        ]
        idf = compute_idf(doc_texts, tokenizer)
        doc_embs = prune_embeddings_by_idf(doc_embs, doc_input_ids, idf, args.idf_prune)

    # Token dropout
    if args.token_dropout > 0:
        print(f"  Applying token dropout ({args.token_dropout*100:.0f}%)...")
        doc_embs = apply_token_dropout(doc_embs, args.token_dropout)

    # Encode queries
    query_embs = model.encode(
        query_texts, batch_size=128, is_query=True, show_progress_bar=True
    )

    # Brute-force scoring (needed for MeanSim, pruned, or dropout variants)
    print("  Computing scores (brute-force)...")
    scores = []
    for i, q_emb in enumerate(query_embs):
        q_emb_t = torch.tensor(q_emb).unsqueeze(0) if isinstance(q_emb, np.ndarray) else q_emb.unsqueeze(0)

        query_scores = []
        # Score in batches to avoid OOM
        batch_size = 1024
        for j in range(0, len(doc_embs), batch_size):
            batch_doc_embs = doc_embs[j:j+batch_size]

            # Pad to same length for batch scoring
            max_len = max(len(e) for e in batch_doc_embs)
            dim = q_emb_t.shape[-1]
            padded = torch.zeros(len(batch_doc_embs), max_len, dim)
            mask = torch.zeros(len(batch_doc_embs), max_len)
            for k, emb in enumerate(batch_doc_embs):
                if isinstance(emb, np.ndarray):
                    emb = torch.tensor(emb)
                padded[k, :len(emb)] = emb
                mask[k, :len(emb)] = 1.0

            batch_scores = colbert_scores(
                q_emb_t, padded,
                documents_mask=mask,
                aggregation=args.aggregation,
            )[0]

            for k, s in enumerate(batch_scores.tolist()):
                doc_idx = j + k
                if doc_ids[doc_idx] != query_ids[i]:
                    query_scores.append({"id": doc_ids[doc_idx], "score": s})

        query_scores.sort(key=lambda x: x["score"], reverse=True)
        scores.append(query_scores[:100])

        if (i + 1) % 100 == 0:
            print(f"    Scored {i+1}/{len(query_embs)} queries")

    result = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=query_ids,
        metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
    )

    print(f"{dataset_name}: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate ColBERT on BEIR (with ablations)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, nargs="+", default=["nfcorpus"])
    parser.add_argument("--aggregation", type=str, default="max", choices=["max", "mean"])
    parser.add_argument("--token_dropout", type=float, default=0.0,
                        help="Fraction of doc tokens to randomly drop (C2)")
    parser.add_argument("--idf_prune", type=float, default=0.0,
                        help="Fraction of doc tokens to KEEP by IDF (e.g., 0.5 for M6)")
    parser.add_argument("--document_length", type=int, default=256)
    parser.add_argument("--results_suffix", type=str, default="",
                        help="Suffix for results directory (e.g., '_meansim', '_dropout30')")
    args = parser.parse_args()

    datasets = ALL_DATASETS if "all" in args.dataset else args.dataset

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
        result = eval_dataset(model, dataset_name, args)
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
