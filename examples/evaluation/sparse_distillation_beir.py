# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0
"""BEIR evaluation for a ColBERT model with a trained SparseProjection head.

Loads a model saved by ``examples/train/sparse_distillation.py`` (the
SparseProjection is part of ``colbert._modules``, so ``model.encode(...)``
already returns per-token sparse codes), encodes the BEIR corpus and queries,
runs exact sparse MaxSim on GPU, and reports nDCG / Recall / MAP.

This is intended as a correctness check for the sparse-distillation training
recipe — it does NOT use an ANN index (SEISMIC) or coarse-then-rerank pruning.
For scalable inference on large corpora, plug the encoded sparse vectors into a
sparse ANN index (e.g. SEISMIC) in a separate script.

Example
-------

::

    python examples/evaluation/sparse_distillation_beir.py \\
        --checkpoint_dir output/GTE-ModernColBERT-v1/SparseDistill-.../final \\
        --dataset_name scifact \\
        --k_retrieve 100
"""
from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from pylate import evaluation, models


QUERY_LENGTHS = {
    "quora": 32,
    "climate-fever": 64,
    "nq": 32,
    "msmarco": 32,
    "hotpotqa": 32,
    "nfcorpus": 32,
    "scifact": 48,
    "trec-covid": 48,
    "fiqa": 32,
    "arguana": 64,
    "scidocs": 48,
    "dbpedia-entity": 32,
    "webis-touche2020": 32,
    "fever": 32,
}


def encode_sparse(
    model: models.ColBERT,
    texts: list[str],
    is_query: bool,
    batch_size: int = 64,
) -> list[torch.Tensor]:
    """Encode a list of texts to per-token sparse codes.

    Because ``SparseProjection`` is appended to ``model._modules``, the
    standard ``model.encode`` pipeline already produces sparse codes — each
    returned tensor has shape ``(num_tokens, sparse_dim)`` with at most ``k``
    non-zero entries per row.
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        is_query=is_query,
        show_progress_bar=True,
        convert_to_tensor=True,
    )
    return [e.cpu() for e in embeddings]


def maxsim_retrieve(
    doc_embeddings: list[torch.Tensor],
    doc_ids: list[str],
    query_embeddings: list[torch.Tensor],
    k: int = 100,
    doc_chunk_size: int = 256,
    device: str | None = None,
) -> list[list[dict]]:
    """Exact sparse MaxSim retrieval on GPU.

    For each (query, document) pair, computes ``sum_i max_j <q_i, d_j>`` where
    inner products are taken on the (non-negative) sparse codes. Sparse codes
    are densified per-chunk on GPU; this is exact and fast for small/medium
    corpora but does not scale to millions of documents — use SEISMIC for that.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    sparse_dim = doc_embeddings[0].shape[-1]
    n_queries = len(query_embeddings)
    n_docs = len(doc_embeddings)

    # Pre-stack queries (small; usually fits in memory)
    q_lengths = [q.shape[0] for q in query_embeddings]
    max_q_tokens = max(q_lengths)
    queries_padded = torch.zeros(n_queries, max_q_tokens, sparse_dim)
    for qi, q in enumerate(query_embeddings):
        queries_padded[qi, : q.shape[0]] = q
    queries_padded = queries_padded.to(device_t)

    all_scores = torch.zeros(n_queries, n_docs)
    for chunk_start in tqdm(
        range(0, n_docs, doc_chunk_size), desc="Sparse MaxSim scoring"
    ):
        chunk_end = min(chunk_start + doc_chunk_size, n_docs)
        chunk_docs = doc_embeddings[chunk_start:chunk_end]
        chunk_len = chunk_end - chunk_start

        lengths = [d.shape[0] for d in chunk_docs]
        max_d_tokens = max(lengths)
        docs_padded = torch.zeros(chunk_len, max_d_tokens, sparse_dim, device=device_t)
        mask = torch.zeros(chunk_len, max_d_tokens, device=device_t, dtype=torch.bool)
        for i, d in enumerate(chunk_docs):
            n = d.shape[0]
            docs_padded[i, :n] = d.to(device_t)
            mask[i, :n] = True

        # sim[qi, qt, di, dt] = <q[qi,qt], d[di,dt]>; reduce dt with max, then sum over qt
        for qi in range(n_queries):
            # (max_q_tokens, sparse_dim) @ (sparse_dim, chunk_len*max_d_tokens)
            sim = torch.mm(
                queries_padded[qi],
                docs_padded.reshape(-1, sparse_dim).t(),
            ).reshape(max_q_tokens, chunk_len, max_d_tokens)
            sim = sim * mask.unsqueeze(0)
            sim = sim.max(dim=2).values.clamp(min=0)  # (max_q_tokens, chunk_len)
            # Mask out query padding tokens
            q_mask = torch.zeros(max_q_tokens, device=device_t)
            q_mask[: q_lengths[qi]] = 1.0
            sim = sim * q_mask.unsqueeze(1)
            all_scores[qi, chunk_start:chunk_end] = sim.sum(dim=0).cpu()

        del docs_padded, mask

    topk_scores, topk_idx = torch.topk(all_scores, k=min(k, n_docs), dim=1)
    results: list[list[dict]] = []
    for qi in range(n_queries):
        results.append(
            [
                {
                    "id": doc_ids[topk_idx[qi, j].item()],
                    "score": topk_scores[qi, j].item(),
                }
                for j in range(topk_idx.shape[1])
            ]
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BEIR evaluation for ColBERT + SparseProjection"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the model saved by examples/train/sparse_distillation.py",
    )
    parser.add_argument("--dataset_name", type=str, default="scifact")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--k_retrieve", type=int, default=100)
    parser.add_argument("--document_length", type=int, default=300)
    parser.add_argument(
        "--doc_chunk_size",
        type=int,
        default=256,
        help="Number of documents densified at a time on GPU.",
    )
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"BEIR sparse-distillation evaluation: {args.dataset_name}")
    print(f"{'=' * 60}")

    # --- Load model (backbone + appended SparseProjection) ---
    query_length = QUERY_LENGTHS.get(args.dataset_name, 32)
    model = models.ColBERT(
        args.checkpoint_dir,
        document_length=args.document_length,
        query_length=query_length,
    )

    # --- Load BEIR dataset ---
    split = "dev" if "msmarco" in args.dataset_name else "test"
    documents, queries, qrels = evaluation.load_beir(args.dataset_name, split=split)
    print(f"Loaded {args.dataset_name}: {len(documents)} docs, {len(queries)} queries")

    # --- Encode documents and queries (sparse codes via appended SparseProjection) ---
    print("\nEncoding documents...")
    doc_texts = [d["text"] for d in documents]
    doc_ids = [d["id"] for d in documents]
    doc_embeddings = encode_sparse(
        model, doc_texts, is_query=False, batch_size=args.batch_size
    )

    print("\nEncoding queries...")
    query_ids = list(queries.keys())
    query_texts = list(queries.values())
    query_embeddings = encode_sparse(
        model, query_texts, is_query=True, batch_size=args.batch_size
    )

    # --- Retrieve with exact sparse MaxSim ---
    print(f"\nExact sparse MaxSim retrieval (k={args.k_retrieve})...")
    scores = maxsim_retrieve(
        doc_embeddings,
        doc_ids,
        query_embeddings,
        k=args.k_retrieve,
        doc_chunk_size=args.doc_chunk_size,
    )

    # --- Evaluate ---
    results = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=query_ids,
        metrics=["ndcg@10", "ndcg@100", "recall@10", "recall@100", "map"],
    )

    print(f"\n{'=' * 60}")
    print(f"Results on {args.dataset_name}:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
