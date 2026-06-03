# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0
"""Scalable BEIR eval for sparse ColBERT codes via a SEISMIC inverted index.

The exact-MaxSim eval (sparse_distillation_beir.py) brute-forces dense 16384-d
token tensors and only fits small corpora. SEISMIC is an approximate inverted
index for learned-sparse retrieval that scales to millions of documents -- but
it is **single-vector** per document. So we **max-pool** each document's token
sparse codes into one sparse document vector (SPLADE-style), index that, and
query with pooled query vectors. This is single-vector sparse retrieval (not
token-level MaxSim), the standard SEISMIC route.

Install: ``uv run --with pyseismic-lsr ...``
"""
from __future__ import annotations

import argparse

import torch
from seismic import SeismicDataset, SeismicIndex

from pylate import evaluation, models


def encode_pooled(model, texts, ids, is_query, batch_size, stream_chunk=2000):
    """Encode texts -> max-pooled per-item sparse vectors: (id, [dims], [values])."""
    pooled = []
    for start in range(0, len(texts), stream_chunk):
        embs = model.encode(
            texts[start : start + stream_chunk],
            batch_size=batch_size, is_query=is_query,
            convert_to_tensor=True, show_progress_bar=True,
        )
        for offset, e in enumerate(embs):
            vec = e.max(dim=0).values  # (sparse_dim,) max-pool over tokens
            nz = (vec > 0).nonzero(as_tuple=True)[0]
            pooled.append(
                (ids[start + offset], nz.cpu().tolist(), vec[nz].cpu().tolist())
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pooled


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--dataset_name", default="scifact")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--k_retrieve", type=int, default=100)
    p.add_argument("--document_length", type=int, default=300)
    p.add_argument("--query_cut", type=int, default=20)
    p.add_argument("--heap_factor", type=float, default=0.7)
    args = p.parse_args()

    split = "dev" if "msmarco" in args.dataset_name else "test"
    documents, queries, qrels = evaluation.load_beir(args.dataset_name, split=split)
    print(f"{args.dataset_name}: {len(documents)} docs, {len(queries)} queries", flush=True)

    model = models.ColBERT(
        args.checkpoint_dir, document_length=args.document_length, query_length=32
    )

    print("Encoding + pooling corpus...", flush=True)
    doc_ids = [d["id"] for d in documents]
    docs_pooled = encode_pooled(
        model, [d["text"] for d in documents], doc_ids, False, args.batch_size
    )

    # SEISMIC uses string component ids; dims -> str.
    dataset = SeismicDataset()
    for did, dims, vals in docs_pooled:
        dataset.add_document(str(did), [str(x) for x in dims], vals)
    print("Building SEISMIC index...", flush=True)
    index = SeismicIndex.build_from_dataset(dataset)
    print(f"  index: {index.len} docs, dim {index.dim}", flush=True)

    print("Encoding queries...", flush=True)
    qids = list(queries.keys())
    q_pooled = encode_pooled(model, [queries[q] for q in qids], qids, True, args.batch_size)

    results = []
    for qid, dims, vals in q_pooled:
        out = index.search(
            str(qid), [str(x) for x in dims], [float(v) for v in vals],
            args.k_retrieve, args.query_cut, args.heap_factor,
        )
        # out: list of (query_id, score, doc_id) tuples (sorted desc).
        ranked = [{"id": str(r[-1]), "score": float(r[1])} for r in out]
        results.append(ranked)

    metrics = evaluation.evaluate(
        scores=results, qrels=qrels, queries=qids,
        metrics=["ndcg@10", "ndcg@100", "recall@10", "recall@100", "map"],
    )
    print(f"\n===== SEISMIC (pooled single-vector) on {args.dataset_name} =====")
    for m, v in metrics.items():
        print(f"  {m}: {v:.4f}")


if __name__ == "__main__":
    main()
