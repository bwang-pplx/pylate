# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0
"""Quick NanoBEIR eval (+ optional activation-spread probe) for a model.

Two uses:
  * Dense teacher ceiling: ``--sparse_dim 0`` evaluates the bare ColBERT model.
  * Untrained sparse baseline: ``--sparse_dim 16384 --init orthogonal`` appends a
    fresh (untrained) SparseProjection and reports both NanoBEIR and how spread
    its activations are -- a fixed LSH/random-projection-style baseline.
"""
from __future__ import annotations

import argparse

import torch

from pylate import evaluation, models


class _ChunkedNanoBEIR(evaluation.NanoBEIREvaluator):
    """NanoBEIR with a small corpus_chunk_size so high-dim sparse codes don't OOM."""

    def __init__(self, *args, corpus_chunk_size: int = 256, **kwargs) -> None:
        self._corpus_chunk_size = corpus_chunk_size
        super().__init__(*args, **kwargs)

    def _load_dataset(self, dataset_name, **kwargs):
        kwargs.setdefault("corpus_chunk_size", self._corpus_chunk_size)
        return super()._load_dataset(dataset_name, **kwargs)


def activation_spread(model, docs, batch_size) -> None:
    vocab = model[-1].out_features
    embeddings = model.encode(
        docs, is_query=False, batch_size=batch_size, convert_to_tensor=True,
        show_progress_bar=False,
    )
    token_freq = torch.zeros(vocab)
    doc_freq = torch.zeros(vocab)
    for emb in embeddings:
        mask = (emb > 0).cpu()
        token_freq += mask.sum(dim=0).float()
        doc_freq += mask.any(dim=0).float()
    used = int((token_freq > 0).sum())
    sorted_freq, _ = torch.sort(token_freq, descending=True)
    cum = torch.cumsum(sorted_freq, 0) / sorted_freq.sum()
    used_pl = doc_freq[doc_freq > 0]
    print(
        f"  spread: dims_used={used}/{vocab} ({100 * used / vocab:.1f}%)  "
        f"top128_share={float(cum[min(128, vocab) - 1]) * 100:.1f}%  "
        f"posting_median={int(used_pl.median())}  posting_max={int(doc_freq.max())}/"
        f"{len(docs)}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="lightonai/GTE-ModernColBERT-v1")
    p.add_argument("--sparse_dim", type=int, default=0, help="0 = dense (no sparse head)")
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--init", type=str, default="default", choices=["default", "orthogonal"])
    p.add_argument("--corpus_chunk_size", type=int, default=256)
    p.add_argument("--probe_docs", type=int, default=1000)
    args = p.parse_args()

    model = models.ColBERT(args.model)
    label = f"DENSE {args.model}"
    if args.sparse_dim > 0:
        dim = model[-1].out_features
        model.append(
            models.SparseProjection(
                in_features=dim, out_features=args.sparse_dim, k=args.k, init=args.init
            )
        )
        label = f"SPARSE(untrained, init={args.init}, dim={args.sparse_dim}, k={args.k})"

    evaluator = _ChunkedNanoBEIR(corpus_chunk_size=args.corpus_chunk_size)
    scores = evaluator(model)
    mean = scores.get("NanoBEIR_mean_MaxSim_ndcg@10") or scores.get(
        "NanoBEIR_mean_cosine_ndcg@10"
    )
    print(f"\n===== {label} =====")
    print(f"NanoBEIR_mean_ndcg@10 = {mean}")

    if args.sparse_dim > 0:
        from datasets import load_dataset

        rows = load_dataset(
            "sentence-transformers/msmarco-bm25", "triplet",
            split=f"train[:{args.probe_docs}]",
        )
        activation_spread(model, [r["positive"] for r in rows], batch_size=64)


if __name__ == "__main__":
    main()
