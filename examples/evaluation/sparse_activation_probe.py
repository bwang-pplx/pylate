# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0
"""Activation-statistics probe for a trained SparseProjection head.

Encodes a sample of documents to sparse codes and reports whether the learned
sparse space is *spread* (a useful inverted index) or *collapsed* onto a small
fixed set of dimensions (dense retrieval in disguise). Optionally contrasts the
trained head against a fresh random-init head of the same architecture, to show
whether training spread or concentrated the activations.

Metrics
-------
- nnz/token: should be ~k (top-k sanity check).
- dims used / dead: how many of the V dims ever activate across the corpus.
- normalized activation entropy (0..1): 1.0 = perfectly uniform dimension usage,
  low = a few dimensions dominate (collapse).
- top-N concentration: share of all activations falling in the N busiest dims.
- posting-list length: docs per dimension (inverted-index cost); a dim active in
  most documents is non-selective.
"""
from __future__ import annotations

import argparse

import torch

from pylate import models


def probe(model: models.ColBERT, label: str, docs: list[str], batch_size: int) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    sparse_module = model[-1]
    vocab = sparse_module.out_features
    k = getattr(sparse_module, "k", None)

    embeddings = model.encode(
        docs,
        is_query=False,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True,
    )

    token_freq = torch.zeros(vocab)  # tokens activating each dim
    doc_freq = torch.zeros(vocab)  # docs activating each dim (posting-list length)
    nnz_per_token: list[torch.Tensor] = []
    total_tokens = 0
    for emb in embeddings:
        mask = (emb > 0).cpu()
        token_freq += mask.sum(dim=0).float()
        doc_freq += mask.any(dim=0).float()
        nnz_per_token.append(mask.sum(dim=1).float())
        total_tokens += mask.shape[0]

    nnz = torch.cat(nnz_per_token)
    used = int((token_freq > 0).sum())
    probs = token_freq / token_freq.sum()
    probs = probs[probs > 0]
    entropy = float(-(probs * probs.log()).sum())
    norm_entropy = entropy / float(torch.log(torch.tensor(float(vocab))))

    sorted_freq, _ = torch.sort(token_freq, descending=True)
    cum = torch.cumsum(sorted_freq, dim=0) / sorted_freq.sum()

    def top_share(n: int) -> float:
        return float(cum[min(n, vocab) - 1]) * 100.0

    used_pl = doc_freq[doc_freq > 0]
    n_docs = len(docs)

    print(f"\n===== {label}  (V={vocab}, k={k}) =====")
    print(f"docs={n_docs}  tokens={total_tokens}")
    print(f"nnz/token: mean={nnz.mean():.1f} max={int(nnz.max())} (expect <= k={k})")
    print(
        f"dims used across corpus: {used}/{vocab} "
        f"({100 * used / vocab:.1f}%)  dead={vocab - used}"
    )
    print(f"activation entropy (normalized 0..1): {norm_entropy:.3f}  (1=uniform)")
    print(
        f"activation share  top-128={top_share(128):.1f}%  "
        f"top-512={top_share(512):.1f}%  top-1024={top_share(1024):.1f}%"
    )
    print(
        f"posting-list length (docs/dim): max={int(doc_freq.max())} "
        f"({100 * float(doc_freq.max()) / n_docs:.0f}% of docs)  "
        f"median={int(used_pl.median())}  mean={used_pl.mean():.1f}"
    )
    print(
        f"non-selective dims (active in >50% of docs): "
        f"{int((doc_freq > 0.5 * n_docs).sum())}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparse activation-stats probe")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="sentence-transformers/msmarco-bm25")
    parser.add_argument("--dataset_config", type=str, default="triplet")
    parser.add_argument("--num_docs", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--baseline_model",
        type=str,
        default="lightonai/GTE-ModernColBERT-v1",
        help="Backbone for the random-init contrast (must match the trained one).",
    )
    parser.add_argument("--skip_random", action="store_true")
    args = parser.parse_args()

    from datasets import load_dataset

    rows = load_dataset(
        args.dataset, args.dataset_config, split=f"train[:{args.num_docs}]"
    )
    docs = [row["positive"] for row in rows]

    trained = models.ColBERT(args.checkpoint, device="cpu")
    probe(trained, f"TRAINED {args.checkpoint}", docs, args.batch_size)

    if not args.skip_random:
        random_model = models.ColBERT(args.baseline_model, device="cpu")
        dim = random_model[-1].out_features
        random_model.append(
            models.SparseProjection(
                in_features=dim,
                out_features=trained[-1].out_features,
                k=trained[-1].k,
            )
        )
        probe(random_model, "RANDOM init (same arch)", docs, args.batch_size)


if __name__ == "__main__":
    main()
