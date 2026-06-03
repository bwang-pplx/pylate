# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0
"""Build an untrained *centered* LSH head on a ColBERT backbone.

Centering the dense token embeddings before the random projection removes the
anisotropic common component, exposing the discriminative residual to the top-k
(predicted to recover LSH quality on anisotropic encoders like pplx-embed).

Trick: top-k(ReLU((x-mu) Wᵀ)) = top-k(ReLU(x Wᵀ + b)) with b = -W·mu, so a
centered LSH head is just the orthogonal SparseProjection with bias = -W·mu,
where mu is the mean of the (pre-normalization) dense token embeddings.
"""
from __future__ import annotations

import argparse

import torch
from datasets import load_dataset

from pylate import models


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--sparse_dim", type=int, default=16384)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--num_docs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--anchor", choices=["centered", "whitened"], default="centered")
    p.add_argument("--trust_remote_code", action="store_true", default=False)
    args = p.parse_args()

    model = models.ColBERT(args.model, trust_remote_code=args.trust_remote_code)
    dim = model[-1].out_features
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    transformer, dense = model[0], model[1]

    rows = load_dataset(
        "sentence-transformers/msmarco-bm25", "triplet", split=f"train[:{args.num_docs}]"
    )
    docs = [r["positive"] for r in rows]

    # Collect pre-normalization dense token embeddings (what the sparse head sees).
    collected = []
    for start in range(0, len(docs), args.batch_size):
        feats = model.tokenize(docs[start : start + args.batch_size], is_query=False)
        feats = {k: v.to(device) for k, v in feats.items()}
        with torch.no_grad():
            feats = dense(transformer(feats))
        mask = feats["attention_mask"].bool()
        collected.append(feats["token_embeddings"][mask].float().cpu())
    tokens = torch.cat(collected, dim=0)
    mean = tokens.mean(0)

    # Orthogonal anchors; optionally compose with a whitening transform.
    anchors = torch.empty(args.sparse_dim, dim)
    torch.nn.init.orthogonal_(anchors)
    if args.anchor == "whitened":
        centered = tokens - mean
        cov = (centered.t() @ centered) / centered.shape[0] + 1e-3 * torch.eye(dim)
        evals, evecs = torch.linalg.eigh(cov)
        whiten = evecs @ torch.diag(evals.clamp_min(1e-6).rsqrt()) @ evecs.t()
        weight = anchors @ whiten          # ReLU((x-mu) Σ^{-1/2} Aᵀ)
    else:  # centered
        weight = anchors                   # ReLU((x-mu) Aᵀ)

    sparse = models.SparseProjection(
        in_features=dim, out_features=args.sparse_dim, k=args.k, bias=True
    )
    with torch.no_grad():
        sparse.linear.weight.copy_(weight)
        sparse.linear.bias.copy_(-(weight @ mean.to(weight.dtype)))
    model.append(sparse)
    model.save_pretrained(args.output)
    print(
        f"saved {args.anchor} LSH -> {args.output} | mean_norm={float(mean.norm()):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
