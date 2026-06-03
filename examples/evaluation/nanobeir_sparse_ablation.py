# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0
"""Training-free sparsification ablation on NanoBEIR (token-level MaxSim).

Builds an untrained sparse head on a frozen ColBERT with different ANCHOR
strategies and measures NanoBEIR mean nDCG@10. All anchors are training-free
(stats only, no backprop):

  * orthogonal : random semi-orthogonal projection (the LSH control).
  * whitened   : center + whiten the dense tokens, then random orthogonal
                 projection — equalizes directions so top-k captures the
                 discriminative residual, not the anisotropic common component.
  * data       : use a random sample of real token-embedding directions as the
                 anchors (data-aware, manifold-adapted; poor-man's centroids).

Token-level MaxSim quality is the *ceiling* for a given sparse code (SMVE-style
pooling trades quality for index speed), so this isolates the effect of the
anchor choice on retrieval quality.
"""
from __future__ import annotations

import argparse

import torch
from datasets import load_dataset

from pylate import evaluation, models


class _ChunkedNanoBEIR(evaluation.NanoBEIREvaluator):
    def __init__(self, *args, corpus_chunk_size: int = 256, **kwargs) -> None:
        self._corpus_chunk_size = corpus_chunk_size
        super().__init__(*args, **kwargs)

    def _load_dataset(self, dataset_name, **kwargs):
        kwargs.setdefault("corpus_chunk_size", self._corpus_chunk_size)
        return super()._load_dataset(dataset_name, **kwargs)


def collect_dense_tokens(model, docs, batch_size):
    """Raw (pre-normalization) dense token embeddings the sparse head sees."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transformer, dense = model[0], model[1]
    chunks = []
    for start in range(0, len(docs), batch_size):
        feats = model.tokenize(docs[start : start + batch_size], is_query=False)
        feats = {k: v.to(device) for k, v in feats.items()}
        with torch.no_grad():
            feats = dense(transformer(feats))
        mask = feats["attention_mask"].bool()
        tok = feats["token_embeddings"]
        chunks.append(tok[mask].float().cpu())
    return torch.cat(chunks, dim=0)  # (N, dim)


def build_anchor_weight(anchor, dim, sparse_dim, fit_tokens):
    """Return (weight (sparse_dim, dim), mean (dim,) or None) for the chosen anchor."""
    mean = fit_tokens.mean(0)
    if anchor == "orthogonal":
        W = torch.empty(sparse_dim, dim)
        torch.nn.init.orthogonal_(W)
        return W, None
    if anchor == "whitened":
        centered = fit_tokens - mean
        cov = (centered.t() @ centered) / centered.shape[0]
        cov += 1e-3 * torch.eye(dim)  # shrinkage
        evals, evecs = torch.linalg.eigh(cov)
        whiten = evecs @ torch.diag(evals.clamp_min(1e-6).rsqrt()) @ evecs.t()
        A = torch.empty(sparse_dim, dim)
        torch.nn.init.orthogonal_(A)
        return A @ whiten, mean  # project (x - mean) through whitening then rotation
    if anchor == "data":
        idx = torch.randperm(fit_tokens.shape[0])[:sparse_dim]
        A = torch.nn.functional.normalize(fit_tokens[idx], dim=-1)
        return A, None
    raise ValueError(anchor)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="lightonai/GTE-ModernColBERT-v1")
    p.add_argument("--anchor", choices=["orthogonal", "whitened", "data"], required=True)
    p.add_argument("--sparse_dim", type=int, default=16384)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--fit_docs", type=int, default=400)
    p.add_argument("--trust_remote_code", action="store_true", default=False)
    args = p.parse_args()

    model = models.ColBERT(args.model, trust_remote_code=args.trust_remote_code)
    dim = model[-1].out_features

    rows = load_dataset(
        "sentence-transformers/msmarco-bm25", "triplet", split=f"train[:{args.fit_docs}]"
    )
    fit_tokens = collect_dense_tokens(model, [r["positive"] for r in rows], batch_size=16)
    weight, mean = build_anchor_weight(args.anchor, dim, args.sparse_dim, fit_tokens)

    sparse = models.SparseProjection(
        in_features=dim, out_features=args.sparse_dim, k=args.k, bias=mean is not None
    )
    with torch.no_grad():
        sparse.linear.weight.copy_(weight)
        if mean is not None:
            sparse.linear.bias.copy_(-(weight @ mean))
    model.append(sparse)

    scores = _ChunkedNanoBEIR(corpus_chunk_size=256)(model)
    mean_ndcg = scores.get("NanoBEIR_mean_MaxSim_ndcg@10") or scores.get(
        "NanoBEIR_mean_cosine_ndcg@10"
    )
    print(f"\n===== anchor={args.anchor} dim={args.sparse_dim} k={args.k} =====")
    print(f"NanoBEIR_mean_ndcg@10 = {mean_ndcg}", flush=True)


if __name__ == "__main__":
    main()
