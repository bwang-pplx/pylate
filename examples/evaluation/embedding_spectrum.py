# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0
"""Effective-rank / anisotropy of dense ColBERT token embeddings.

Diagnoses why LSH-style sparsification (random projection + top-k) is nearly
free on some encoders but lossy on others: a low-effective-rank / anisotropic
embedding cloud compresses cheaply (top-k keeps most of the signal), a
high-effective-rank / isotropic one does not.

Reports, per model: participation ratio (effective rank), fraction of variance
in the top direction (anisotropy), dimensions to reach 90% variance, and mean
pairwise cosine of (normalized) token embeddings.
"""
from __future__ import annotations

import argparse

import torch
from datasets import load_dataset

from pylate import models


def spectrum(name: str, trust_remote_code: bool, docs: list[str], batch_size: int) -> None:
    model = models.ColBERT(name, trust_remote_code=trust_remote_code)
    embeddings = model.encode(
        docs,
        is_query=False,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    tokens = torch.cat([e.float().cpu() for e in embeddings], dim=0)
    normalized = torch.nn.functional.normalize(tokens, p=2, dim=-1)
    centered = normalized - normalized.mean(0, keepdim=True)
    variance = torch.linalg.svdvals(centered) ** 2
    participation_ratio = float(variance.sum() ** 2 / (variance**2).sum())
    top1 = float(variance[0] / variance.sum())
    dims_90 = int((torch.cumsum(variance, 0) / variance.sum() < 0.90).sum()) + 1
    sample = normalized[torch.randperm(normalized.shape[0])[:3000]]
    mean_cosine = float((sample @ sample.t()).mean())
    print(
        f"{name.split('/')[-1]}: tokens={tokens.shape[0]} dim={tokens.shape[1]} "
        f"eff_rank(PR)={participation_ratio:.1f} top1_var={top1:.3f} "
        f"dims@90%={dims_90} mean_pair_cos={mean_cosine:.3f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_docs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    rows = load_dataset(
        "sentence-transformers/msmarco-bm25", "triplet", split=f"train[:{args.num_docs}]"
    )
    docs = [row["positive"] for row in rows]
    spectrum("lightonai/GTE-ModernColBERT-v1", False, docs, args.batch_size)
    spectrum("perplexity-ai/pplx-embed-v1-late-0.6b", True, docs, args.batch_size)


if __name__ == "__main__":
    main()
