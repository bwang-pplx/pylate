# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0
"""Training-free sparsification baselines on a frozen ColBERT backbone.

Methods (all untrained):
  * ``wta``: Winner-Take-All. Random projection 128->out, then keep the argmax
    within each of ``k`` fixed windows. Exactly k nonzeros, spread by
    construction (one active dim per window) -- a load-balanced LSH variant.
  * ``lexical``: frozen MLM-head-style projection. Project the 768-d backbone
    hidden states onto the vocabulary via the model's own (tied) input
    embeddings, ReLU, top-k -> a vocab-grounded sparse code. Escapes the
    rank-128 ColBERT bottleneck and needs no training.

Reports NanoBEIR mean nDCG@10 plus an activation-spread summary.
"""
from __future__ import annotations

import argparse

import torch
from torch import nn

from pylate import evaluation, models


class _ChunkedNanoBEIR(evaluation.NanoBEIREvaluator):
    def __init__(self, *args, corpus_chunk_size: int = 256, **kwargs) -> None:
        self._corpus_chunk_size = corpus_chunk_size
        super().__init__(*args, **kwargs)

    def _load_dataset(self, dataset_name, **kwargs):
        kwargs.setdefault("corpus_chunk_size", self._corpus_chunk_size)
        return super()._load_dataset(dataset_name, **kwargs)


class WTAHead(nn.Module):
    """Random projection + per-window argmax (Winner-Take-All)."""

    def __init__(self, in_features: int, out_features: int, k: int) -> None:
        super().__init__()
        if out_features % k != 0:
            raise ValueError("out_features must be divisible by k (one window per nonzero).")
        self.out_features = out_features
        self.k = k
        self.linear = nn.Linear(in_features, out_features, bias=False)
        nn.init.orthogonal_(self.linear.weight)

    def forward(self, features: dict) -> dict:
        x = self.linear(features["token_embeddings"]).relu()
        *lead, dim = x.shape
        windows = x.reshape(*lead, self.k, dim // self.k)
        idx = windows.argmax(dim=-1, keepdim=True)
        winners = torch.zeros_like(windows).scatter_(-1, idx, windows.gather(-1, idx))
        features["token_embeddings"] = winners.reshape(*lead, dim)
        return features


class LexicalHead(nn.Module):
    """Frozen MLM-head-style projection onto the (tied) vocabulary, ReLU + top-k."""

    def __init__(self, embedding_weight: torch.Tensor, k: int) -> None:
        super().__init__()
        # embedding_weight: (vocab, hidden); tied MLM decoder ~= embeddings^T.
        self.register_buffer("vocab", embedding_weight.detach().clone())
        self.out_features = embedding_weight.shape[0]
        self.k = k

    def forward(self, features: dict) -> dict:
        logits = (features["token_embeddings"] @ self.vocab.t()).relu()
        topv, topi = logits.topk(self.k, dim=-1)
        codes = torch.zeros_like(logits).scatter_(-1, topi, topv)
        features["token_embeddings"] = codes
        return features


def activation_spread(model, docs, batch_size) -> None:
    vocab = model[-1].out_features
    embs = model.encode(docs, is_query=False, batch_size=batch_size,
                        convert_to_tensor=True, show_progress_bar=False)
    token_freq = torch.zeros(vocab)
    doc_freq = torch.zeros(vocab)
    for emb in embs:
        mask = (emb > 0).cpu()
        token_freq += mask.sum(dim=0).float()
        doc_freq += mask.any(dim=0).float()
    sorted_freq, _ = torch.sort(token_freq, descending=True)
    cum = torch.cumsum(sorted_freq, 0) / sorted_freq.sum()
    used_pl = doc_freq[doc_freq > 0]
    print(f"  spread: dims_used={int((token_freq>0).sum())}/{vocab}  "
          f"top128_share={float(cum[min(128,vocab)-1])*100:.1f}%  "
          f"posting_median={int(used_pl.median())}  posting_max={int(doc_freq.max())}/{len(docs)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["wta", "lexical"], required=True)
    p.add_argument("--model", default="lightonai/GTE-ModernColBERT-v1")
    p.add_argument("--out_features", type=int, default=16384)  # wta only
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--corpus_chunk_size", type=int, default=256)
    p.add_argument("--probe_docs", type=int, default=1000)
    args = p.parse_args()

    model = models.ColBERT(args.model)
    if args.method == "wta":
        dim = model[-1].out_features
        model.append(WTAHead(in_features=dim, out_features=args.out_features, k=args.k))
        label = f"WTA(out={args.out_features}, k={args.k})"
        chunk = args.corpus_chunk_size
    else:  # lexical: drop the 128-d Dense, project 768-d hidden via tied embeddings
        emb = model[0].auto_model.get_input_embeddings().weight
        model[1] = LexicalHead(emb, k=args.k)  # replace Dense
        label = f"LEXICAL(vocab={model[-1].out_features}, k={args.k})"
        chunk = min(args.corpus_chunk_size, 128)  # 50k-dim -> smaller chunk

    scores = _ChunkedNanoBEIR(corpus_chunk_size=chunk)(model)
    mean = scores.get("NanoBEIR_mean_MaxSim_ndcg@10") or scores.get("NanoBEIR_mean_cosine_ndcg@10")
    print(f"\n===== {label} (training-free) =====")
    print(f"NanoBEIR_mean_ndcg@10 = {mean}")

    from datasets import load_dataset
    rows = load_dataset("sentence-transformers/msmarco-bm25", "triplet",
                        split=f"train[:{args.probe_docs}]")
    activation_spread(model, [r["positive"] for r in rows], batch_size=32)


if __name__ == "__main__":
    main()
