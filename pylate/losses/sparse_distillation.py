from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from ..models import ColBERT
from .contrastive import extract_skiplist_mask

__all__ = ["SparseDistillation"]


def _cross_similarity(queries: torch.Tensor, documents: torch.Tensor) -> torch.Tensor:
    """Query-token by document-token similarity matrix.

    Parameters
    ----------
    queries
        Query token embeddings of shape ``(batch_size, num_query_tokens, dim)``.
    documents
        Document token embeddings of shape ``(batch_size, num_doc_tokens, dim)``.

    Returns
    -------
    Similarity tensor of shape ``(batch_size, num_query_tokens, num_doc_tokens)``.
    This is the cross-similarity ColBERT retrieval relies on (MaxSim is taken over
    it), so distilling it preserves the retrieval-relevant structure.
    """
    return torch.einsum("bqh,bdh->bqd", queries, documents)


class SparseDistillation(nn.Module):
    """Token-level distillation loss for learning a sparse projection on top of a
    ColBERT model (Option B / token-level distillation).

    The loss trains an additional :class:`~pylate.models.SparseProjection` module
    so that the query-token by document-token similarity matrix produced by the
    *sparse* codes approximates the one produced by the original *dense* ColBERT
    token embeddings. This cross similarity is exactly what ColBERT's MaxSim
    scoring is computed over, so distilling it preserves the retrieval-relevant
    structure. It is task-specific: it operates on the query and document token
    embeddings of real (query, documents) pairs, and requires neither
    reconstruction, a sparse contrastive loss, nor the full SSR auxiliary loss
    stack.

    Both dense and sparse token embeddings are L2-normalized before the
    similarity is computed, so the dense (cosine) target and the sparse target
    live on the same scale. Because the sparse codes are non-negative, their
    cosine similarity lies in ``[0, 1]``; the dense target is clamped to
    ``[0, 1]`` as well so a non-negative sparse model can drive the loss to zero.
    The squared error is averaged over valid (non-padding, non-skiplist) token
    pairs only, making the loss independent of the padding ratio.

    The backbone (every module before the sparse projection) is expected to be
    frozen; by default this loss freezes it for you (``freeze_backbone=True``) so
    that only the sparse projection is trained. This is backward-compatible: it
    only affects the model passed to this loss.

    Parameters
    ----------
    model
        A :class:`~pylate.models.ColBERT` model whose last module is a
        :class:`~pylate.models.SparseProjection`.
    sparse_projection_index
        Index of the sparse projection module within the model. The dense token
        embeddings are the output of the model up to (but excluding) this module;
        the sparse codes are the output after applying it. Defaults to ``-1``
        (the last module).
    freeze_backbone
        Whether to freeze every module except the sparse projection so that only
        the sparse projection is trained. Defaults to ``True``.
    clamp_dense_target
        Whether to clamp the dense cosine target to ``[0, 1]`` so that the
        non-negative sparse codes can match it. Defaults to ``True``.
    size_average
        Average the loss over the valid token pairs (mean) instead of summing.
        Defaults to ``True``.

    Examples
    --------
    >>> from pylate import losses, models

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    ... )
    >>> _ = model.append(
    ...     models.SparseProjection(in_features=128, out_features=256, k=8)
    ... )

    >>> distillation = losses.SparseDistillation(model=model)

    >>> query = model.tokenize(["fruits are healthy."], is_query=True)
    >>> documents = model.tokenize([
    ...     "fruits are good for health.",
    ...     "fruits are bad for health.",
    ... ], is_query=False)

    >>> loss = distillation(sentence_features=[query, documents])
    >>> assert isinstance(loss.item(), float)

    """

    def __init__(
        self,
        model: ColBERT,
        sparse_projection_index: int = -1,
        freeze_backbone: bool = True,
        clamp_dense_target: bool = True,
        size_average: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.sparse_projection_index = sparse_projection_index
        self.clamp_dense_target = clamp_dense_target
        self.size_average = size_average
        if freeze_backbone:
            self.freeze_backbone()

    def _module(self, model: ColBERT, index: int):
        """Return the module at ``index`` (resolved against the module list)."""
        modules = list(model._modules.values())
        return modules[index]

    def freeze_backbone(self) -> None:
        """Freeze every module except the sparse projection.

        Disables gradients for all parameters of the backbone (the modules before
        the sparse projection) so that only the sparse projection is trained.
        """
        model = self.model.module if hasattr(self.model, "module") else self.model
        sparse_module = self._module(model, self.sparse_projection_index)
        for module in model._modules.values():
            requires_grad = module is sparse_module
            for parameter in module.parameters():
                parameter.requires_grad = requires_grad

    def _encode(
        self, features: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the model, returning ``(dense_embeddings, sparse_codes)``.

        The dense embeddings are the token embeddings produced by the backbone
        (with gradients disabled) and the sparse codes are the output of the
        sparse projection applied to them.
        """
        model = self.model.module if hasattr(self.model, "module") else self.model
        modules = list(model._modules.values())
        sparse_module = self._module(model, self.sparse_projection_index)
        sparse_position = modules.index(sparse_module)

        # Run the frozen backbone without tracking gradients.
        with torch.no_grad():
            backbone_features = dict(features)
            for module in modules[:sparse_position]:
                backbone_features = module(backbone_features)
        dense_embeddings = backbone_features["token_embeddings"].detach()

        sparse_features = sparse_module(dict(backbone_features))
        sparse_codes = sparse_features["token_embeddings"]
        return dense_embeddings, sparse_codes

    def forward(
        self,
        sentence_features: Iterable[dict[str, torch.Tensor]],
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the token-level distillation loss.

        Parameters
        ----------
        sentence_features
            List of tokenized sentences. The first is the query and the rest are
            documents.
        labels
            Unused; kept for compatibility with the SentenceTransformer trainer.

        """
        model = self.model.module if hasattr(self.model, "module") else self.model
        sentence_features = list(sentence_features)
        masks = extract_skiplist_mask(
            sentence_features=sentence_features, skiplist=model.skiplist
        )

        query_dense, query_sparse = self._encode(sentence_features[0])
        query_dense = torch.nn.functional.normalize(query_dense, p=2, dim=-1)
        query_sparse = torch.nn.functional.normalize(query_sparse, p=2, dim=-1)
        query_mask = masks[0].to(query_dense.dtype)

        squared_error = 0.0
        valid_pairs = 0.0
        for features, mask in zip(sentence_features[1:], masks[1:]):
            document_dense, document_sparse = self._encode(features)
            document_dense = torch.nn.functional.normalize(document_dense, p=2, dim=-1)
            document_sparse = torch.nn.functional.normalize(
                document_sparse, p=2, dim=-1
            )

            dense_similarity = _cross_similarity(query_dense, document_dense)
            sparse_similarity = _cross_similarity(query_sparse, document_sparse)
            if self.clamp_dense_target:
                dense_similarity = dense_similarity.clamp_min(0.0)

            pair_mask = query_mask.unsqueeze(2) * mask.to(query_dense.dtype).unsqueeze(
                1
            )
            squared_error = (
                squared_error
                + ((sparse_similarity - dense_similarity) ** 2 * pair_mask).sum()
            )
            valid_pairs = valid_pairs + pair_mask.sum()

        if not self.size_average:
            return squared_error
        if isinstance(valid_pairs, float):
            return squared_error
        return squared_error / valid_pairs.clamp_min(1.0)
