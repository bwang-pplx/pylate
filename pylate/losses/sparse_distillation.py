from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from ..models import ColBERT
from .contrastive import extract_skiplist_mask

__all__ = ["SparseDistillation"]


def _token_similarity(embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Token-token similarity matrix for a batch of sequences.

    Parameters
    ----------
    embeddings
        Token embeddings of shape ``(batch_size, num_tokens, dim)``.
    mask
        Boolean/float mask of shape ``(batch_size, num_tokens)`` zeroing padding
        and skiplist tokens.

    Returns
    -------
    Similarity tensor of shape ``(batch_size, num_tokens, num_tokens)`` where
    contributions from masked tokens are zeroed on both axes.
    """
    similarity = torch.einsum("bsh,bth->bst", embeddings, embeddings)
    mask = mask.to(similarity.dtype)
    return similarity * mask.unsqueeze(2) * mask.unsqueeze(1)


class SparseDistillation(nn.Module):
    """Token-level distillation loss for learning a sparse projection on top of a
    ColBERT model (Option B / token-level distillation).

    The loss trains an additional :class:`~pylate.models.SparseProjection` module
    so that the token-token similarity matrices produced by the *sparse* codes
    approximate those produced by the original *dense* ColBERT token embeddings.
    It is task-specific: it operates on the query and document token embeddings of
    real (query, documents) pairs, and requires neither reconstruction, a sparse
    contrastive loss, nor the full SSR auxiliary loss stack.

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
    normalize_dense
        Whether to L2-normalize the dense token embeddings before computing their
        similarity matrix (matching ColBERT's MaxSim scoring). Defaults to
        ``True``.
    size_average
        Average the loss over the mini-batch (mean) instead of summing.
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
        normalize_dense: bool = True,
        size_average: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.sparse_projection_index = sparse_projection_index
        self.normalize_dense = normalize_dense
        self.loss_function = nn.MSELoss(reduction="mean" if size_average else "sum")
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
        masks = extract_skiplist_mask(
            sentence_features=sentence_features, skiplist=model.skiplist
        )

        loss = 0.0
        for features, mask in zip(sentence_features, masks):
            dense_embeddings, sparse_codes = self._encode(features)
            if self.normalize_dense:
                dense_embeddings = torch.nn.functional.normalize(
                    dense_embeddings, p=2, dim=-1
                )
            dense_similarity = _token_similarity(dense_embeddings, mask)
            sparse_similarity = _token_similarity(sparse_codes, mask)
            loss = loss + self.loss_function(sparse_similarity, dense_similarity)

        return loss / len(sentence_features)
