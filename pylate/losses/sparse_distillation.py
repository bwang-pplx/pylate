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


def _masked_maxsim(
    similarity: torch.Tensor, query_mask: torch.Tensor, document_mask: torch.Tensor
) -> torch.Tensor:
    """ColBERT MaxSim score per (query, document) pair from a cross-similarity.

    ``similarity`` is ``(batch, num_query_tokens, num_doc_tokens)``. Invalid
    document tokens are masked out before the max over document tokens, and
    invalid query tokens are zeroed before the sum, giving a ``(batch,)`` score.
    """
    neg_inf = torch.finfo(similarity.dtype).min
    masked = similarity.masked_fill(document_mask.unsqueeze(1) == 0, neg_inf)
    per_query_token_max = masked.max(dim=2).values
    per_query_token_max = per_query_token_max * query_mask
    return per_query_token_max.sum(dim=1)


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
    live on the same scale. Normalization means the sparse codes are trained as
    *directions* in the sparse space rather than as magnitude-weighted (term
    weight) vectors. Because the sparse codes are non-negative, their cosine
    similarity lies in ``[0, 1]``; with ``clamp_dense_target=True`` (default) the
    dense target is clamped to ``[0, 1]`` as well so a non-negative sparse model
    can drive the loss to zero -- i.e. it distills the *positive cone* of the
    dense interactions, which is what MaxSim selects over. Set
    ``clamp_dense_target=False`` to keep the signed dense target instead (the
    non-negative sparse codes then cannot represent the negative entries, so the
    loss has a non-zero floor). The squared error is averaged over valid
    (non-padding, non-skiplist) token pairs only, making the loss independent of
    the padding ratio.

    Train/serve consistency: this commits to *cosine* sparse late interaction.
    PyLate's ``ColBERT.encode`` runs the full module stack (including this sparse
    projection) and L2-normalizes the final token embeddings when
    ``normalize_embeddings=True`` (the default), so the codes stored for
    retrieval are normalized exactly as they are here -- there is no train/serve
    skew. Retrieval must keep ``normalize_embeddings=True`` for this to hold.

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
    flops_lambda
        Weight of the FLOPS regularizer (SPLADE-style). The FLOPS term is
        ``sum_j (mean_token activation_j)^2`` computed separately over the valid
        query tokens and document tokens and summed. Because it penalizes the
        *square* of each dimension's average activation, it punishes dimensions
        that fire across many tokens/documents, pushing activations to spread
        across many dimensions with short posting lists (a cheap inverted index).
        ``0.0`` (default) disables it and the loss is the pure distillation MSE.
    flops_warmup_steps
        If ``> 0``, ramp the FLOPS weight quadratically from ``0`` to
        ``flops_lambda`` over this many ``forward`` calls (SPLADE warmup), so the
        model learns the distillation task before being squeezed sparse. ``0``
        (default) applies ``flops_lambda`` from the first step.
    flops_on_raw
        Whether to compute the FLOPS term on the *raw* sparse codes (before L2
        normalization) rather than the normalized ones. Defaults to ``True``,
        matching SPLADE: normalizing caps every token's mass at unit norm, which
        blunts the FLOPS penalty (concentrating mass becomes "free" under a fixed
        norm budget). The raw activations give the regularizer a much sharper
        signal. The distillation MSE always uses the normalized codes regardless.
    center_dense
        Whether to subtract the batch mean of the valid dense token embeddings
        from the dense teacher before building the target similarity. Defaults to
        ``False``. Contextualized token embeddings are anisotropic (a large shared
        component), so the uncentered similarity matrix is dominated by a
        near-constant positive offset -- cheap to reproduce by collapsing onto a
        few shared sparse dimensions, while the discriminative residual that
        drives ranking is under-fit. Centering removes that common component so
        the target reflects discriminative structure. Only the *target* is
        centered; the sparse codes (and retrieval) are unchanged.
    distillation_mode
        ``"token"`` (default) minimizes the MSE over the full query-token ×
        document-token similarity matrix. ``"score"`` instead reduces each
        (query, document) pair to its ColBERT **MaxSim score** and distills those
        with a KL divergence over the candidate documents (positive + negatives),
        like standard ColBERT knowledge distillation. ``"score"`` optimizes the
        ranking signal directly rather than every token pair (most of which are
        irrelevant to MaxSim), and requires at least two documents per query.

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
        flops_lambda: float = 0.0,
        flops_warmup_steps: int = 0,
        flops_on_raw: bool = True,
        center_dense: bool = False,
        distillation_mode: str = "token",
    ) -> None:
        super().__init__()
        if distillation_mode not in ("token", "score"):
            raise ValueError(
                f"distillation_mode must be 'token' or 'score', got "
                f"{distillation_mode!r}."
            )
        self.model = model
        self.sparse_projection_index = sparse_projection_index
        self.clamp_dense_target = clamp_dense_target
        self.size_average = size_average
        self.flops_lambda = flops_lambda
        self.flops_warmup_steps = flops_warmup_steps
        self.flops_on_raw = flops_on_raw
        self.center_dense = center_dense
        self.distillation_mode = distillation_mode
        # Forward-call counter for the (optional) quadratic FLOPS warmup. Plain
        # int (not a buffer): only used for lambda scheduling, not checkpointed.
        self._step = 0
        # Most recent loss components, for logging/inspection.
        self.last_distillation = 0.0
        self.last_flops = 0.0
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

    def unfreeze_backbone(self) -> None:
        """Re-enable gradients for every module, undoing :meth:`freeze_backbone`.

        ``freeze_backbone`` mutates the user's model in place; call this to
        restore the original ``requires_grad`` state (all parameters trainable)
        if the model is reused outside this loss.
        """
        model = self.model.module if hasattr(self.model, "module") else self.model
        for parameter in model.parameters():
            parameter.requires_grad = True

    def _encode(
        self, features: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the model, returning ``(dense_embeddings, sparse_codes)``.

        The dense embeddings are the token embeddings produced by the backbone
        (with gradients disabled) and the sparse codes are the output of the
        sparse projection applied to them. The backbone is forced into eval mode
        for this pass so that dropout/other stochastic layers do not make the
        distillation targets noisy; its previous train/eval state is restored
        afterwards. The sparse projection keeps its own train/eval state (its
        straight-through estimator depends on it).
        """
        model = self.model.module if hasattr(self.model, "module") else self.model
        modules = list(model._modules.values())
        sparse_module = self._module(model, self.sparse_projection_index)
        sparse_position = modules.index(sparse_module)
        backbone_modules = modules[:sparse_position]

        was_training = [module.training for module in backbone_modules]
        for module in backbone_modules:
            module.eval()
        try:
            # Run the frozen backbone without tracking gradients.
            with torch.no_grad():
                backbone_features = dict(features)
                for module in backbone_modules:
                    backbone_features = module(backbone_features)
        finally:
            for module, training in zip(backbone_modules, was_training):
                module.train(training)
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
        if len(sentence_features) < 2:
            raise ValueError(
                "SparseDistillation requires at least one document in addition to "
                "the query (sentence_features must have length >= 2)."
            )
        masks = extract_skiplist_mask(
            sentence_features=sentence_features, skiplist=model.skiplist
        )

        # Encode every group up front: centering needs the batch mean over all
        # valid dense tokens before any similarity is formed. Stores raw dense
        # and raw sparse codes plus the float mask per group.
        encoded = []
        for features, mask in zip(sentence_features, masks):
            dense, sparse_raw = self._encode(features)
            encoded.append((dense, sparse_raw, mask.to(dense.dtype)))

        # Mean of the valid (non-padding, non-skiplist) dense token embeddings,
        # pooled across query and documents. Subtracted from the dense teacher to
        # remove the anisotropic common component before building the target.
        dense_mean = None
        if self.center_dense:
            activation_sum = encoded[0][0].new_zeros(encoded[0][0].shape[-1])
            token_count = encoded[0][0].new_zeros(())
            for dense, _, mask in encoded:
                activation_sum = activation_sum + (dense * mask.unsqueeze(-1)).sum(
                    dim=(0, 1)
                )
                token_count = token_count + mask.sum()
            dense_mean = activation_sum / token_count.clamp_min(1.0)

        def prepare(dense, sparse_raw):
            if dense_mean is not None:
                dense = dense - dense_mean
            dense_n = torch.nn.functional.normalize(dense, p=2, dim=-1)
            sparse_n = torch.nn.functional.normalize(sparse_raw, p=2, dim=-1)
            flops_codes = sparse_raw if self.flops_on_raw else sparse_n
            return dense_n, sparse_n, flops_codes

        query_dense, query_sparse, query_flops_codes = prepare(*encoded[0][:2])
        query_mask = encoded[0][2]

        squared_error = query_dense.new_zeros(())
        valid_pairs = query_dense.new_zeros(())
        # Per-document MaxSim scores (score mode only).
        dense_scores: list[torch.Tensor] = []
        sparse_scores: list[torch.Tensor] = []

        # FLOPS regularizer accumulators (document side). Summed activations per
        # dimension over valid document tokens, plus the valid-token count, so we
        # can form the per-dimension mean once at the end.
        compute_flops = self.flops_lambda > 0
        if compute_flops:
            doc_activation_sum = query_sparse.new_zeros(query_sparse.shape[-1])
            doc_token_count = query_sparse.new_zeros(())

        for document_dense_raw, document_sparse_raw, mask in encoded[1:]:
            document_dense, document_sparse, document_flops_codes = prepare(
                document_dense_raw, document_sparse_raw
            )

            dense_similarity = _cross_similarity(query_dense, document_dense)
            sparse_similarity = _cross_similarity(query_sparse, document_sparse)
            if self.clamp_dense_target:
                dense_similarity = dense_similarity.clamp_min(0.0)

            if self.distillation_mode == "score":
                dense_scores.append(
                    _masked_maxsim(dense_similarity, query_mask, mask)
                )
                sparse_scores.append(
                    _masked_maxsim(sparse_similarity, query_mask, mask)
                )
            else:
                pair_mask = query_mask.unsqueeze(2) * mask.unsqueeze(1)
                squared_error = (
                    squared_error
                    + ((sparse_similarity - dense_similarity) ** 2 * pair_mask).sum()
                )
                valid_pairs = valid_pairs + pair_mask.sum()

            if compute_flops:
                document_mask = mask.unsqueeze(-1)
                doc_activation_sum = doc_activation_sum + (
                    document_flops_codes * document_mask
                ).sum(dim=(0, 1))
                doc_token_count = doc_token_count + document_mask.sum()

        if self.distillation_mode == "score":
            if len(dense_scores) < 2:
                raise ValueError(
                    "distillation_mode='score' needs at least two documents per "
                    "query (positive + negative) to distill a ranking."
                )
            # (batch, num_documents) scores; KL of the candidate distributions.
            dense_score_matrix = torch.stack(dense_scores, dim=1)
            sparse_score_matrix = torch.stack(sparse_scores, dim=1)
            target = torch.nn.functional.log_softmax(dense_score_matrix, dim=-1)
            student = torch.nn.functional.log_softmax(sparse_score_matrix, dim=-1)
            distillation = torch.nn.functional.kl_div(
                student,
                target,
                reduction="batchmean" if self.size_average else "sum",
                log_target=True,
            )
        elif self.size_average:
            if valid_pairs == 0:
                raise ValueError(
                    "No valid (non-padding, non-skiplist) query-document token "
                    "pairs to compute the loss over."
                )
            distillation = squared_error / valid_pairs
        else:
            distillation = squared_error

        self.last_distillation = float(distillation.detach())
        if not compute_flops:
            return distillation

        # FLOPS = sum_j (mean_token activation_j)^2, query and document sides
        # summed. Penalizing the square of each dimension's mean activation
        # discourages dimensions shared across many tokens -> short posting lists.
        query_activation_sum = (query_flops_codes * query_mask.unsqueeze(-1)).sum(
            dim=(0, 1)
        )
        query_token_count = query_mask.sum().clamp_min(1.0)
        flops_query = (query_activation_sum / query_token_count).pow(2).sum()
        flops_document = (doc_activation_sum / doc_token_count.clamp_min(1.0)).pow(2).sum()
        flops = flops_query + flops_document

        if self.training:
            self._step += 1
        if self.flops_warmup_steps > 0:
            scale = min(1.0, (self._step / self.flops_warmup_steps) ** 2)
        else:
            scale = 1.0
        effective_lambda = self.flops_lambda * scale

        self.last_flops = float(flops.detach())
        return distillation + effective_lambda * flops
