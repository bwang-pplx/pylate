from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from ..models import ColBERT
from .contrastive import extract_skiplist_mask

__all__ = ["SparseReconstruction"]


class SparseReconstruction(nn.Module):
    """Self-supervised sparse-autoencoder loss for a SparseProjection head.

    Trains the :class:`~pylate.models.SparseProjection` so its sparse code
    *reconstructs* the frozen dense ColBERT token embedding -- a pure
    autoencoder objective with **no retrieval labels, no distillation, no
    contrastive term**. Unlike matching the token-similarity matrix (which a
    random projection already satisfies, giving no learning signal), forcing the
    sparse code to recover the dense vector is a genuine signal: every active
    dimension must carry reconstructable information, which preserves the dense
    geometry (so sparse MaxSim approximates dense MaxSim) and spreads the code
    across dimensions.

    The decoder is **tied** to the encoder (``x_hat = z @ W`` where ``W`` is the
    SparseProjection's linear weight), so the only trainable parameters are the
    encoder's -- optimized by the standard trainer with the backbone frozen, no
    extra optimizer plumbing.

    Parameters
    ----------
    model
        A :class:`~pylate.models.ColBERT` whose module at
        ``sparse_projection_index`` is a :class:`~pylate.models.SparseProjection`.
    sparse_projection_index
        Index of the sparse projection module. Defaults to ``-1`` (last module).
    freeze_backbone
        Freeze every module except the sparse projection. Defaults to ``True``.
    normalize_target
        Reconstruct the L2-normalized dense embedding (what MaxSim scores over)
        rather than the raw one. Defaults to ``True``.
    ortho_lambda
        Weight of a decoder-orthogonality regularizer ``||WᵀW − I||²`` on the
        encoder weight ``W`` (shape ``out × in``), which keeps the ``in``
        columns orthonormal. With orthonormal columns the projection preserves
        inner products (``zᵢ·zⱼ ≈ xᵢ·xⱼ``), so the sparse similarity tracks the
        dense one even as reconstruction adapts ``W`` -- preventing the
        "reconstruction improves but retrieval drops" drift. Defaults to ``0.0``.
    """

    def __init__(
        self,
        model: ColBERT,
        sparse_projection_index: int = -1,
        freeze_backbone: bool = True,
        normalize_target: bool = True,
        ortho_lambda: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.sparse_projection_index = sparse_projection_index
        self.normalize_target = normalize_target
        self.ortho_lambda = ortho_lambda
        self.last_reconstruction = 0.0
        self.last_ortho = 0.0
        if freeze_backbone:
            self.freeze_backbone()

    def _module(self, model: ColBERT, index: int):
        return list(model._modules.values())[index]

    def freeze_backbone(self) -> None:
        """Freeze every module except the sparse projection."""
        model = self.model.module if hasattr(self.model, "module") else self.model
        sparse_module = self._module(model, self.sparse_projection_index)
        for module in model._modules.values():
            requires_grad = module is sparse_module
            for parameter in module.parameters():
                parameter.requires_grad = requires_grad

    def _encode(self, features: dict[str, torch.Tensor]):
        """Return ``(dense, sparse_codes, sparse_module)``.

        The backbone runs under ``eval`` + ``no_grad`` (deterministic, frozen);
        the sparse projection keeps its own train/eval state.
        """
        model = self.model.module if hasattr(self.model, "module") else self.model
        modules = list(model._modules.values())
        sparse_module = self._module(model, self.sparse_projection_index)
        position = modules.index(sparse_module)
        backbone_modules = modules[:position]

        was_training = [module.training for module in backbone_modules]
        for module in backbone_modules:
            module.eval()
        try:
            with torch.no_grad():
                backbone_features = dict(features)
                for module in backbone_modules:
                    backbone_features = module(backbone_features)
        finally:
            for module, training in zip(backbone_modules, was_training):
                module.train(training)

        dense = backbone_features["token_embeddings"].detach()
        sparse = sparse_module(dict(backbone_features))["token_embeddings"]
        return dense, sparse, sparse_module

    def forward(
        self,
        sentence_features: Iterable[dict[str, torch.Tensor]],
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Masked MSE reconstruction loss, averaged over valid token features."""
        model = self.model.module if hasattr(self.model, "module") else self.model
        sentence_features = list(sentence_features)
        masks = extract_skiplist_mask(
            sentence_features=sentence_features, skiplist=model.skiplist
        )

        squared_error = None
        valid = None
        sparse_module = None
        for features, mask in zip(sentence_features, masks):
            dense, sparse, sparse_module = self._encode(features)
            target = dense
            if self.normalize_target:
                target = torch.nn.functional.normalize(target, p=2, dim=-1)
            # Tied-weight decoder: x_hat = z @ W (W: out_features x in_features).
            reconstruction = sparse @ sparse_module.linear.weight
            token_mask = mask.to(dense.dtype).unsqueeze(-1)
            error = ((reconstruction - target) ** 2 * token_mask).sum()
            count = token_mask.sum() * dense.shape[-1]
            squared_error = error if squared_error is None else squared_error + error
            valid = count if valid is None else valid + count

        if valid == 0:
            raise ValueError("No valid (non-padding, non-skiplist) tokens to reconstruct.")
        reconstruction_loss = squared_error / valid
        self.last_reconstruction = float(reconstruction_loss.detach())

        if self.ortho_lambda <= 0:
            return reconstruction_loss

        # Keep the projection inner-product-preserving: ||WᵀW − I||² over the
        # in_features columns (WᵀW is in×in), so sparse similarity tracks dense.
        weight = sparse_module.linear.weight  # (out, in)
        gram = weight.t() @ weight  # (in, in)
        identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        ortho = ((gram - identity) ** 2).mean()
        self.last_ortho = float(ortho.detach())
        return reconstruction_loss + self.ortho_lambda * ortho
