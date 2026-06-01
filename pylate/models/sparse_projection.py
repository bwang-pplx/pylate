from __future__ import annotations

import json
import os

import torch
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model
from torch import nn

__all__ = ["SparseProjection"]

# Non-negative activations keep the sparse codes in the cosine "positive cone"
# that :class:`~pylate.losses.SparseDistillation` distills into. ``softplus`` is
# offered as a smooth alternative to ``relu`` that has a non-zero gradient for
# negative pre-activations (avoiding dead units), at the cost of exact zeros only
# coming from the TopK mask.
_ACTIVATIONS = {
    "relu": nn.ReLU,
    "softplus": nn.Softplus,
}


class SparseProjection(nn.Module):
    """Sparse projection module that maps dense ColBERT token embeddings to
    non-negative, optionally TopK-sparse, token codes for sparse late interaction.

    The module is an additional projection meant to be appended on top of an
    existing (typically frozen) ColBERT backbone. It learns a sparse code per
    token via a single linear layer followed by a non-negative activation
    (``relu``), and optionally a hard per-token TopK mask that keeps only the
    ``k`` largest entries and zeroes the rest. This is the encoder used for the
    token-level distillation described in
    :class:`~pylate.losses.SparseDistillation`: no decoder is required because
    the training signal comes from matching token-token similarities rather than
    from reconstruction.

    Parameters
    ----------
    in_features
        Size of the input (dense) token embeddings.
    out_features
        Size of the sparse code (typically larger than ``in_features``).
    k
        If set, keep only the ``k`` largest entries of each token code and zero
        the rest (hard TopK sparsity). If ``None``, only the non-negative
        activation is applied (no hard sparsity). Defaults to ``None``.
    bias
        Whether to add a bias vector to the linear layer. Defaults to ``False``.
    activation
        Name of the non-negative activation applied before the TopK mask. One of
        ``"relu"`` (default) or ``"softplus"``. ``relu`` zeroes negative
        pre-activations (and gives them zero gradient); ``softplus`` is smooth
        and keeps a non-zero gradient everywhere, which can help avoid dead
        units, at the cost of relying solely on the TopK mask for hard zeros.

    Examples
    --------
    >>> import torch
    >>> from pylate import models

    >>> projection = models.SparseProjection(in_features=128, out_features=256, k=8)

    >>> features = {"token_embeddings": torch.randn(2, 4, 128)}
    >>> features = projection(features)

    >>> features["token_embeddings"].shape
    torch.Size([2, 4, 256])

    >>> # Codes are non-negative and at most k entries per token are non-zero.
    >>> bool((features["token_embeddings"] >= 0).all())
    True
    >>> bool((features["token_embeddings"] > 0).sum(dim=-1).max() <= 8)
    True

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k: int | None = None,
        bias: bool = False,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if k is not None and (k <= 0 or k > out_features):
            raise ValueError(
                f"k must be in the range [1, out_features={out_features}], got {k}."
            )
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(_ACTIVATIONS)}, got {activation!r}."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.activation = activation
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation_function = _ACTIVATIONS[activation]()

    def topk_mask(self, codes: torch.Tensor) -> torch.Tensor:
        """Keep only the ``k`` largest entries of each token code, zeroing the rest.

        Operates on the last dimension and preserves the input shape. When ``k``
        is ``None`` the codes are returned unchanged.

        The forward output is always hard-sparse (exactly the kept entries). When
        the module is in training mode a straight-through estimator is used so
        that gradients still flow to the dropped (non-top-k) logits: the backward
        pass sees the dense pre-mask codes. This avoids the dead-logit problem
        where entries that are never selected would never receive a gradient.
        """
        if self.k is None:
            return codes
        topk_values, topk_indices = codes.topk(self.k, dim=-1)
        mask = torch.zeros_like(codes)
        mask.scatter_(dim=-1, index=topk_indices, src=torch.ones_like(topk_values))
        sparse_codes = codes * mask
        if self.training:
            # Straight-through: forward is hard-sparse, backward flows to all
            # logits through ``codes``.
            return codes + (sparse_codes - codes).detach()
        return sparse_codes

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Projects token embeddings to sparse, non-negative token codes."""
        token_embeddings = features["token_embeddings"]
        codes = self.activation_function(self.linear(token_embeddings))
        codes = self.topk_mask(codes)
        features["token_embeddings"] = codes
        return features

    def get_config_dict(self) -> dict:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "k": self.k,
            "bias": self.linear.bias is not None,
            "activation": self.activation,
        }

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs):
        """Save the module configuration and weights to ``output_path``."""
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut)
        if safe_serialization:
            save_safetensors_model(self, os.path.join(output_path, "model.safetensors"))
        else:
            torch.save(
                self.state_dict(), os.path.join(output_path, "pytorch_model.bin")
            )

    @staticmethod
    def load(input_path) -> "SparseProjection":
        """Load a SparseProjection module from ``input_path``."""
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        model = SparseProjection(**config)

        if os.path.exists(os.path.join(input_path, "model.safetensors")):
            load_safetensors_model(model, os.path.join(input_path, "model.safetensors"))
            return model

        model.load_state_dict(
            torch.load(
                os.path.join(input_path, "pytorch_model.bin"),
                map_location=torch.device("cpu"),
            )
        )
        return model
