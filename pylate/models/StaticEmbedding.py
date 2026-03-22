from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file as save_safetensors_file
from sentence_transformers.models.InputModule import InputModule
from tokenizers import Tokenizer
from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizerFast

__all__ = ["StaticEmbedding"]

logger = logging.getLogger(__name__)


class StaticEmbedding(InputModule):
    """Static token embedding module for ColBERT-style multi-vector models.

    Unlike sentence-transformers' StaticEmbedding which pools tokens into a single
    vector via EmbeddingBag, this module preserves per-token embeddings for use with
    ColBERT's late-interaction (MaxSim) scoring.

    Each token is mapped to a fixed embedding vector via a simple lookup table
    (nn.Embedding), with no transformer attention layers. This makes encoding
    extremely fast (O(n) instead of O(n^2)) while retaining token-level granularity
    for fine-grained matching.

    Parameters
    ----------
    tokenizer
        A fast tokenizer from transformers or tokenizers.
    embedding_weights
        Pre-trained embedding weights. If provided, initializes the embedding
        table with these weights.
    embedding_dim
        Dimension of the embeddings. Required if embedding_weights is not provided.

    Examples
    --------
    >>> from pylate.models import StaticEmbedding, ColBERT, Dense

    >>> from transformers import AutoTokenizer

    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    >>> static_embedding = StaticEmbedding(tokenizer, embedding_dim=128)

    >>> model = ColBERT(
    ...     modules=[static_embedding, Dense(128, 128)],
    ...     device="cpu",
    ... )

    """

    config_file_name = "config.json"
    config_keys = ["embedding_dim", "num_embeddings"]

    def __init__(
        self,
        tokenizer: Tokenizer | PreTrainedTokenizerFast | None = None,
        embedding_weights: np.ndarray | torch.Tensor | None = None,
        embedding_dim: int | None = None,
        base_model: str | None = None,
    ) -> None:
        super().__init__()

        if tokenizer is None:
            # Will be loaded later via load()
            return

        # Store the original tokenizer for HF compatibility
        if isinstance(tokenizer, PreTrainedTokenizerFast):
            self._hf_tokenizer = tokenizer
            self._fast_tokenizer = tokenizer._tokenizer
        elif isinstance(tokenizer, Tokenizer):
            self._hf_tokenizer = None
            self._fast_tokenizer = tokenizer
        else:
            raise ValueError(
                "The tokenizer must be a fast tokenizer from `transformers` or `tokenizers`."
            )

        vocab_size = self._fast_tokenizer.get_vocab_size()

        if embedding_weights is not None:
            if isinstance(embedding_weights, np.ndarray):
                embedding_weights = torch.from_numpy(embedding_weights)
            self.embedding = nn.Embedding.from_pretrained(
                embedding_weights, freeze=False
            )
        elif embedding_dim is not None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else:
            raise ValueError(
                "Either `embedding_weights` or `embedding_dim` must be provided."
            )

        self.num_embeddings = self.embedding.num_embeddings
        self.embedding_dim = self.embedding.embedding_dim
        self.base_model = base_model

        # Set up the HF tokenizer for ColBERT compatibility
        if self._hf_tokenizer is None:
            self._hf_tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=self._fast_tokenizer
            )

        self.tokenizer = self._hf_tokenizer
        self.max_seq_length = 512

    def tokenize(self, texts: list[str], **kwargs) -> dict[str, torch.Tensor]:
        """Tokenize input texts into input_ids and attention_mask.

        This produces standard transformer-style tokenized output (padded batch
        with attention masks) rather than the flat offset-based format used by
        sentence-transformers' StaticEmbedding.

        Parameters
        ----------
        texts
            List of input texts to tokenize.

        Returns
        -------
            Dictionary with 'input_ids' and 'attention_mask' tensors.
        """
        padding = kwargs.get("padding", True)
        encoded = self.tokenizer(
            texts,
            padding=padding,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        result = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        if "token_type_ids" in encoded:
            result["token_type_ids"] = encoded["token_type_ids"]
        return result

    def forward(
        self, features: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        """Look up per-token embeddings and return them without pooling.

        Parameters
        ----------
        features
            Dictionary with 'input_ids' and 'attention_mask' tensors.

        Returns
        -------
            Dictionary with 'token_embeddings', 'input_ids', and 'attention_mask'.
        """
        input_ids = features["input_ids"]
        token_embeddings = self.embedding(input_ids)
        features["token_embeddings"] = token_embeddings
        return features

    def get_word_embedding_dimension(self) -> int:
        """Return the dimension of the token embeddings."""
        return self.embedding_dim

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        self._max_seq_length = value

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        """Resize the embedding table to accommodate new tokens.

        Parameters
        ----------
        new_num_tokens
            The new vocabulary size.
        """
        old_num_tokens = self.embedding.num_embeddings
        if new_num_tokens == old_num_tokens:
            return

        new_embedding = nn.Embedding(new_num_tokens, self.embedding_dim)
        # Copy old weights
        num_to_copy = min(old_num_tokens, new_num_tokens)
        new_embedding.weight.data[:num_to_copy] = self.embedding.weight.data[
            :num_to_copy
        ]
        self.embedding = new_embedding
        self.num_embeddings = new_num_tokens

    def save(
        self, output_path: str, *args, safe_serialization: bool = True, **kwargs
    ) -> None:
        """Save the static embedding model to disk.

        Parameters
        ----------
        output_path
            Directory to save the model to.
        safe_serialization
            If True, save using safetensors format.
        """
        os.makedirs(output_path, exist_ok=True)

        # Save weights
        if safe_serialization:
            save_safetensors_file(
                self.state_dict(), os.path.join(output_path, "model.safetensors")
            )
        else:
            torch.save(
                self.state_dict(), os.path.join(output_path, "pytorch_model.bin")
            )

        # Save tokenizer
        self.tokenizer.save_pretrained(output_path)

        # Save config
        config = {
            "embedding_dim": self.embedding_dim,
            "num_embeddings": self.num_embeddings,
            "base_model": self.base_model,
        }
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, input_path: str, **kwargs) -> "StaticEmbedding":
        """Load a StaticEmbedding from disk.

        Parameters
        ----------
        input_path
            Directory to load the model from.

        Returns
        -------
            A StaticEmbedding instance.
        """
        config_path = os.path.join(input_path, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        tokenizer = AutoTokenizer.from_pretrained(input_path)

        # Filter kwargs to only those accepted by load_torch_weights
        weights = cls.load_torch_weights(input_path)
        embedding_weights = weights.get(
            "embedding.weight", weights.get("embeddings", None)
        )

        instance = cls(
            tokenizer=tokenizer,
            embedding_weights=embedding_weights,
            embedding_dim=config.get("embedding_dim"),
            base_model=config.get("base_model"),
        )
        return instance

    @classmethod
    def from_distillation(
        cls,
        model_name: str,
        device: str | None = None,
        pca_dims: int | None = 256,
        apply_zipf: bool = True,
    ) -> "StaticEmbedding":
        """Create a StaticEmbedding from a model2vec distillation.

        This distills a transformer model into static per-token embeddings
        using the model2vec package.

        Parameters
        ----------
        model_name
            The name of the transformer model to distill.
        device
            Device for distillation computation.
        pca_dims
            Number of PCA dimensions for reduction.
        apply_zipf
            Whether to apply Zipf weighting.

        Returns
        -------
            A StaticEmbedding instance with distilled weights.
        """
        try:
            from model2vec.distill import distill
        except ImportError:
            raise ImportError(
                "To use this method, install model2vec: `pip install model2vec[distill]`"
            )

        static_model = distill(
            model_name,
            device=device,
            pca_dims=pca_dims,
            apply_zipf=apply_zipf,
        )

        if isinstance(static_model.embedding, np.ndarray):
            embedding_weights = torch.from_numpy(static_model.embedding)
        else:
            embedding_weights = static_model.embedding.weight

        # Build an HF tokenizer from the model2vec tokenizer
        fast_tokenizer = static_model.tokenizer
        hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=fast_tokenizer)

        return cls(
            tokenizer=hf_tokenizer,
            embedding_weights=embedding_weights,
            base_model=model_name,
        )

    @classmethod
    def from_model2vec(cls, model_id_or_path: str) -> "StaticEmbedding":
        """Create a StaticEmbedding from a pretrained model2vec model.

        Parameters
        ----------
        model_id_or_path
            The model2vec model identifier or local path.

        Returns
        -------
            A StaticEmbedding instance with the model2vec weights.
        """
        try:
            from model2vec import StaticModel
        except ImportError:
            raise ImportError(
                "To use this method, install model2vec: `pip install model2vec`"
            )

        static_model = StaticModel.from_pretrained(model_id_or_path)

        if isinstance(static_model.embedding, np.ndarray):
            embedding_weights = torch.from_numpy(static_model.embedding)
        else:
            embedding_weights = static_model.embedding.weight

        fast_tokenizer = static_model.tokenizer
        hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=fast_tokenizer)

        return cls(
            tokenizer=hf_tokenizer,
            embedding_weights=embedding_weights,
            base_model=model_id_or_path,
        )
