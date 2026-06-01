"""Tests for the sparse projection module and token-level distillation loss.

These cover Option B (token-level distillation): a SparseProjection module on
top of a frozen ColBERT backbone, trained so the sparse query-token by
document-token similarities approximate the dense ones.
"""

from __future__ import annotations

import pytest
import torch

from pylate.losses.sparse_distillation import _cross_similarity
from pylate.models import SparseProjection


class TestSparseProjectionShapes:
    def test_output_shape(self) -> None:
        proj = SparseProjection(in_features=16, out_features=64)
        features = {"token_embeddings": torch.randn(3, 5, 16)}
        out = proj(features)["token_embeddings"]
        assert out.shape == (3, 5, 64)

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError):
            SparseProjection(in_features=16, out_features=8, k=0)
        with pytest.raises(ValueError):
            SparseProjection(in_features=16, out_features=8, k=9)


class TestSparseProjectionSparsity:
    def test_non_negative(self) -> None:
        proj = SparseProjection(in_features=16, out_features=64).eval()
        out = proj({"token_embeddings": torch.randn(4, 7, 16)})["token_embeddings"]
        assert bool((out >= 0).all())

    def test_topk_caps_nonzeros_per_token(self) -> None:
        k = 5
        proj = SparseProjection(in_features=16, out_features=64, k=k).eval()
        out = proj({"token_embeddings": torch.randn(4, 7, 16)})["token_embeddings"]
        nnz = (out > 0).sum(dim=-1)
        assert int(nnz.max()) <= k

    def test_topk_forward_is_hard_sparse_in_train_mode(self) -> None:
        # Even with the straight-through estimator active, the forward value is
        # exactly the hard-sparse code (only the STE backward differs).
        k = 5
        proj = SparseProjection(in_features=16, out_features=64, k=k).train()
        out = proj({"token_embeddings": torch.randn(4, 7, 16)})["token_embeddings"]
        assert int((out > 0).sum(dim=-1).max()) <= k

    def test_no_topk_keeps_all_positive(self) -> None:
        proj = SparseProjection(in_features=8, out_features=32, k=None).eval()
        # Force a dense positive input through positive weights so relu keeps all.
        with torch.no_grad():
            proj.linear.weight.abs_()
        out = proj({"token_embeddings": torch.ones(2, 3, 8)})["token_embeddings"]
        assert int((out > 0).sum(dim=-1).min()) == 32

    def test_topk_keeps_largest_values(self) -> None:
        proj = SparseProjection(in_features=4, out_features=6, k=2).eval()
        codes = torch.tensor([[[5.0, 1.0, 4.0, 0.0, 3.0, 2.0]]])
        masked = proj.topk_mask(codes)
        # Only the two largest (5.0 and 4.0) survive.
        expected = torch.tensor([[[5.0, 0.0, 4.0, 0.0, 0.0, 0.0]]])
        assert torch.allclose(masked, expected)


class TestSparseProjectionStraightThrough:
    def test_gradients_flow_to_non_topk_logits(self) -> None:
        # With the straight-through estimator, every output dimension's weights
        # receive a gradient, not only the top-k ones that survive the forward.
        torch.manual_seed(0)
        proj = SparseProjection(in_features=8, out_features=16, k=2).train()
        out = proj({"token_embeddings": torch.randn(3, 5, 8)})["token_embeddings"]
        out.sum().backward()
        per_output_grad = proj.linear.weight.grad.abs().sum(dim=-1)
        # All 16 output rows (including dropped logits) get a non-zero gradient.
        assert int((per_output_grad > 0).sum()) == 16

    def test_eval_mode_blocks_non_topk_gradients(self) -> None:
        # In eval mode the mask is a hard constant, so only kept entries get a
        # gradient. This documents the difference from training behaviour.
        torch.manual_seed(0)
        proj = SparseProjection(in_features=8, out_features=16, k=2).eval()
        out = proj({"token_embeddings": torch.randn(3, 5, 8)})["token_embeddings"]
        out.sum().backward()
        per_output_grad = proj.linear.weight.grad.abs().sum(dim=-1)
        assert int((per_output_grad > 0).sum()) < 16


class TestSparseProjectionPersistence:
    def test_save_and_load_roundtrip(self, tmp_path) -> None:
        proj = SparseProjection(in_features=16, out_features=64, k=8, bias=True).eval()
        x = {"token_embeddings": torch.randn(2, 4, 16)}
        before = proj({k: v.clone() for k, v in x.items()})["token_embeddings"]

        proj.save(str(tmp_path))
        loaded = SparseProjection.load(str(tmp_path)).eval()
        assert loaded.get_config_dict() == proj.get_config_dict()

        after = loaded({k: v.clone() for k, v in x.items()})["token_embeddings"]
        assert torch.allclose(before, after)


class TestCrossSimilarity:
    def test_shape_is_query_by_document(self) -> None:
        queries = torch.randn(2, 4, 8)
        documents = torch.randn(2, 6, 8)
        sim = _cross_similarity(queries, documents)
        assert sim.shape == (2, 4, 6)

    def test_matches_manual_dot_products(self) -> None:
        queries = torch.randn(1, 3, 5)
        documents = torch.randn(1, 2, 5)
        sim = _cross_similarity(queries, documents)
        expected = queries[0] @ documents[0].T
        assert torch.allclose(sim[0], expected)


class _IdentityBackbone(torch.nn.Module):
    """Backbone stub that passes token embeddings through unchanged."""

    def forward(self, features):
        return features


class _SyntheticModel(torch.nn.Module):
    """Minimal stand-in for a ColBERT model: an identity backbone followed by a
    sparse projection. Lets us exercise the loss math without a download.
    """

    def __init__(self, sparse_projection: SparseProjection) -> None:
        super().__init__()
        self.backbone = _IdentityBackbone()
        self.sparse = sparse_projection
        self.skiplist = []


def _make_features(token_embeddings: torch.Tensor) -> dict:
    batch, tokens, _ = token_embeddings.shape
    return {
        "token_embeddings": token_embeddings,
        "input_ids": torch.zeros(batch, tokens, dtype=torch.long),
        "attention_mask": torch.ones(batch, tokens, dtype=torch.long),
    }


class TestSparseDistillationMath:
    def test_loss_reaches_near_zero_when_sparse_equals_dense(self) -> None:
        """If the sparse projection is the identity on non-negative inputs, the
        sparse and dense similarities coincide and the loss is ~0. Verifies the
        normalization/clamp choices make the objective achievable.
        """
        from pylate import losses

        dim = 6
        proj = SparseProjection(in_features=dim, out_features=dim, k=None).eval()
        with torch.no_grad():
            proj.linear.weight.copy_(torch.eye(dim))
        model = _SyntheticModel(proj).eval()
        loss_fn = losses.SparseDistillation(model=model, freeze_backbone=False)

        # Non-negative dense embeddings so relu is a no-op and codes == dense.
        query = _make_features(torch.rand(2, 4, dim))
        document = _make_features(torch.rand(2, 5, dim))
        loss = loss_fn(sentence_features=[query, document])
        assert float(loss) < 1e-10

    def test_loss_independent_of_padding(self) -> None:
        """Padding the documents with masked tokens must not change the loss,
        because the MSE is averaged over valid token pairs only.
        """
        from pylate import losses

        dim = 6
        torch.manual_seed(0)
        proj = SparseProjection(in_features=dim, out_features=8, k=None).eval()
        model = _SyntheticModel(proj).eval()
        loss_fn = losses.SparseDistillation(model=model, freeze_backbone=False)

        query = _make_features(torch.rand(1, 3, dim))
        document = _make_features(torch.rand(1, 4, dim))
        base = float(loss_fn(sentence_features=[query, document]))

        # Append two padding tokens to the document with attention_mask = 0.
        padded_embeddings = torch.cat(
            [document["token_embeddings"], torch.rand(1, 2, dim)], dim=1
        )
        padded = {
            "token_embeddings": padded_embeddings,
            "input_ids": torch.zeros(1, 6, dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0]]),
        }
        padded_loss = float(loss_fn(sentence_features=[query, padded]))
        assert abs(base - padded_loss) < 1e-6


def _build_model_with_sparse_projection():
    """Build a small ColBERT model with an appended sparse projection.

    Skips the test if the backbone cannot be downloaded (e.g. offline CI).
    """
    from pylate import models

    try:
        model = models.ColBERT(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )
    except Exception as exception:  # pragma: no cover - network dependent
        pytest.skip(f"Could not load backbone model: {exception}")
    embedding_size = model[-1].out_features
    model.append(SparseProjection(in_features=embedding_size, out_features=128, k=8))
    return model


class TestSparseDistillationFreeze:
    def test_only_sparse_projection_trainable(self) -> None:
        from pylate import losses

        model = _build_model_with_sparse_projection()
        losses.SparseDistillation(model=model)

        last_index = str(len(model) - 1)
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert trainable, "Expected the sparse projection to remain trainable."
        assert all(name.startswith(last_index + ".") for name in trainable)

    def test_freeze_disabled_keeps_backbone_trainable(self) -> None:
        from pylate import losses

        model = _build_model_with_sparse_projection()
        losses.SparseDistillation(model=model, freeze_backbone=False)

        backbone = list(model._modules.values())[0]
        assert any(p.requires_grad for p in backbone.parameters())


class TestSparseDistillationLoss:
    def test_loss_backpropagates_into_sparse_only(self) -> None:
        from pylate import losses

        model = _build_model_with_sparse_projection()
        loss_fn = losses.SparseDistillation(model=model)

        query = model.tokenize(["fruits are healthy."], is_query=True)
        documents = model.tokenize(
            ["fruits are good for health.", "fruits are bad for health."],
            is_query=False,
        )
        loss = loss_fn(sentence_features=[query, documents])
        assert loss.requires_grad
        assert float(loss) >= 0.0

        loss.backward()
        sparse = list(model._modules.values())[-1]
        assert sparse.linear.weight.grad is not None
        backbone = list(model._modules.values())[0]
        assert next(backbone.parameters()).grad is None
