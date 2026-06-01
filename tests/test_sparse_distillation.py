"""Tests for the sparse projection module and token-level distillation loss.

These cover Option B (token-level distillation): a SparseProjection module on
top of a frozen ColBERT backbone, trained so sparse token-token similarities
approximate the dense ones.
"""

from __future__ import annotations

import pytest
import torch

from pylate.losses.sparse_distillation import _token_similarity
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
        proj = SparseProjection(in_features=16, out_features=64)
        out = proj({"token_embeddings": torch.randn(4, 7, 16)})["token_embeddings"]
        assert bool((out >= 0).all())

    def test_topk_caps_nonzeros_per_token(self) -> None:
        k = 5
        proj = SparseProjection(in_features=16, out_features=64, k=k)
        out = proj({"token_embeddings": torch.randn(4, 7, 16)})["token_embeddings"]
        nnz = (out > 0).sum(dim=-1)
        assert int(nnz.max()) <= k

    def test_no_topk_keeps_all_positive(self) -> None:
        proj = SparseProjection(in_features=8, out_features=32, k=None)
        # Force a dense positive input through positive weights so relu keeps all.
        with torch.no_grad():
            proj.linear.weight.abs_()
        out = proj({"token_embeddings": torch.ones(2, 3, 8)})["token_embeddings"]
        assert int((out > 0).sum(dim=-1).min()) == 32

    def test_topk_keeps_largest_values(self) -> None:
        proj = SparseProjection(in_features=4, out_features=6, k=2)
        codes = torch.tensor([[[5.0, 1.0, 4.0, 0.0, 3.0, 2.0]]])
        masked = proj.topk_mask(codes)
        # Only the two largest (5.0 and 4.0) survive.
        expected = torch.tensor([[[5.0, 0.0, 4.0, 0.0, 0.0, 0.0]]])
        assert torch.allclose(masked, expected)


class TestSparseProjectionPersistence:
    def test_save_and_load_roundtrip(self, tmp_path) -> None:
        proj = SparseProjection(in_features=16, out_features=64, k=8, bias=True)
        x = {"token_embeddings": torch.randn(2, 4, 16)}
        before = proj({k: v.clone() for k, v in x.items()})["token_embeddings"]

        proj.save(str(tmp_path))
        loaded = SparseProjection.load(str(tmp_path))
        assert loaded.get_config_dict() == proj.get_config_dict()

        after = loaded({k: v.clone() for k, v in x.items()})["token_embeddings"]
        assert torch.allclose(before, after)


class TestTokenSimilarity:
    def test_masked_tokens_zeroed(self) -> None:
        embeddings = torch.randn(2, 4, 8)
        mask = torch.tensor([[1, 1, 0, 1], [1, 0, 0, 1]])
        sim = _token_similarity(embeddings, mask)
        assert sim.shape == (2, 4, 4)
        # Row/column of a masked token must be all zeros.
        assert bool((sim[0, 2] == 0).all())
        assert bool((sim[0, :, 2] == 0).all())

    def test_identical_inputs_zero_loss(self) -> None:
        embeddings = torch.randn(2, 4, 8)
        mask = torch.ones(2, 4)
        a = _token_similarity(embeddings, mask)
        b = _token_similarity(embeddings, mask)
        assert torch.allclose(a, b)


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
