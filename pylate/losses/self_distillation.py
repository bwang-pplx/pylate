from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..models import ColBERT
from ..scores import colbert_scores
from ..utils import all_gather, all_gather_with_gradients, get_rank, get_world_size
from .contrastive import extract_skiplist_mask


class SelfDistillation(nn.Module):
    """DINO-style self-distillation loss for ColBERT with optional SSD-style truncation.

    Uses an EMA (momentum) teacher to produce soft targets from the model's own
    scores. Centering prevents collapse, asymmetric temperatures provide a richer
    signal than hard contrastive labels, and optional top-k truncation performs
    SSD-style support compression.

    Input format is the same as ``Contrastive``: query + positive pairs with
    in-batch negatives.

    Parameters
    ----------
    model
        ColBERT model (serves as the student).
    score_metric
        ColBERT scoring function. Defaults to colbert_scores.
    size_average
        Average by the size of the mini-batch.
    gather_across_devices
        Whether to gather embeddings across devices for more in-batch negatives.
    T_teacher
        Teacher temperature. Lower values produce sharper (more confident)
        targets. Should be less than T_student. Note: ColBERT scores are
        sums of per-token max cosine similarities (typical range ~10–25),
        much larger than single cosine similarities. Temperatures must be
        high enough (~1.0) that softmax produces non-degenerate distributions.
        The Contrastive loss uses 0.03, but that works only with hard labels
        (cross-entropy), not soft targets (KL divergence).
    T_student
        Student temperature. Higher values produce softer predictions.
    top_k
        If set, only the top-k scoring documents per query are retained in the
        teacher distribution (SSD-style support compression). The rest get
        ``-inf`` before softmax, resulting in zero probability. If None, all
        documents contribute to the teacher targets (pure DINO).
    momentum
        EMA momentum for teacher updates. Values close to 1.0 mean the teacher
        changes slowly (more stable targets).
    center_momentum
        EMA momentum for the centering running mean.

    Examples
    --------
    >>> from pylate import models, losses

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    ... )

    >>> loss = losses.SelfDistillation(model=model)

    >>> anchor = model.tokenize([
    ...     "fruits are healthy.",
    ...     "the weather is nice.",
    ... ], is_query=True)

    >>> positive = model.tokenize([
    ...     "fruits are good for health.",
    ...     "it is sunny outside.",
    ... ], is_query=False)

    >>> sentence_features = [anchor, positive]

    >>> loss_value = loss(sentence_features=sentence_features)
    >>> assert isinstance(loss_value.item(), float)
    """

    def __init__(
        self,
        model: ColBERT,
        score_metric=colbert_scores,
        size_average: bool = True,
        gather_across_devices: bool = False,
        T_teacher: float = 1.0,
        T_student: float = 2.0,
        top_k: int | None = None,
        momentum: float = 0.999,
        center_momentum: float = 0.9,
    ) -> None:
        super(SelfDistillation, self).__init__()
        self.model = model
        self.score_metric = score_metric
        self.size_average = size_average
        self.gather_across_devices = gather_across_devices
        self.T_teacher = T_teacher
        self.T_student = T_student
        self.top_k = top_k
        self.momentum = momentum
        self.center_momentum = center_momentum

        # EMA teacher: a frozen copy of the student, updated via momentum
        self.teacher = deepcopy(model)
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Running center for teacher score centering (prevents collapse)
        self.register_buffer("center", torch.zeros(1))

    @torch.no_grad()
    def _ema_update_teacher(self) -> None:
        """Update teacher parameters as EMA of student parameters."""
        student = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        for t_param, s_param in zip(
            self.teacher.parameters(), student.parameters()
        ):
            t_param.data.mul_(self.momentum).add_(
                s_param.data, alpha=1 - self.momentum
            )

    @torch.no_grad()
    def _update_center(self, teacher_scores: torch.Tensor) -> None:
        """Update the running center with current batch's mean teacher score."""
        batch_center = teacher_scores.mean()
        self.center = (
            self.center * self.center_momentum
            + batch_center * (1 - self.center_momentum)
        )

    def _build_teacher_targets(self, teacher_scores: torch.Tensor) -> torch.Tensor:
        """Build soft teacher targets with centering, optional truncation, and sharpening.

        Pipeline: raw scores -> center -> truncate (optional) -> sharp softmax.

        Parameters
        ----------
        teacher_scores
            Raw teacher ColBERT scores of shape (batch_size, n_documents).

        Returns
        -------
        torch.Tensor
            Teacher probability distribution of shape (batch_size, n_documents).
        """
        # Centering: prevents collapse
        centered = teacher_scores - self.center

        # Support compression: keep only top-k, rest -> -inf
        if self.top_k is not None and self.top_k < centered.size(1):
            top_k_values, _ = torch.topk(centered, self.top_k, dim=1)
            threshold = top_k_values[:, -1:]
            centered = centered.masked_fill(centered < threshold, float("-inf"))

        # Sharpening: low temperature softmax
        return F.softmax(centered / self.T_teacher, dim=-1)

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the self-distillation loss.

        Parameters
        ----------
        sentence_features
            List of tokenized sentences. The first element is the anchor
            (queries), the rest are positive and optional negative groups.
            In-batch negatives are used automatically.
        labels
            Ignored. Kept for compatibility with SentenceTransformerTrainer.
        """
        # --- Student encoding (with gradients) ---
        student_embeddings = [
            F.normalize(
                self.model(sentence_feature)["token_embeddings"], p=2, dim=-1
            )
            for sentence_feature in sentence_features
        ]

        # --- Teacher encoding (no gradients, eval mode like DINO) ---
        self.teacher.eval()
        with torch.no_grad():
            teacher_embeddings = [
                F.normalize(
                    self.teacher(sentence_feature)["token_embeddings"],
                    p=2,
                    dim=-1,
                )
                for sentence_feature in sentence_features
            ]

        # Handle (D)DP wrapped models
        skiplist = (
            self.model.skiplist
            if hasattr(self.model, "skiplist")
            else self.model.module.skiplist
        )
        do_query_expansion = (
            self.model.do_query_expansion
            if hasattr(self.model, "do_query_expansion")
            else self.model.module.do_query_expansion
        )
        masks = extract_skiplist_mask(
            sentence_features=sentence_features, skiplist=skiplist
        )

        batch_size = student_embeddings[0].size(0)

        # Gather across devices for more in-batch negatives
        if self.gather_across_devices:
            student_embeddings = [
                student_embeddings[0],
                *[
                    torch.cat(all_gather_with_gradients(emb))
                    for emb in student_embeddings[1:]
                ],
            ]
            teacher_embeddings = [
                teacher_embeddings[0],
                *[
                    torch.cat(all_gather(emb))
                    for emb in teacher_embeddings[1:]
                ],
            ]
            masks = [
                masks[0],
                *[torch.cat(all_gather(mask)) for mask in masks[1:]],
            ]

        queries_mask = masks[0] if not do_query_expansion else None

        # --- Score matrices (batch_queries x batch_documents) ---
        student_scores = torch.cat(
            [
                self.score_metric(
                    student_embeddings[0],
                    group_emb,
                    queries_mask=queries_mask,
                    documents_mask=doc_mask,
                )
                for group_emb, doc_mask in zip(
                    student_embeddings[1:], masks[1:]
                )
            ],
            dim=1,
        )

        with torch.no_grad():
            teacher_scores = torch.cat(
                [
                    self.score_metric(
                        teacher_embeddings[0],
                        group_emb,
                        queries_mask=queries_mask,
                        documents_mask=doc_mask,
                    )
                    for group_emb, doc_mask in zip(
                        teacher_embeddings[1:], masks[1:]
                    )
                ],
                dim=1,
            )

        # --- Teacher targets: centering -> truncation -> sharpening ---
        P_t = self._build_teacher_targets(teacher_scores)

        # --- Student predictions: softer temperature ---
        P_s = F.log_softmax(student_scores / self.T_student, dim=-1)

        # Cross-entropy loss: H(P_t, P_s) = -sum(P_t * log(P_s))
        loss = torch.sum(-P_t * P_s, dim=-1)
        loss = loss.mean() if self.size_average else loss.sum()

        # Scale by world size when gathering across devices
        if self.gather_across_devices:
            loss *= get_world_size()

        # --- Update teacher EMA and center ---
        self._ema_update_teacher()
        self._update_center(teacher_scores)

        return loss
