# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0
"""Training script for sparse token-level distillation on top of a ColBERT model.

Trains a :class:`pylate.models.SparseProjection` (learnable sparse projection
appended to a frozen ColBERT backbone) with
:class:`pylate.losses.SparseDistillation`. The loss matches the
query x document cross-Gram of the sparse codes to the dense ColBERT teacher's
cross-Gram (clamped to ``[0, 1]``), so the sparse projection learns a
late-interaction-faithful sparse encoder without requiring a decoder.

Example
-------

Quick smoke test on a small public distillation dataset::

    python examples/train/sparse_distillation.py \\
        --model_name lightonai/GTE-ModernColBERT-v1 \\
        --dataset_name sentence-transformers/msmarco-bm25 --dataset_config triplet \\
        --sparse_dim 16384 --k 32 \\
        --batch_size 8 --stop_at_step 50

The dataset must be a *text* triplet set (``query`` / ``positive`` / ``negative*``
columns); ``sentence-transformers/msmarco-bm25`` has multi-negative configs too
(e.g. ``--dataset_config triplet-50``). Override ``--dataset_name`` /
``--dataset_config`` to use a richer triplet set you have access to.
"""
from __future__ import annotations

import argparse
import itertools
import random
from typing import Callable

import torch
from accelerate.utils import set_seed
from datasets import DatasetDict, load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.training_args import MultiDatasetBatchSamplers
from transformers import TrainerCallback

from pylate import evaluation, models
from pylate.losses import SparseDistillation
from pylate.models import SparseProjection
from pylate.utils import AnisotropyCallback

set_seed(42)

# ---------------------------------------------------------------------------
# Collator (loss-agnostic; samples a fixed number of negatives per batch)
# ---------------------------------------------------------------------------


class ColBERTCollatorSampleNeg:
    """Collator for ColBERT that samples a random subset of negatives per batch."""

    def __init__(
        self,
        tokenize_fn: Callable,
        valid_label_columns: list[str] | None = None,
        num_negatives: int = 7,
    ) -> None:
        self.tokenize_fn = tokenize_fn
        self.num_negatives = num_negatives
        if valid_label_columns is None:
            valid_label_columns = ["label", "scores"]
        self.valid_label_columns = valid_label_columns

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch = {"return_loss": True}
        columns = list(features[0].keys())

        if "dataset_name" in columns:
            columns.remove("dataset_name")
            batch["dataset_name"] = features[0]["dataset_name"]

        for label_column in self.valid_label_columns:
            if label_column in columns:
                batch["label"] = torch.tensor([row[label_column] for row in features])
                columns.remove(label_column)
                break

        negative_columns = [col for col in columns if col.startswith("negative_")]
        other_columns = [col for col in columns if not col.startswith("negative_")]

        if self.num_negatives is not None and negative_columns:
            k = min(self.num_negatives, len(negative_columns))
            sampled_negatives = random.sample(negative_columns, k)
            columns_to_process = other_columns + sampled_negatives
        else:
            columns_to_process = columns

        for column in columns_to_process:
            if "_id" not in column:
                is_query = "query" in column or "anchor" in column
                texts = [row[column] for row in features]
                if isinstance(texts[0], list):
                    texts = list(itertools.chain(*texts))
                tokenized = self.tokenize_fn(texts, is_query=is_query, pad=True)
                for key, value in tokenized.items():
                    batch[f"{column}_{key}"] = value

        return batch


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_train_datasets(
    dataset_name: str = "sentence-transformers/msmarco-bm25",
    config_name: str | None = "triplet",
    splits: list[str] | None = None,
) -> DatasetDict:
    """Load a triplet-text dataset (query / positive / negative_*).

    ``SparseDistillation`` consumes one document group per text column, so this
    expects a *text* triplet dataset (each row has ``query`` plus ``positive`` /
    ``negative*`` strings), not the id-based knowledge-distillation format
    (``query_id`` / ``document_ids`` / ``scores``). ``config_name`` is the HF
    dataset config (e.g. ``"triplet"`` for ``sentence-transformers/msmarco-bm25``);
    pass ``None`` for datasets that have no named configs.
    """
    train_dataset = DatasetDict()
    if splits is None:
        splits = ["train"]
    for split in splits:
        train_dataset[split] = load_dataset(dataset_name, config_name, split=split)
    return train_dataset


# ---------------------------------------------------------------------------
# Combined loss (weighted sum of several losses sharing one model)
# ---------------------------------------------------------------------------


class CombinedLoss(torch.nn.Module):
    """Weighted sum of losses that share the same model.

    Used to add a structure-preserving auxiliary (distillation or
    reconstruction) on top of the supervised contrastive ranking loss.
    """

    def __init__(self, losses: list, weights: list[float]) -> None:
        super().__init__()
        self.losses = torch.nn.ModuleList(losses)
        self.weights = weights

    def forward(self, sentence_features, labels=None):
        total = None
        for loss_fn, weight in zip(self.losses, self.weights):
            value = weight * loss_fn(sentence_features=sentence_features, labels=labels)
            total = value if total is None else total + value
        return total


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class StopAtStepCallback(TrainerCallback):
    """Stop training once ``stop_at_step`` global steps have been reached."""

    def __init__(self, stop_at_step: int):
        self.stop_at_step = stop_at_step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_at_step:
            print(f"\n  Reached target step {self.stop_at_step}. Stopping training...")
            control.should_training_stop = True
        return control


class ChunkedNanoBEIREvaluator(evaluation.NanoBEIREvaluator):
    """NanoBEIR evaluator that caps ``corpus_chunk_size`` on its sub-evaluators.

    The dense IR evaluator pads an entire corpus chunk of token-embedding
    matrices into one tensor. With high-dimensional sparse codes (e.g. 16384)
    the default chunk size (50000) needs ~90 GB and OOMs. Injecting a small
    ``corpus_chunk_size`` keeps each padded chunk small; the sparse dimension is
    contracted away in MaxSim, so the metrics are unchanged.
    """

    def __init__(self, *args, corpus_chunk_size: int = 256, **kwargs) -> None:
        self._corpus_chunk_size = corpus_chunk_size
        super().__init__(*args, **kwargs)

    def _load_dataset(self, dataset_name, **ir_evaluator_kwargs):
        ir_evaluator_kwargs.setdefault("corpus_chunk_size", self._corpus_chunk_size)
        return super()._load_dataset(dataset_name, **ir_evaluator_kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train sparse token-level distillation: SparseProjection + "
            "SparseDistillation on top of a frozen ColBERT backbone."
        )
    )

    # Backbone
    parser.add_argument(
        "--model_name", type=str, default="lightonai/GTE-ModernColBERT-v1"
    )
    parser.add_argument("--document_length", type=int, default=300)
    parser.add_argument(
        "--no_freeze_backbone",
        dest="freeze_backbone",
        action="store_false",
        help="If set, also train the ColBERT backbone (default: backbone frozen).",
    )
    parser.set_defaults(freeze_backbone=True)

    # SparseProjection architecture
    parser.add_argument(
        "--sparse_dim",
        type=int,
        default=16384,
        help="Sparse code dimension (out_features of SparseProjection).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=32,
        help="TopK sparsity level. Set to 0 to disable the hard TopK mask.",
    )
    parser.add_argument(
        "--activation", type=str, default="relu", choices=["relu", "softplus"]
    )
    parser.add_argument("--bias", action="store_true", default=False)
    parser.add_argument(
        "--sparse_init",
        type=str,
        default="default",
        choices=["default", "orthogonal"],
        help="SparseProjection init. 'orthogonal' gives a spread/selective LSH-style start.",
    )

    # SparseDistillation knobs
    parser.add_argument(
        "--no_clamp_dense_target",
        dest="clamp_dense_target",
        action="store_false",
        help="Disable clamping the dense teacher cross-similarities to [0, 1].",
    )
    parser.set_defaults(clamp_dense_target=True)
    parser.add_argument(
        "--flops_lambda",
        type=float,
        default=0.0,
        help=(
            "Weight of the SPLADE-style FLOPS regularizer that spreads activations "
            "across dimensions (short posting lists). 0 = pure distillation."
        ),
    )
    parser.add_argument(
        "--flops_warmup_steps",
        type=int,
        default=0,
        help="Quadratically ramp flops_lambda over this many steps (0 = no warmup).",
    )
    parser.add_argument(
        "--flops_on_normalized",
        dest="flops_on_raw",
        action="store_false",
        help="Compute FLOPS on L2-normalized codes instead of raw (weaker signal).",
    )
    parser.set_defaults(flops_on_raw=True)
    parser.add_argument(
        "--center_dense",
        action="store_true",
        default=False,
        help=(
            "Subtract the batch mean of dense token embeddings from the teacher "
            "before building the target (removes the anisotropic common component)."
        ),
    )
    parser.add_argument(
        "--distillation_mode",
        type=str,
        default="token",
        choices=["token", "score"],
        help=(
            "'token': MSE over the full token-token similarity matrix. 'score': "
            "KL over per-document MaxSim scores (ranking-aware, like ColBERT KD)."
        ),
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="distillation",
        choices=["distillation", "reconstruction", "contrastive"],
        help=(
            "'distillation': match the dense teacher's token similarities. "
            "'reconstruction': self-supervised sparse autoencoder. "
            "'contrastive': supervised contrastive on the sparse codes "
            "(positive vs negatives) -- trains only the sparse head (backbone "
            "frozen), the cheap label-supervised option."
        ),
    )
    parser.add_argument(
        "--contrastive_temperature",
        type=float,
        default=0.05,
        help="Temperature for the contrastive objective (lower = sharper).",
    )
    parser.add_argument(
        "--aux_objective",
        type=str,
        default="none",
        choices=["none", "distillation", "reconstruction"],
        help=(
            "Auxiliary structure-preserving loss added to 'contrastive': "
            "'distillation' (match dense token similarities, with --center_dense) "
            "or 'reconstruction' (sparse autoencoder, with --ortho_lambda)."
        ),
    )
    parser.add_argument(
        "--aux_lambda",
        type=float,
        default=0.0,
        help="Weight of the auxiliary loss added to the contrastive objective.",
    )
    parser.add_argument(
        "--reconstruct_raw",
        dest="reconstruct_normalized",
        action="store_false",
        help="Reconstruct raw (not L2-normalized) dense embeddings.",
    )
    parser.set_defaults(reconstruct_normalized=True)
    parser.add_argument(
        "--ortho_lambda",
        type=float,
        default=0.0,
        help=(
            "Reconstruction: weight of the ||WtW - I|| orthogonality regularizer "
            "that keeps the projection similarity-preserving (stops the peak-then-"
            "drop drift)."
        ),
    )

    # Dataset
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="sentence-transformers/msmarco-bm25",
        help="HuggingFace dataset id with query/positive/negative_* text columns.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="triplet",
        help=(
            "HuggingFace dataset config name (e.g. 'triplet'). Pass 'none' for "
            "datasets without named configs."
        ),
    )
    parser.add_argument("--dataset_splits", type=str, nargs="+", default=["train"])

    # Optimization
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--num_negatives", type=int, default=7)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")

    # Eval / logging / saving
    parser.add_argument(
        "--no_eval",
        dest="eval_during_training",
        action="store_false",
        help="Disable the in-loop NanoBEIR evaluator (enabled by default).",
    )
    parser.set_defaults(eval_during_training=True)
    parser.add_argument(
        "--no_eval_on_start",
        dest="eval_on_start",
        action="store_false",
        help="Skip the step-0 baseline evaluation (run by default).",
    )
    parser.set_defaults(eval_on_start=True)
    parser.add_argument(
        "--eval_corpus_chunk_size",
        type=int,
        default=256,
        help=(
            "Number of corpus docs padded/scored at once during NanoBEIR eval. "
            "The dense evaluator pads a whole chunk of token-embedding matrices "
            "into one tensor; a small value keeps the high-dimensional sparse "
            "corpus tensor (e.g. 16384-dim) from OOMing. The 16384 dim is "
            "contracted in MaxSim, so this does not change the metrics."
        ),
    )
    parser.add_argument(
        "--track_anisotropy",
        action="store_true",
        default=False,
        help="Log token-level anisotropy (mean_norm/eff_rank/mean_pair_cos) every logging_steps.",
    )
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--stop_at_step", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="Trainer reporters (e.g. 'none', 'tensorboard', 'wandb').",
    )

    args = parser.parse_args()

    # --- Load datasets ---
    dataset_config = None if args.dataset_config.lower() == "none" else args.dataset_config
    print(
        f"Loading dataset: {args.dataset_name} "
        f"(config={dataset_config}, splits={args.dataset_splits})"
    )
    train_dataset = load_train_datasets(
        dataset_name=args.dataset_name,
        config_name=dataset_config,
        splits=args.dataset_splits,
    )
    print(train_dataset)

    # --- Load ColBERT backbone ---
    colbert = models.ColBERT(args.model_name, document_length=args.document_length)
    input_dim = colbert.get_sentence_embedding_dimension()
    print(f"ColBERT embedding dim: {input_dim}")

    # --- Append SparseProjection as the last module ---
    # SparseDistillation locates the sparse module via ``model._modules`` at
    # ``sparse_projection_index=-1``, so we must append it last.
    k = args.k if args.k > 0 else None
    sparse = SparseProjection(
        in_features=input_dim,
        out_features=args.sparse_dim,
        k=k,
        bias=args.bias,
        activation=args.activation,
        init=args.sparse_init,
    )
    colbert.append(sparse)
    print(
        f"Appended SparseProjection: in={input_dim} out={args.sparse_dim} "
        f"k={k} activation={args.activation}"
    )

    # --- Loss ---
    # freeze_backbone=True flips requires_grad so only the sparse projection is
    # trainable; the standard AdamW optimizer below then only updates those
    # parameters.
    if args.objective == "reconstruction":
        from pylate.losses import SparseReconstruction

        loss = SparseReconstruction(
            model=colbert,
            sparse_projection_index=-1,
            freeze_backbone=args.freeze_backbone,
            normalize_target=args.reconstruct_normalized,
            ortho_lambda=args.ortho_lambda,
        )
    elif args.objective == "contrastive":
        from pylate.losses import Contrastive

        # Contrastive does not freeze anything; freeze the backbone here so only
        # the sparse head trains (label-supervised, backbone fixed).
        if args.freeze_backbone:
            sparse_module = colbert[-1]
            for module in colbert._modules.values():
                trainable = module is sparse_module
                for parameter in module.parameters():
                    parameter.requires_grad = trainable
        contrastive = Contrastive(
            model=colbert,
            temperature=args.contrastive_temperature,
            gather_across_devices=True,
        )
        if args.aux_objective == "none":
            loss = contrastive
        elif args.aux_objective == "distillation":
            aux = SparseDistillation(
                model=colbert,
                freeze_backbone=False,  # backbone already frozen above
                clamp_dense_target=args.clamp_dense_target,
                center_dense=args.center_dense,
                distillation_mode=args.distillation_mode,
                flops_lambda=args.flops_lambda,
                flops_warmup_steps=args.flops_warmup_steps,
            )
            loss = CombinedLoss([contrastive, aux], [1.0, args.aux_lambda])
        else:  # reconstruction
            from pylate.losses import SparseReconstruction

            aux = SparseReconstruction(
                model=colbert,
                freeze_backbone=False,
                normalize_target=args.reconstruct_normalized,
                ortho_lambda=args.ortho_lambda,
            )
            loss = CombinedLoss([contrastive, aux], [1.0, args.aux_lambda])
    else:
        loss = SparseDistillation(
            model=colbert,
            sparse_projection_index=-1,
            freeze_backbone=args.freeze_backbone,
            clamp_dense_target=args.clamp_dense_target,
            flops_lambda=args.flops_lambda,
            flops_warmup_steps=args.flops_warmup_steps,
            flops_on_raw=args.flops_on_raw,
            center_dense=args.center_dense,
            distillation_mode=args.distillation_mode,
        )

    n_trainable = sum(p.numel() for p in colbert.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable:,}")

    # --- Evaluator ---
    # NanoBEIR with a small corpus_chunk_size so the high-dimensional sparse
    # corpus tensor does not OOM (see ChunkedNanoBEIREvaluator). Evaluated at
    # step 0 (baseline) and every eval_steps. For a full-scale, index-based
    # sparse eval use examples/evaluation/sparse_distillation_beir.py.
    dev_evaluator = (
        ChunkedNanoBEIREvaluator(corpus_chunk_size=args.eval_corpus_chunk_size)
        if args.eval_during_training
        else None
    )

    # --- Run name / output dir ---
    model_shortname = args.model_name.split("/")[-1]
    run_name = args.run_name or (
        f"SparseDistill-{model_shortname}-"
        f"d{args.sparse_dim}-k{args.k}-"
        f"lr{args.learning_rate:.0e}-bs{args.batch_size}"
    )
    output_dir = args.output_dir or f"output/{model_shortname}/{run_name}"

    print(f"\n{'=' * 60}")
    print("Sparse Distillation Training Configuration:")
    print(f"{'=' * 60}")
    print(f"Backbone: {args.model_name} (frozen={args.freeze_backbone})")
    print(
        f"Sparse: dim={args.sparse_dim}, k={args.k}, activation={args.activation}"
    )
    print(f"Clamp dense target: {args.clamp_dense_target}")
    print(
        f"Optim: lr={args.learning_rate}, wd={args.weight_decay}, bs={args.batch_size}, epochs={args.num_train_epochs}"
    )
    print(f"Negatives per batch: {args.num_negatives}")
    print(f"Run name: {run_name}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}\n")

    # --- Training args ---
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
        eval_strategy="steps" if dev_evaluator is not None else "no",
        eval_steps=args.eval_steps,
        eval_on_start=args.eval_on_start if dev_evaluator is not None else False,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        fp16=False,
        bf16=args.bf16,
        seed=42,
        report_to=args.report_to,
        run_name=run_name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        dataloader_num_workers=4,
        accelerator_config={"split_batches": True},
    )

    # --- Collator ---
    data_collator = ColBERTCollatorSampleNeg(
        tokenize_fn=colbert.tokenize, num_negatives=args.num_negatives
    )

    # --- Callbacks ---
    callbacks: list[TrainerCallback] = []
    if args.stop_at_step > 0:
        callbacks.append(StopAtStepCallback(args.stop_at_step))
    if args.track_anisotropy:
        probe = list(train_dataset[args.dataset_splits[0]]["positive"][:256])
        callbacks.append(
            AnisotropyCallback(
                colbert, probe_texts=probe, every_n_steps=args.logging_steps
            )
        )

    # --- Trainer ---
    # SparseDistillation already set requires_grad correctly, so the default
    # SentenceTransformerTrainer/AdamW only updates SparseProjection params.
    trainer = SentenceTransformerTrainer(
        model=colbert,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=dev_evaluator,
        callbacks=callbacks if callbacks else None,
        data_collator=data_collator,
    )

    trainer.train()

    # --- Save ---
    # SparseProjection is part of colbert._modules, so save_pretrained writes
    # it alongside the backbone. Reload with models.ColBERT(<path>) and
    # model.encode(...) will run the full backbone + sparse projection pipeline.
    final_path = f"{output_dir}/final"
    colbert.save_pretrained(final_path)
    print(f"\n{'=' * 60}")
    print("Training completed!")
    print(f"Model saved to: {final_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
