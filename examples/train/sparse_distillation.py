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
        --dataset_name lightonai/ms-marco-en-bge-gemma \\
        --sparse_dim 16384 --k 32 \\
        --batch_size 8 --stop_at_step 50

Override ``--dataset_name`` to use a richer distillation set you have access to
(e.g. ``lightonai/nv-embed-supervised-distill-dedup``).
"""
from __future__ import annotations

import argparse
import itertools
import random
from typing import Callable

import torch
from accelerate.utils import set_seed

set_seed(42)

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
    dataset_name: str = "lightonai/ms-marco-en-bge-gemma",
    splits: list[str] | None = None,
) -> DatasetDict:
    """Load a PyLate-compatible distillation dataset (query / positive / negative_*)."""
    train_dataset = DatasetDict()
    if splits is None:
        splits = ["train"]
    for split in splits:
        train_dataset[split] = load_dataset(dataset_name, split=split)
    return train_dataset


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

    # SparseDistillation knobs
    parser.add_argument(
        "--no_clamp_dense_target",
        dest="clamp_dense_target",
        action="store_false",
        help="Disable clamping the dense teacher cross-similarities to [0, 1].",
    )
    parser.set_defaults(clamp_dense_target=True)

    # Dataset
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="lightonai/ms-marco-en-bge-gemma",
        help="HuggingFace dataset id with query/positive/negative_* columns.",
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
    print(f"Loading dataset: {args.dataset_name} (splits={args.dataset_splits})")
    train_dataset = load_train_datasets(
        dataset_name=args.dataset_name, splits=args.dataset_splits
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
    loss = SparseDistillation(
        model=colbert,
        sparse_projection_index=-1,
        freeze_backbone=args.freeze_backbone,
        clamp_dense_target=args.clamp_dense_target,
    )

    n_trainable = sum(p.numel() for p in colbert.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable:,}")

    # --- Evaluator ---
    # Standard NanoBEIR uses cosine MaxSim, matching SparseDistillation's
    # training target (queries/docs are L2-normalised inside the loss).
    dev_evaluator = evaluation.NanoBEIREvaluator()

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
        eval_strategy="steps",
        eval_steps=args.eval_steps,
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
