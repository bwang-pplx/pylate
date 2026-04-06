"""Ablation: SelfDistillation vs Contrastive on ModernBERT-base + MS MARCO.

Runs three experiments with identical hyperparameters, only changing the loss:
  1. Contrastive (baseline)
  2. SelfDistillation (DINO-style, no truncation)
  3. SelfDistillation + top_k truncation (SSD-style support compression)

Evaluates on NanoBEIR every 250 steps. Results saved to output/ablation-*/.

Dataset: MS MARCO triplets (502,931 samples)
Hardware: 8x H200 GPUs
Global batch size: 1024 (128 per device x 8 GPUs)

Usage:
  # Run all three experiments
  accelerate launch --num_processes 8 scripts/ablation_self_distillation.py --loss all

  # Run one experiment
  accelerate launch --num_processes 8 scripts/ablation_self_distillation.py --loss contrastive
"""

from __future__ import annotations

import argparse
import os

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import evaluation, losses, models, utils


def train_run(
    loss_name: str,
    model_name: str = "answerdotai/ModernBERT-base",
    batch_size: int = 128,
    lr: float = 3e-5,
    num_train_epochs: int = 3,
    eval_steps: int = 250,
    save_steps: int = 500,
    logging_steps: int = 10,
    document_length: int = 180,
    T_teacher: float = 1.0,
    T_student: float = 2.0,
    top_k: int | None = None,
    momentum: float = 0.999,
    contrastive_temperature: float = 0.03,
    wandb_project: str = "ablation-self-distillation",
):
    run_name = f"ablation-{loss_name}-modernbert-base"
    output_dir = f"output/{run_name}"

    os.environ["WANDB_PROJECT"] = wandb_project

    # Model
    model = models.ColBERT(
        model_name_or_path=model_name,
        document_length=document_length,
    )

    # Dataset: MS MARCO triplets (query, positive, negative) — 502,931 samples
    # Load on main process first to avoid race condition on split caching
    dataset = load_dataset(
        "sentence-transformers/msmarco-bm25", "triplet", split="train"
    )
    splits = dataset.train_test_split(test_size=0.01, seed=42)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]
    # Force cache write before other processes try to read
    train_dataset.flatten_indices()
    eval_dataset.flatten_indices()

    # Loss
    if loss_name == "contrastive":
        train_loss = losses.Contrastive(
            model=model,
            temperature=contrastive_temperature,
            gather_across_devices=True,
        )
    elif loss_name == "self-distill":
        train_loss = losses.SelfDistillation(
            model=model,
            T_teacher=T_teacher,
            T_student=T_student,
            momentum=momentum,
            gather_across_devices=True,
        )
    elif loss_name == "self-distill-topk":
        train_loss = losses.SelfDistillation(
            model=model,
            T_teacher=T_teacher,
            T_student=T_student,
            top_k=top_k,
            momentum=momentum,
            gather_across_devices=True,
        )
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    # Evaluator: NanoBEIR (13 datasets, no index needed)
    dev_evaluator = evaluation.NanoBEIREvaluator()

    # Training args
    # 8x H200: 128 per device x 8 = 1024 global batch
    # gather_across_devices=True: 1024 in-batch negatives
    # Steps per epoch: ~500K / 1024 ≈ 490
    # Total steps: 490 * 3 ≈ 1470
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        fp16=False,
        bf16=True,
        run_name=run_name,
        learning_rate=lr,
        warmup_ratio=0.1,
        seed=42,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,
        report_to="wandb",
    )

    # Train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
        data_collator=utils.ColBERTCollator(model.tokenize),
    )

    trainer.train()
    model.save_pretrained(f"{output_dir}/final")
    print(f"[{loss_name}] Done. Model saved to {output_dir}/final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation: SelfDistillation vs Contrastive"
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="all",
        choices=["contrastive", "self-distill", "self-distill-topk", "all"],
        help="Which loss to run. 'all' runs all three sequentially.",
    )
    parser.add_argument("--model", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--document-length", type=int, default=180)
    parser.add_argument("--contrastive-temperature", type=float, default=0.03)
    parser.add_argument("--wandb-project", type=str, default="ablation-self-distillation")

    # SelfDistillation hyperparameters
    parser.add_argument("--T-teacher", type=float, default=1.0)
    parser.add_argument("--T-student", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--momentum", type=float, default=0.999)

    args = parser.parse_args()

    run_losses = (
        ["contrastive", "self-distill", "self-distill-topk"]
        if args.loss == "all"
        else [args.loss]
    )

    for loss_name in run_losses:
        print(f"\n{'='*60}")
        print(f"Running: {loss_name}")
        print(f"{'='*60}\n")
        train_run(
            loss_name=loss_name,
            model_name=args.model,
            batch_size=args.batch_size,
            lr=args.lr,
            num_train_epochs=args.epochs,
            eval_steps=args.eval_steps,
            document_length=args.document_length,
            T_teacher=args.T_teacher,
            T_student=args.T_student,
            top_k=args.top_k if loss_name == "self-distill-topk" else None,
            momentum=args.momentum,
            contrastive_temperature=args.contrastive_temperature,
            wandb_project=args.wandb_project,
        )
