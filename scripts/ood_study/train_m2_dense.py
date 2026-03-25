"""Train M2: Dense mean-pool bi-encoder (sentence-transformers).

Single-vector baseline with mean pooling and dot product scoring.
Serves as A4 in the H1 vs H2 decomposition.

Usage:
    python scripts/ood_study/train_m2_dense.py --seed 1
    accelerate launch --num_processes NUM_GPUS scripts/ood_study/train_m2_dense.py --seed 1
"""

from __future__ import annotations

import argparse
import os

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss


def main():
    parser = argparse.ArgumentParser(description="Train M2: Dense mean-pool")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--backbone", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--output_dir", type=str, default="output/ood_study")
    parser.add_argument("--wandb_project", type=str, default="ood-study")
    args = parser.parse_args()

    run_name = f"m2-dense-seed{args.seed}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.environ["WANDB_PROJECT"] = args.wandb_project

    # Load MS MARCO triplets (query, positive, negative)
    dataset = load_dataset(
        "sentence-transformers/msmarco-bm25", "triplet", split="train"
    )

    # Dense bi-encoder with mean pooling (sentence-transformers default)
    model = SentenceTransformer(args.backbone)

    # MultipleNegativesRankingLoss = InfoNCE with in-batch negatives
    train_loss = MultipleNegativesRankingLoss(model=model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="steps",
        eval_steps=5000,
        save_steps=5000,
        logging_steps=100,
        fp16=False,
        bf16=True,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        seed=args.seed,
        report_to="wandb",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=train_loss,
    )

    trainer.train()
    model.save_pretrained(os.path.join(output_dir, "final"))


if __name__ == "__main__":
    main()
