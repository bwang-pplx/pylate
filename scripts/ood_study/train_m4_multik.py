"""Train M4: Multi-K (stride mean-pool → K vectors + MaxSim).

Splits N token embeddings into K equal segments, mean-pools each segment,
and uses MaxSim over the K resulting vectors. No extra learned parameters.

NOTE: Requires stride-pool implementation in PyLate (not yet available).
See the forward pass modification needed in pylate/models/colbert.py.

Usage:
    python scripts/ood_study/train_m4_multik.py --K 8 --seed 1
    accelerate launch --num_processes NUM_GPUS scripts/ood_study/train_m4_multik.py --K 8 --seed 1
"""

from __future__ import annotations

import argparse
import os

import torch
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import evaluation, losses, models, utils


def main():
    parser = argparse.ArgumentParser(description="Train M4: Multi-K")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--K", type=int, default=8, help="Number of pooled vectors per document")
    parser.add_argument("--backbone", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--query_length", type=int, default=32)
    parser.add_argument("--document_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="output/ood_study")
    parser.add_argument("--wandb_project", type=str, default="ood-study")
    args = parser.parse_args()

    run_name = f"m4-K{args.K}-seed{args.seed}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.environ["WANDB_PROJECT"] = args.wandb_project

    dataset = load_dataset(
        "sentence-transformers/msmarco-bm25", "triplet", split="train"
    )

    # TODO: Pass pool_k=args.K to ColBERT model once stride-pool is implemented.
    # The model should apply stride mean-pool after the projection layer,
    # reducing N token vectors to K vectors for documents (not queries).
    model = models.ColBERT(
        model_name_or_path=args.backbone,
        embedding_size=args.embedding_size,
        query_length=args.query_length,
        document_length=args.document_length,
    )
    train_loss = losses.Contrastive(
        model=model,
        temperature=args.temperature,
        gather_across_devices=True,
    )

    dev_evaluator = evaluation.NanoBEIREvaluator()

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
        accelerator_config={"split_batches": True},
        report_to="wandb",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
        data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
    )

    trainer.train()
    model.save_pretrained(os.path.join(output_dir, "final"))


if __name__ == "__main__":
    main()
