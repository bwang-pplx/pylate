"""Train CE: Cross-Encoder for reranking (sentence-transformers).

Trained on MS MARCO with the same backbone (ModernBERT-base) as all other
models. Used as a BM25-recall-conditioned OOD ceiling — reranks BM25 top-1000.

Usage:
    python scripts/ood_study/train_ce.py
    accelerate launch --num_processes NUM_GPUS scripts/ood_study/train_ce.py
"""

from __future__ import annotations

import argparse
import os

from datasets import load_dataset
from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss


def main():
    parser = argparse.ArgumentParser(description="Train CE: Cross-Encoder")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--backbone", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--output_dir", type=str, default="output/ood_study")
    parser.add_argument("--wandb_project", type=str, default="ood-study")
    args = parser.parse_args()

    run_name = f"ce-modernbert-seed{args.seed}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.environ["WANDB_PROJECT"] = args.wandb_project

    # Load MS MARCO triplets — CE expects (query, passage, label) pairs.
    # We convert triplets to pairs: (query, positive, 1) and (query, negative, 0).
    dataset = load_dataset(
        "sentence-transformers/msmarco-bm25", "triplet", split="train"
    )

    # Reshape triplets into pairs with binary labels
    def triplet_to_pairs(batch):
        queries = batch["query"] + batch["query"]
        passages = batch["positive"] + batch["negative"]
        labels = [1.0] * len(batch["query"]) + [0.0] * len(batch["query"])
        return {"sentence1": queries, "sentence2": passages, "label": labels}

    dataset = dataset.map(
        triplet_to_pairs,
        batched=True,
        remove_columns=dataset.column_names,
    )
    dataset = dataset.shuffle(seed=args.seed)

    # Cross-encoder: full query-document attention, single relevance score
    model = CrossEncoder(args.backbone, num_labels=1)

    train_loss = BinaryCrossEntropyLoss(model=model)

    training_args = CrossEncoderTrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
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

    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=train_loss,
    )

    trainer.train()
    model.save_pretrained(os.path.join(output_dir, "final"))


if __name__ == "__main__":
    main()
