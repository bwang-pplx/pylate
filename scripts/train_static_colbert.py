"""Train a static ColBERT model using Qwen2 BPE tokenizer.

Replaces the transformer encoder with a static token embedding lookup table
while keeping ColBERT's multi-vector late-interaction (MaxSim) scoring.
This gives extreme encoding speed with fine-grained token-level matching.

Usage:
    # Single GPU
    python scripts/train_static_colbert.py

    # Multi-GPU
    accelerate launch --num_processes NUM_GPUS scripts/train_static_colbert.py

    # Custom settings
    python scripts/train_static_colbert.py --embedding_dim 256 --bs 2048 --lr 2e-1
"""

from __future__ import annotations

import argparse
import os

import torch
from datasets import DatasetDict, load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.training_args import MultiDatasetBatchSamplers
from transformers import AutoTokenizer

from pylate import evaluation, losses, models, utils


# Same datasets as the full ColBERT training
DATASETS = [
    ("arguana_hn", "bowang0911/en-arguana_hn", 4065),
    ("cqadupstack_hn", "bowang0911/en-cqadupstack_hn", 50000),
    ("aila", "bowang0911/en-aila", 7028),
    ("coir_apps", "bowang0911/code-coir_apps", 2000),
    ("nv_embed_fever", "bowang0911/en-nv_embed_fever", 119320),
    ("fiqa_hn", "bowang0911/en-fiqa_hn", 9913),
    ("nfcorpus_hn", "bowang0911/en-nfcorpus_hn", 9913),
    ("nv_embed_hotpotqa", "bowang0911/en-nv_embed_hotpotqa", 137292),
    ("nv_embed_ms_marco", "bowang0911/en-nv_embed_ms-marco", 451298),
    ("nv_embed_nq", "bowang0911/en-nv_embed_nq", 65082),
    ("miracl_hn_ar", "bowang0911/ar-miracl-hn", 6217),
    ("miracl_hn_bn", "bowang0911/bn-miracl-hn", 3859),
    ("miracl_hn_en", "bowang0911/en-miracl-hn", 7899),
    ("miracl_hn_es", "bowang0911/es-miracl-hn", 10025),
    ("miracl_hn_fa", "bowang0911/fa-miracl-hn", 4277),
    ("miracl_hn_fi", "bowang0911/fi-miracl-hn", 4928),
    ("miracl_hn_fr", "bowang0911/fr-miracl-hn", 2321),
    ("miracl_hn_hi", "bowang0911/hi-miracl-hn", 2469),
    ("miracl_hn_id", "bowang0911/id-miracl-hn", 12505),
    ("miracl_hn_ja", "bowang0911/ja-miracl-hn", 6984),
    ("miracl_hn_ko", "bowang0911/ko-miracl-hn", 1973),
    ("miracl_hn_ru", "bowang0911/ru-miracl-hn", 10000),
    ("miracl_hn_sw", "bowang0911/sw-miracl-hn", 2687),
    ("miracl_hn_te", "bowang0911/te-miracl-hn", 4119),
    ("miracl_hn_th", "bowang0911/th-miracl-hn", 4778),
    ("miracl_hn_zh", "bowang0911/zh-miracl-hn", 3187),
]


def load_train_datasets() -> DatasetDict:
    """Load datasets and resample to match target weights."""
    train_dataset = DatasetDict()
    for split_name, repo_id, target_size in DATASETS:
        print(f"Loading {repo_id}...")
        ds = load_dataset(repo_id, split="train")
        actual_size = len(ds)

        if actual_size < target_size:
            full_repeats = target_size // actual_size
            remainder = target_size % actual_size
            indices = list(range(actual_size)) * full_repeats
            if remainder > 0:
                indices += list(range(remainder))
            ds = ds.select(indices)
        elif actual_size > target_size:
            ds = ds.shuffle(seed=42).select(range(target_size))

        train_dataset[split_name] = ds
        print(f"  {split_name}: {actual_size} -> {len(ds)} examples")
    return train_dataset


def main():
    parser = argparse.ArgumentParser(description="Train Static ColBERT")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="perplexity-ai/pplx-embed-v1-0.6b",
        help="Tokenizer to use for the static embedding table",
    )
    parser.add_argument(
        "--projection_dim",
        type=int,
        default=128,
        help="Output dimension of the Dense projection layer",
    )
    parser.add_argument(
        "--random_init",
        action="store_true",
        help="Use random embeddings instead of pretrained weights. "
        "When not set, loads the embedding table from the base model.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=None,
        help="Dimension of the static token embeddings (only used with --random_init, "
        "otherwise inferred from the pretrained model)",
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-1,
        help="Learning rate (static models benefit from high LR)",
    )
    parser.add_argument("--bs", type=int, default=2048)
    parser.add_argument("--temp", type=float, default=0.03)
    parser.add_argument("--query_length", type=int, default=32)
    parser.add_argument("--document_length", type=int, default=180)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="output/static-colbert")
    parser.add_argument("--run_name", type=str, default="static-colbert")
    parser.add_argument("--wandb_project", type=str, default="static-colbert")
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    args = parser.parse_args()

    os.environ["WANDB_PROJECT"] = args.wandb_project

    # Load datasets
    train_dataset = load_train_datasets()

    # Build static ColBERT model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    if args.random_init:
        embedding_dim = args.embedding_dim or 1024
        print(f"Using random initialization with embedding_dim={embedding_dim}")
        static_embedding = models.StaticEmbedding(
            tokenizer, embedding_dim=embedding_dim
        )
    else:
        # Load pretrained embedding weights from the base model
        from transformers import AutoModel

        print(f"Loading pretrained embedding weights from {args.tokenizer}...")
        base_model = AutoModel.from_pretrained(args.tokenizer, trust_remote_code=True)
        pretrained_weights = base_model.embed_tokens.weight.detach().clone()
        embedding_dim = pretrained_weights.shape[1]
        del base_model
        print(f"Loaded embedding table: {pretrained_weights.shape[0]} tokens x {embedding_dim} dim")
        static_embedding = models.StaticEmbedding(
            tokenizer, embedding_weights=pretrained_weights
        )

    model = models.ColBERT(
        modules=[static_embedding, models.Dense(embedding_dim, args.projection_dim)],
        device="cpu",
        query_length=args.query_length,
        document_length=args.document_length,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.1f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.1f}M")

    # Loss
    train_loss = losses.Contrastive(
        model=model,
        temperature=args.temp,
        gather_across_devices=True,
    )

    # Evaluator
    dev_evaluator = evaluation.NanoBEIREvaluator()

    # Training arguments — static models train well with large batches and high LR
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=50,
        bf16=True,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,
        report_to="wandb",
    )

    # Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
        data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
    )

    trainer.train()
    model.save_pretrained(f"{args.output_dir}/final")
    print(f"Model saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
