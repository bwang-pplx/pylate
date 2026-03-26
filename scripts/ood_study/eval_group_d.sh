#!/bin/bash
# Evaluate D2 (dim=64 ColBERT)

set -e

QOS="search"
GPUS=1
CPUS=8
MEM="64G"
TIME="24:00:00"
COMMON_DIR="output/ood_study"
LOG_DIR="logs/ood_study_eval"

mkdir -p "$LOG_DIR"

# D2 (dim=64 ColBERT) — PLAID eval
sbatch --job-name="eval-d2" --qos=$QOS --gres=gpu:$GPUS --cpus-per-task=$CPUS --mem=$MEM --time=$TIME \
    --chdir="$PWD" --output="$LOG_DIR/eval-d2_%j.out" --error="$LOG_DIR/eval-d2_%j.err" \
    --wrap="export TORCH_COMPILE_DISABLE=1 && python scripts/eval_beir.py --model $COMMON_DIR/d2-dim64-seed1/final --document_length 512 --dataset all"
echo "Submitted: eval-d2"
