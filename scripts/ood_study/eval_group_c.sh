#!/bin/bash
# Evaluate Group C: C2 (30% token dropout on M5)

set -e

QOS="search"
GPUS=1
CPUS=8
MEM="64G"
TIME="24:00:00"
COMMON_DIR="output/ood_study"
LOG_DIR="logs/ood_study_eval"

mkdir -p "$LOG_DIR"

# C2: ColBERT with 30% random doc token dropout — brute-force
sbatch --job-name="eval-c2" --qos=$QOS --gres=gpu:$GPUS --cpus-per-task=$CPUS --mem=$MEM --time=$TIME \
    --chdir="$PWD" --output="$LOG_DIR/eval-c2_%j.out" --error="$LOG_DIR/eval-c2_%j.err" \
    --wrap="export TORCH_COMPILE_DISABLE=1 && python scripts/ood_study/eval_colbert_beir.py --model $COMMON_DIR/m5-colbert-seed1/final --token_dropout 0.3 --document_length 512 --dataset all"
echo "Submitted: eval-c2"
