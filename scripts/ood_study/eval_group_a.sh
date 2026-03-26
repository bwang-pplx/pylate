#!/bin/bash
# Evaluate Group A experiments: A1, A2-zs, A2-nat, A4
# These are the key H1 vs H2 decomposition experiments.

set -e

QOS="search"
GPUS=1
CPUS=8
MEM="64G"
TIME="24:00:00"
COMMON_DIR="output/ood_study"
LOG_DIR="logs/ood_study_eval"

mkdir -p "$LOG_DIR"

# A1 (= M5 ColBERT MaxSim) — PLAID eval
sbatch --job-name="eval-a1" --qos=$QOS --gres=gpu:$GPUS --cpus-per-task=$CPUS --mem=$MEM --time=$TIME \
    --output="$LOG_DIR/eval-a1_%j.out" --error="$LOG_DIR/eval-a1_%j.err" \
    --wrap="export TORCH_COMPILE_DISABLE=1 && python scripts/eval_beir.py --model $COMMON_DIR/m5-colbert-seed1/final --document_length 512 --dataset all"
echo "Submitted: eval-a1"

# A2-zs (M5 checkpoint, MeanSim at inference) — brute-force
sbatch --job-name="eval-a2zs" --qos=$QOS --gres=gpu:$GPUS --cpus-per-task=$CPUS --mem=$MEM --time=$TIME \
    --output="$LOG_DIR/eval-a2zs_%j.out" --error="$LOG_DIR/eval-a2zs_%j.err" \
    --wrap="export TORCH_COMPILE_DISABLE=1 && python scripts/ood_study/eval_colbert_beir.py --model $COMMON_DIR/m5-colbert-seed1/final --aggregation mean --document_length 512 --dataset all"
echo "Submitted: eval-a2zs"

# A2-nat (MeanSim trained, MeanSim eval) — brute-force
sbatch --job-name="eval-a2nat" --qos=$QOS --gres=gpu:$GPUS --cpus-per-task=$CPUS --mem=$MEM --time=$TIME \
    --output="$LOG_DIR/eval-a2nat_%j.out" --error="$LOG_DIR/eval-a2nat_%j.err" \
    --wrap="export TORCH_COMPILE_DISABLE=1 && python scripts/ood_study/eval_colbert_beir.py --model $COMMON_DIR/a2nat-meansim-seed1/final --aggregation mean --document_length 512 --dataset all"
echo "Submitted: eval-a2nat"

# A4 (= M2 Dense mean-pool) — MTEB eval
sbatch --job-name="eval-a4" --qos=$QOS --gres=gpu:$GPUS --cpus-per-task=$CPUS --mem=$MEM --time=$TIME \
    --output="$LOG_DIR/eval-a4_%j.out" --error="$LOG_DIR/eval-a4_%j.err" \
    --wrap="export TORCH_COMPILE_DISABLE=1 && python scripts/ood_study/eval_dense_beir.py --model $COMMON_DIR/m2-dense-seed1/final --dataset all"
echo "Submitted: eval-a4"

echo ""
echo "4 eval jobs submitted. Check status: squeue -u \$USER"
