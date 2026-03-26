#!/bin/bash
# Submit all OOD study training runs to SLURM.
# Each run uses 1 full node (8 GPUs).
#
# Usage:
#   bash scripts/ood_study/slurm_submit.sh

set -e

QOS="search"
GPUS=8                   # GPUs per job (1 full node)
CPUS=64                  # CPUs per job (for dataloaders)
MEM="64G"                # Memory per job
TIME="24:00:00"          # Max wall time per job
OUTPUT_DIR="output/ood_study"
WANDB_PROJECT="ood-study"
LOG_DIR="logs/ood_study"

mkdir -p "$LOG_DIR"

submit() {
    local JOB_NAME=$1
    shift
    local CMD="export TORCH_COMPILE_DISABLE=1 && accelerate launch --num_processes $GPUS $@"

    sbatch \
        --job-name="$JOB_NAME" \
        --qos="$QOS" \
        --gres=gpu:$GPUS \
        --cpus-per-task=$CPUS \
        --mem=$MEM \
        --time=$TIME \
        --chdir="$PWD" \
        --output="$LOG_DIR/${JOB_NAME}_%j.out" \
        --error="$LOG_DIR/${JOB_NAME}_%j.err" \
        --wrap="$CMD"

    echo "Submitted: $JOB_NAME"
}

COMMON="--output_dir $OUTPUT_DIR --wandb_project $WANDB_PROJECT"

# 7 jobs, 1 seed each
submit "m5"      "scripts/ood_study/train_m5_colbert.py --seed 1 $COMMON"
submit "m2"      "scripts/ood_study/train_m2_dense.py --seed 1 $COMMON"
submit "ce"      "scripts/ood_study/train_ce.py --seed 1 $COMMON"
submit "a2nat"   "scripts/ood_study/train_a2nat_meansim.py --seed 1 $COMMON"
submit "d2"      "scripts/ood_study/train_d2_dim64.py --seed 1 $COMMON"
submit "m5-hn7"  "scripts/ood_study/train_m5_hn7.py --seed 1 $COMMON"
submit "m2-hn7"  "scripts/ood_study/train_m2_dense_hn7.py --seed 1 $COMMON"

echo ""
echo "7 jobs submitted. Check status with: squeue -u \$USER"
echo "Logs in: $LOG_DIR/"
