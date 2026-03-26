#!/bin/bash
# Submit all OOD study training runs to SLURM.
# Each run uses 1 GPU. All jobs are independent and run in parallel.
#
# Edit the SLURM parameters below to match your cluster.
#
# Usage:
#   bash scripts/ood_study/slurm_submit.sh

QOS="search"             # SLURM QoS
GPUS=4                   # GPUs per job
CPUS=32                  # CPUs per job (for dataloaders)
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
        --output="$LOG_DIR/${JOB_NAME}_%j.out" \
        --error="$LOG_DIR/${JOB_NAME}_%j.err" \
        --chdir="$PWD" \
        --wrap="$CMD"

    echo "Submitted: $JOB_NAME"
}

COMMON="--output_dir $OUTPUT_DIR --wandb_project $WANDB_PROJECT"

# ===========================================================================
# Phase 1: Core spectrum
# ===========================================================================

# M5: ColBERT-Full
for SEED in 1 2 3; do
    submit "m5-s${SEED}" "scripts/ood_study/train_m5_colbert.py --seed $SEED $COMMON"
done

# M2: Dense mean-pool
for SEED in 1 2 3; do
    submit "m2-s${SEED}" "scripts/ood_study/train_m2_dense.py --seed $SEED $COMMON"
done

# CE: Cross-Encoder (single seed)
submit "ce-s1" "scripts/ood_study/train_ce.py --seed 1 $COMMON"

# M4: Multi-K (uncomment when stride-pool is implemented)
# for K in 4 8 16; do
#     for SEED in 1 2 3; do
#         submit "m4-K${K}-s${SEED}" "scripts/ood_study/train_m4_multik.py --K $K --seed $SEED $COMMON"
#     done
# done

# ===========================================================================
# Phase 2: Ablations
# ===========================================================================

# A2-nat: MeanSim
for SEED in 1 2 3; do
    submit "a2nat-s${SEED}" "scripts/ood_study/train_a2nat_meansim.py --seed $SEED $COMMON"
done

# D2: dim=64
for SEED in 1 2 3; do
    submit "d2-s${SEED}" "scripts/ood_study/train_d2_dim64.py --seed $SEED $COMMON"
done

# HN-7 confound check
for SEED in 1 2 3; do
    submit "m5-hn7-s${SEED}" "scripts/ood_study/train_m5_hn7.py --seed $SEED $COMMON"
    submit "m2-hn7-s${SEED}" "scripts/ood_study/train_m2_dense_hn7.py --seed $SEED $COMMON"
done

echo ""
echo "All jobs submitted. Check status with: squeue -u \$USER"
echo "Logs in: $LOG_DIR/"
