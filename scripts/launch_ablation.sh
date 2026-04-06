#!/bin/bash
# Launch all three ablation runs as separate Slurm jobs.
# Usage: bash scripts/launch_ablation.sh

QOS="${QOS:-search}"
GPUS="${GPUS:-8}"

mkdir -p logs

for LOSS in self-distill self-distill-topk; do
    JOB_NAME="ablation-${LOSS}"

    sbatch \
        --job-name="${JOB_NAME}" \
        --qos="${QOS}" \
        --nodes=1 \
        --gres=gpu:${GPUS} \
        --cpus-per-task=$((12 * GPUS)) \
        --mem=0 \
        --time=4:00:00 \
        --output=logs/${JOB_NAME}-%j.out \
        --error=logs/${JOB_NAME}-%j.err \
        --wrap="accelerate launch --num_processes ${GPUS} scripts/ablation_self_distillation.py --loss ${LOSS}"

    echo "Submitted: ${JOB_NAME}"
done
