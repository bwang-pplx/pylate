#!/bin/bash
# Launch ablation runs as separate Slurm jobs.
# Usage: bash scripts/launch_ablation.sh

QOS="${QOS:-search}"
GPUS="${GPUS:-8}"

mkdir -p logs

# Self-distillation experiments on ModernBERT-base
for LOSS in self-distill self-distill-topk; do
    JOB_NAME="ablation-${LOSS}-modernbert"

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

# Contrastive baseline on bidirectional-qwen3-0.6b-diffusion
JOB_NAME="ablation-contrastive-qwen3"
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
    --wrap="accelerate launch --num_processes ${GPUS} scripts/ablation_self_distillation.py --loss contrastive --model perplexity-ai/bidirectional-qwen3-0.6b-diffusion"

echo "Submitted: ${JOB_NAME}"
