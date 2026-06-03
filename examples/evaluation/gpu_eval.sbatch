#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=dev
#SBATCH --qos=search
#SBATCH --chdir=/traindata/bo/pylate
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00

# Reusable single-GPU eval runner. Pass the command via EVAL_CMD, e.g.:
#   sbatch --export=ALL,EVAL_CMD="uv run --extra dev python examples/evaluation/nanobeir_quick.py --sparse_dim 0" examples/evaluation/gpu_eval.sbatch
set -euo pipefail
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p logs

# Skip broken-driver nodes (requeue onto another one).
if ! uv run --extra dev python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "CUDA unavailable on $(hostname) -- requeuing (attempt ${SLURM_RESTART_COUNT:-0})."
    if [ "${SLURM_RESTART_COUNT:-0}" -lt 8 ]; then
        scontrol update jobid="${SLURM_JOB_ID}" ExcNodeList="$(hostname)" 2>/dev/null || true
        scontrol requeue "${SLURM_JOB_ID}"; sleep 5; exit 0
    fi
    echo "Too many bad-node requeues; aborting."; exit 1
fi

echo "eval on $(hostname)"
eval "${EVAL_CMD}"
