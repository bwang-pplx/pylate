#!/bin/bash
#SBATCH --job-name=sparse_distill
#SBATCH --partition=dev
#SBATCH --qos=search
#SBATCH --chdir=/traindata/bo/pylate
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # one task; torchrun spawns one proc per GPU under it
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0                    # all memory on the node
#SBATCH --time=24:00:00

set -euo pipefail

# ---------------------------------------------------------------------------
# Run config (edit these; everything below is plumbing)
# ---------------------------------------------------------------------------
# All knobs are env-overridable, e.g.:
#   sbatch --export=ALL,LR=1e-4,K=64 examples/train/sparse_distillation.sbatch
MODEL_NAME="${MODEL_NAME:-lightonai/GTE-ModernColBERT-v1}"
DATASET_NAME="${DATASET_NAME:-sentence-transformers/msmarco-bm25}"
DATASET_CONFIG="${DATASET_CONFIG:-triplet}"   # text triplet config (query/positive/negative)
SPARSE_DIM="${SPARSE_DIM:-16384}"
K="${K:-32}"
FLOPS_LAMBDA="${FLOPS_LAMBDA:-0.0}"           # SPLADE FLOPS reg weight (0 = off)
FLOPS_WARMUP_STEPS="${FLOPS_WARMUP_STEPS:-0}"  # quadratic warmup of flops_lambda
NUM_EPOCHS="${NUM_EPOCHS:-1}"
# With accelerator split_batches=True (set in the training script) this is the
# GLOBAL batch, split across the 8 GPUs -> 16 per GPU here.
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-3e-4}"
REPORT_TO="${REPORT_TO:-wandb}"               # uses $WANDB_API_KEY from the environment
EVAL_STEPS="${EVAL_STEPS:-100}"               # NanoBEIR eval cadence; also eval at step 0 (baseline)
STOP_AT_STEP="${STOP_AT_STEP:--1}"            # set to e.g. 50 for a quick end-to-end smoke test
EXTRA_ARGS="${EXTRA_ARGS:-}"                  # e.g. "--no_clamp_dense_target"
OUTPUT_DIR="output/sparse_distill-${SLURM_JOB_ID}"

NUM_GPUS=$(echo "${SLURM_GPUS_ON_NODE:-8}")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-pylate-sparse-distill}"
# Model + msmarco + all NanoBEIR datasets are pre-cached, so run fully offline.
# This avoids HF Hub rate-limiting from 8 ranks fetching datasets at eval time.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# Keep BLAS from oversubscribing the cores each DDP rank shares.
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-96} / NUM_GPUS ))
# HF cache lives on the shared /traindata filesystem via $HOME.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

mkdir -p logs "${OUTPUT_DIR}"

# Preflight: some nodes in the draining pool have a broken CUDA driver (torch
# can't see the GPUs -> the bf16 check fails before training). If CUDA is not
# usable here, exclude this node and requeue onto another one (bounded retries).
if ! uv run --extra dev python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "CUDA unavailable on $(hostname) -- bad node. Requeuing (attempt ${SLURM_RESTART_COUNT:-0})."
    if [ "${SLURM_RESTART_COUNT:-0}" -lt 8 ]; then
        scontrol update jobid="${SLURM_JOB_ID}" ExcNodeList="$(hostname)" 2>/dev/null || true
        scontrol requeue "${SLURM_JOB_ID}"
        sleep 5
        exit 0
    fi
    echo "Too many bad-node requeues; aborting."
    exit 1
fi

echo "Job ${SLURM_JOB_ID} on $(hostname) | GPUs=${NUM_GPUS} | OMP_NUM_THREADS=${OMP_NUM_THREADS}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

# ---------------------------------------------------------------------------
# Launch: DDP across the node's GPUs via torchrun, inside the project's uv env.
# `python -m torch.distributed.run` is torchrun; using the module form avoids
# depending on a console-script being on PATH.
# ---------------------------------------------------------------------------
uv run --extra dev --with wandb python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    examples/train/sparse_distillation.py \
    --model_name "${MODEL_NAME}" \
    --dataset_name "${DATASET_NAME}" \
    --dataset_config "${DATASET_CONFIG}" \
    --sparse_dim "${SPARSE_DIM}" \
    --k "${K}" \
    --flops_lambda "${FLOPS_LAMBDA}" \
    --flops_warmup_steps "${FLOPS_WARMUP_STEPS}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LR}" \
    --report_to "${REPORT_TO}" \
    --eval_steps "${EVAL_STEPS}" \
    --stop_at_step "${STOP_AT_STEP}" \
    --output_dir "${OUTPUT_DIR}" \
    ${EXTRA_ARGS}

echo "Done. Model saved under ${OUTPUT_DIR}/final"
