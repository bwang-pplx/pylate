#!/usr/bin/env bash
# Layer pruning sweep for pplx-embed-v1-late-0.6b (28 layers)
#
# Creates a wandb sweep and launches 8 agents (one per GPU).
# Each agent picks configs from the sweep queue automatically.
#
# Usage:
#   bash scripts/run_pruning_experiments.sh                    # 8 GPUs
#   bash scripts/run_pruning_experiments.sh 4                  # 4 GPUs
#   WANDB_PROJECT=my-proj bash scripts/run_pruning_experiments.sh

set -euo pipefail

NUM_GPUS="${1:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-colbert-layer-pruning}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
PYTHON=".venv/bin/python"
SCRIPT="scripts/layer_pruning_experiment.py"

echo "============================================"
echo " Layer Pruning Sweep"
echo " Project: $WANDB_PROJECT"
echo " GPUs: $NUM_GPUS"
echo "============================================"

# Create sweep and capture the sweep ID
ENTITY_FLAG=""
if [[ -n "$WANDB_ENTITY" ]]; then
    ENTITY_FLAG="--wandb-entity $WANDB_ENTITY"
fi

CREATE_OUTPUT=$($PYTHON $SCRIPT create-sweep \
    --wandb-project "$WANDB_PROJECT" \
    $ENTITY_FLAG 2>&1)

echo "$CREATE_OUTPUT"

# Extract sweep ID from output (last word of the "Sweep created: <id>" line)
SWEEP_ID=$(echo "$CREATE_OUTPUT" | grep "Sweep created:" | awk '{print $NF}')

if [[ -z "$SWEEP_ID" ]]; then
    echo "ERROR: Failed to create sweep"
    exit 1
fi

# Build full sweep path
if [[ -n "$WANDB_ENTITY" ]]; then
    FULL_SWEEP_ID="$WANDB_ENTITY/$WANDB_PROJECT/$SWEEP_ID"
else
    FULL_SWEEP_ID="$WANDB_PROJECT/$SWEEP_ID"
fi

mkdir -p results

echo ""
echo "Launching $NUM_GPUS agents for sweep: $FULL_SWEEP_ID"
echo ""

# Launch agents — one per GPU
PIDS=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
    echo "Starting agent on GPU $i..."
    CUDA_VISIBLE_DEVICES=$i $PYTHON $SCRIPT agent "$FULL_SWEEP_ID" \
        > "results/agent_gpu${i}.log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All agents launched. PIDs: ${PIDS[*]}"
echo "Logs: results/agent_gpu{0..${NUM_GPUS}}.log"
echo ""
echo "Monitor at: https://wandb.ai/$WANDB_PROJECT/sweeps/$SWEEP_ID"
echo ""
echo "Waiting for all agents to complete..."

# Wait for all agents
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [[ $FAILED -eq 0 ]]; then
    echo "All agents completed successfully!"
else
    echo "WARNING: $FAILED agent(s) failed. Check logs."
fi
