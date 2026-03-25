#!/bin/bash
# Launch all OOD study training runs.
# Edit NUM_GPUS and uncomment sections as needed.
#
# Usage:
#   bash scripts/ood_study/launch_all.sh

set -e

NUM_GPUS=1  # Set to your GPU count
LAUNCH="python"  # Change to "accelerate launch --num_processes $NUM_GPUS" for multi-GPU

OUTPUT_DIR="output/ood_study"
WANDB_PROJECT="ood-study"
COMMON="--output_dir $OUTPUT_DIR --wandb_project $WANDB_PROJECT"

# ===========================================================================
# Phase 1 (Week 1-2): Core spectrum
# ===========================================================================

# M5: ColBERT-Full (base model — all inference-only variants derive from this)
for SEED in 1 2 3; do
    $LAUNCH scripts/ood_study/train_m5_colbert.py --seed $SEED $COMMON
done

# M2: Dense mean-pool (sentence-transformers)
for SEED in 1 2 3; do
    $LAUNCH scripts/ood_study/train_m2_dense.py --seed $SEED $COMMON
done

# CE: Cross-Encoder (sentence-transformers, single seed — reference only)
$LAUNCH scripts/ood_study/train_ce.py --seed 1 $COMMON

# M4: Multi-K (K=4, 8, 16) — requires stride-pool implementation
# for K in 4 8 16; do
#     for SEED in 1 2 3; do
#         $LAUNCH scripts/ood_study/train_m4_multik.py --K $K --seed $SEED $COMMON
#     done
# done

# ===========================================================================
# Phase 2 (Week 3-4): Ablations
# ===========================================================================

# A2-nat: MeanSim native training
for SEED in 1 2 3; do
    $LAUNCH scripts/ood_study/train_a2nat_meansim.py --seed $SEED $COMMON
done

# D2: dim=64 projection (backbone frozen)
for SEED in 1 2 3; do
    $LAUNCH scripts/ood_study/train_d2_dim64.py --seed $SEED $COMMON
done

# HN-7 confound check
for SEED in 1 2 3; do
    $LAUNCH scripts/ood_study/train_m5_hn7.py --seed $SEED $COMMON
    $LAUNCH scripts/ood_study/train_m2_dense_hn7.py --seed $SEED $COMMON
done

echo "All training runs launched."
