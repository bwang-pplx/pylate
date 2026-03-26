#!/bin/bash
# Submit all OOD study evaluation jobs to SLURM.
# Each eval job uses 1 GPU.
#
# Usage:
#   bash scripts/ood_study/slurm_eval.sh

set -e

QOS="search"
GPUS=1
CPUS=8
MEM="64G"
TIME="24:00:00"
OUTPUT_DIR="output/ood_study"
LOG_DIR="logs/ood_study_eval"

mkdir -p "$LOG_DIR"

submit() {
    local JOB_NAME=$1
    shift
    local CMD="export TORCH_COMPILE_DISABLE=1 && $@"

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

# ===========================================================================
# Standard ColBERT models — use PLAID index (fast)
# M5 (= A1), A2-nat, M5-HN7, D2
# ===========================================================================
for MODEL in m5-colbert a2nat-meansim m5-colbert-hn7 d2-dim64; do
    for SEED in 1 2 3; do
        DIR="$OUTPUT_DIR/${MODEL}-seed${SEED}/final"
        submit "eval-${MODEL}-s${SEED}" \
            "python scripts/eval_beir.py --model $DIR --dataset all"
    done
done

# ===========================================================================
# Dense models — use MTEB
# M2, M2-HN7
# ===========================================================================
for MODEL in m2-dense m2-dense-hn7; do
    for SEED in 1 2 3; do
        DIR="$OUTPUT_DIR/${MODEL}-seed${SEED}/final"
        submit "eval-${MODEL}-s${SEED}" \
            "python scripts/ood_study/eval_dense_beir.py --model $DIR --dataset all"
    done
done

# ===========================================================================
# Cross-encoder — rerank BM25 top-1000
# ===========================================================================
submit "eval-ce-s1" \
    "python scripts/ood_study/eval_ce_beir.py --model $OUTPUT_DIR/ce-modernbert-seed1/final --dataset all"

# ===========================================================================
# Ablation variants — brute-force scoring (from M5 checkpoints)
# ===========================================================================
for SEED in 1 2 3; do
    DIR="$OUTPUT_DIR/m5-colbert-seed${SEED}/final"

    # A2-zs: MeanSim zero-shot
    submit "eval-a2zs-s${SEED}" \
        "python scripts/ood_study/eval_colbert_beir.py --model $DIR --aggregation mean --dataset all"

    # C2: 30% token dropout
    submit "eval-c2-s${SEED}" \
        "python scripts/ood_study/eval_colbert_beir.py --model $DIR --token_dropout 0.3 --dataset all"

    # M6: keep 50% tokens by IDF
    submit "eval-m6-s${SEED}" \
        "python scripts/ood_study/eval_colbert_beir.py --model $DIR --idf_prune 0.5 --dataset all"

    # M7: keep 25% tokens by IDF
    submit "eval-m7-s${SEED}" \
        "python scripts/ood_study/eval_colbert_beir.py --model $DIR --idf_prune 0.25 --dataset all"
done

echo ""
echo "All eval jobs submitted. Check status with: squeue -u \$USER"
echo "Logs in: $LOG_DIR/"
echo ""
echo "After all jobs finish, run:"
echo "  python scripts/ood_study/collect_results.py"
