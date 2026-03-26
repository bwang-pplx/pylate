#!/bin/bash
QOS="search"
GPUS=4
CPUS=32
MEM="64G"
TIME="24:00:00"
LOG_DIR="logs/ood_study"
COMMON="--output_dir output/ood_study --wandb_project ood-study"

submit() {
    local JOB_NAME=$1
    shift
    sbatch \
        --job-name="$JOB_NAME" \
        --qos="$QOS" \
        --gres=gpu:$GPUS \
        --cpus-per-task=$CPUS \
        --mem=$MEM \
        --time=$TIME \
        --output="$LOG_DIR/${JOB_NAME}_%j.out" \
        --error="$LOG_DIR/${JOB_NAME}_%j.err" \
        --wrap="export TORCH_COMPILE_DISABLE=1 && accelerate launch --num_processes $GPUS $@"
    echo "Submitted: $JOB_NAME"
}

submit "m5-s1" "scripts/ood_study/train_m5_colbert.py --seed 1 $COMMON"
submit "m5-s3" "scripts/ood_study/train_m5_colbert.py --seed 3 $COMMON"
submit "d2-s1" "scripts/ood_study/train_d2_dim64.py --seed 1 $COMMON"
submit "d2-s2" "scripts/ood_study/train_d2_dim64.py --seed 2 $COMMON"
submit "d2-s3" "scripts/ood_study/train_d2_dim64.py --seed 3 $COMMON"
