#!/usr/bin/env bash
# Layer pruning experiments for pplx-embed-v1-late-0.6b (28 layers)
#
# All results append to the same JSON file for comparison.
# Re-running skips already-evaluated configs automatically.
#
# Usage:
#   bash scripts/run_pruning_experiments.sh          # run all
#   bash scripts/run_pruning_experiments.sh quick     # baseline + 2 strategies
#   bash scripts/run_pruning_experiments.sh baseline   # baseline only

set -euo pipefail

MODEL="perplexity-ai/pplx-embed-v1-late-0.6b"
OUTPUT="results/layer_pruning_results.json"
DATASETS="scifact nfcorpus fiqa2018 scidocs arguana"
BATCH_SIZE=32
SCRIPT="scripts/layer_pruning_experiment.py"

PYTHON=".venv/bin/python"
RUN="$PYTHON $SCRIPT --model $MODEL --output $OUTPUT --datasets $DATASETS --batch-size $BATCH_SIZE"

MODE="${1:-all}"

echo "============================================"
echo " Layer Pruning Experiments"
echo " Model: $MODEL"
echo " Output: $OUTPUT"
echo " Mode: $MODE"
echo "============================================"

# --- Phase 1: Baseline (full 28 layers) ---
if [[ "$MODE" == "all" || "$MODE" == "baseline" || "$MODE" == "quick" ]]; then
    echo ""
    echo ">>> Phase 1: Baseline (28 layers)"
    $RUN --strategies uniform --keep-layers 99
    # keep-layers 99 > 28 so no pruned configs are generated, only "full"
fi

# --- Phase 2: Coarse sweep — one strategy, big steps ---
if [[ "$MODE" == "all" || "$MODE" == "quick" ]]; then
    echo ""
    echo ">>> Phase 2: Coarse sweep (tail strategy, 24/20/14/7 layers)"
    $RUN --strategies tail --keep-layers 24 20 14 7 --skip-baseline
fi

# --- Phase 3: All strategies at key layer counts ---
if [[ "$MODE" == "all" ]]; then
    echo ""
    echo ">>> Phase 3: Uniform strategy"
    $RUN --strategies uniform --keep-layers 24 20 16 14 10 7 --skip-baseline

    echo ""
    echo ">>> Phase 4: Head-tail strategy"
    $RUN --strategies head_tail --keep-layers 24 20 16 14 10 --skip-baseline

    echo ""
    echo ">>> Phase 5: Middle-out strategy"
    $RUN --strategies middle_out --keep-layers 24 20 16 14 10 --skip-baseline
fi

# --- Print final summary ---
echo ""
echo "============================================"
echo " All experiments complete!"
echo " Results: $OUTPUT"
echo "============================================"

# Print summary from the results file
$PYTHON -c "
import json
from pathlib import Path

results = json.loads(Path('$OUTPUT').read_text())
baseline = results.get('full', {}).get('mean_ndcg@10')

print()
print(f\"{'Config':<25} {'Layers':>6} {'Params':>8} {'nDCG@10':>10} {'delta':>10} {'MRR@10':>10} {'Recall@10':>10}\")
print('-' * 85)
for name, r in sorted(results.items(), key=lambda x: -x[1]['num_layers']):
    delta = r['mean_ndcg@10'] - baseline if baseline else 0
    d = f'{delta:+.4f}' if name != 'full' else 'baseline'
    print(f\"{name:<25} {r['num_layers']:>6} {r['params_M']:>7.1f}M {r['mean_ndcg@10']:>10.4f} {d:>10} {r['mean_mrr@10']:>10.4f} {r['mean_recall@10']:>10.4f}\")
"
