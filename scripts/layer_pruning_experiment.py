"""Layer pruning experiment for ColBERT models via wandb sweep.

Each pruning config runs as an independent wandb sweep agent, enabling
parallel execution across multiple GPUs.

Usage:
    # Create sweep and launch 8 agents (one per GPU):
    bash scripts/run_pruning_experiments.sh

    # Or manually:
    # 1. Create the sweep
    python scripts/layer_pruning_experiment.py create-sweep --wandb-project colbert-layer-pruning

    # 2. Launch agents (one per GPU)
    CUDA_VISIBLE_DEVICES=0 wandb agent <sweep_id>
    CUDA_VISIBLE_DEVICES=1 wandb agent <sweep_id>
    ...

    # Run a single config without wandb:
    python scripts/layer_pruning_experiment.py run \
        --model perplexity-ai/pplx-embed-v1-late-0.6b \
        --strategy tail --num-layers 14 --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import wandb
from torch import nn

from pylate import evaluation, models

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

TOTAL_LAYERS = 28  # pplx-embed-v1-late-0.6b


def get_transformer_layers(model: models.ColBERT) -> nn.ModuleList:
    """Get the transformer layer list from the model backbone."""
    auto_model = model[0].auto_model
    for attr in ["layers", "encoder.layer", "layer"]:
        obj = auto_model
        for part in attr.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and isinstance(obj, nn.ModuleList):
            return obj
    for name, module in auto_model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 4:
            return module
    raise RuntimeError(
        f"Could not find transformer layers. "
        f"Top-level attributes: {[n for n, _ in auto_model.named_children()]}"
    )


def set_transformer_layers(model: models.ColBERT, layers: nn.ModuleList) -> None:
    """Set the transformer layer list and update config."""
    auto_model = model[0].auto_model
    for attr in ["layers", "encoder.layer", "layer"]:
        parts = attr.split(".")
        obj = auto_model
        for part in parts[:-1]:
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, parts[-1]):
            existing = getattr(obj, parts[-1])
            if isinstance(existing, nn.ModuleList) and len(existing) > 4:
                setattr(obj, parts[-1], layers)
                auto_model.config.num_hidden_layers = len(layers)
                return
    for name, module in auto_model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 4:
            parent_name = ".".join(name.split(".")[:-1]) if "." in name else ""
            child_name = name.split(".")[-1]
            parent = (
                auto_model
                if not parent_name
                else dict(auto_model.named_modules())[parent_name]
            )
            setattr(parent, child_name, layers)
            auto_model.config.num_hidden_layers = len(layers)
            return
    raise RuntimeError("Could not set transformer layers")


# ---------------------------------------------------------------------------
# Pruning strategies
# ---------------------------------------------------------------------------


def compute_keep_indices(strategy: str, num_keep: int, total: int) -> list[int]:
    """Compute which layer indices to keep for a given strategy."""
    if num_keep >= total:
        return list(range(total))

    if strategy == "full":
        return list(range(total))

    if strategy == "uniform":
        step = total / num_keep
        indices = [int(i * step) for i in range(num_keep)]
        if 0 not in indices:
            indices[0] = 0
        if total - 1 not in indices:
            indices[-1] = total - 1
        return sorted(set(indices))

    if strategy == "tail":
        return list(range(total - num_keep, total))

    if strategy == "head_tail":
        head = num_keep // 4
        tail = num_keep - head
        return sorted(set(list(range(head)) + list(range(total - tail, total))))

    if strategy == "middle_out":
        half = num_keep // 2
        return sorted(set(list(range(half)) + list(range(total - half, total))))

    raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_pruned_model(
    model_name: str,
    strategy: str,
    num_layers: int,
    device: str | None = None,
    batch_size: int = 32,
    datasets: list[str] | None = None,
) -> dict:
    """Load model, prune layers, evaluate on NanoBEIR, return metrics."""
    print(f"Loading model: {model_name}")
    model = models.ColBERT(
        model_name_or_path=model_name,
        trust_remote_code=True,
        device=device,
    )

    total = len(get_transformer_layers(model))
    keep_indices = compute_keep_indices(strategy, num_layers, total)

    config_name = f"{strategy}_{num_layers}L" if strategy != "full" else "full"
    print(f"Config: {config_name} — keeping {len(keep_indices)}/{total} layers")
    print(f"Layers: {keep_indices}")

    if strategy != "full":
        original_layers = get_transformer_layers(model)
        new_layers = nn.ModuleList([original_layers[i] for i in keep_indices])
        set_transformer_layers(model, new_layers)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    eval_kwargs = {"batch_size": batch_size}
    if datasets is not None:
        eval_kwargs["dataset_names"] = datasets
    evaluator = evaluation.NanoBEIREvaluator(**eval_kwargs)

    start_time = time.time()
    metrics = evaluator(model)
    elapsed = time.time() - start_time

    ndcg10 = metrics.get("NanoBEIR_mean_MaxSim_ndcg@10", 0)
    mrr10 = metrics.get("NanoBEIR_mean_MaxSim_mrr@10", 0)
    recall10 = metrics.get("NanoBEIR_mean_MaxSim_recall@10", 0)

    print(f"nDCG@10: {ndcg10:.4f} | MRR@10: {mrr10:.4f} | Recall@10: {recall10:.4f}")
    print(f"Params: {n_params:.1f}M | Time: {elapsed:.1f}s")

    return {
        "config_name": config_name,
        "strategy": strategy,
        "num_layers": len(keep_indices),
        "total_layers": total,
        "layers_kept": keep_indices,
        "params_M": round(n_params, 1),
        "time_s": round(elapsed, 1),
        "mean_ndcg@10": round(ndcg10, 4),
        "mean_mrr@10": round(mrr10, 4),
        "mean_recall@10": round(recall10, 4),
        "all_metrics": {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in metrics.items()
        },
    }


# ---------------------------------------------------------------------------
# Wandb sweep
# ---------------------------------------------------------------------------

STRATEGIES = ["full", "uniform", "tail", "head_tail", "middle_out"]
LAYER_COUNTS = [28, 24, 20, 16, 14, 10, 7]


def build_valid_configs() -> list[str]:
    """Build list of valid 'strategy:num_layers' config strings."""
    configs = []
    for s in STRATEGIES:
        for n in LAYER_COUNTS:
            if s == "full" and n == TOTAL_LAYERS:
                configs.append(f"{s}:{n}")
            elif s != "full" and n < TOTAL_LAYERS:
                configs.append(f"{s}:{n}")
    return configs


def build_sweep_config(model_name: str) -> dict:
    """Build wandb sweep config with grid search over all valid configs."""
    configs = build_valid_configs()
    return {
        "name": f"layer-pruning-{model_name.split('/')[-1]}",
        "method": "grid",
        "parameters": {
            # Single parameter encodes both strategy and num_layers
            "config": {"values": configs},
        },
    }


def sweep_agent_fn():
    """Function called by each wandb sweep agent."""
    run = wandb.init()

    # Parse "strategy:num_layers" config string
    config_str = wandb.config.config
    strategy, num_layers_str = config_str.split(":")
    num_layers = int(num_layers_str)

    # Set a readable run name
    config_name = f"{strategy}_{num_layers}L" if strategy != "full" else "full"
    run.name = config_name

    model_name = run.config.get("model", "perplexity-ai/pplx-embed-v1-late-0.6b")
    batch_size = run.config.get("batch_size", 32)

    result = evaluate_pruned_model(
        model_name=model_name,
        strategy=strategy,
        num_layers=num_layers,
        batch_size=batch_size,
    )

    # Log all metrics
    log_data = {
        "num_layers": result["num_layers"],
        "params_M": result["params_M"],
        "time_s": result["time_s"],
        "mean_ndcg@10": result["mean_ndcg@10"],
        "mean_mrr@10": result["mean_mrr@10"],
        "mean_recall@10": result["mean_recall@10"],
    }
    for k, v in result["all_metrics"].items():
        if isinstance(v, (int, float)):
            log_data[k] = v
    wandb.log(log_data)

    # Also save to local JSON
    output_path = "results/layer_pruning_results.json"
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if output_file.exists():
        with open(output_file) as f:
            existing = json.load(f)
    existing[result["config_name"]] = result
    with open(output_file, "w") as f:
        json.dump(existing, f, indent=2)

    wandb.finish()


# ---------------------------------------------------------------------------
# Standalone run (no wandb)
# ---------------------------------------------------------------------------


def run_standalone(args):
    """Run a single config without wandb."""
    result = evaluate_pruned_model(
        model_name=args.model,
        strategy=args.strategy,
        num_layers=args.num_layers,
        device=args.device,
        batch_size=args.batch_size,
        datasets=args.datasets,
    )

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if output_file.exists():
        with open(output_file) as f:
            existing = json.load(f)
    existing[result["config_name"]] = result
    with open(output_file, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Results saved to {args.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="ColBERT layer pruning experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- create-sweep ---
    sp_create = subparsers.add_parser("create-sweep", help="Create a wandb sweep")
    sp_create.add_argument(
        "--wandb-project",
        default="colbert-layer-pruning",
        help="Wandb project name",
    )
    sp_create.add_argument(
        "--wandb-entity",
        default=None,
        help="Wandb entity (team/user)",
    )
    sp_create.add_argument(
        "--model",
        default="perplexity-ai/pplx-embed-v1-late-0.6b",
    )

    # --- agent ---
    sp_agent = subparsers.add_parser("agent", help="Run as a wandb sweep agent")
    sp_agent.add_argument("sweep_id", help="Wandb sweep ID (entity/project/sweep_id)")
    sp_agent.add_argument("--count", type=int, default=None, help="Max runs for this agent")

    # --- run (standalone, no wandb) ---
    sp_run = subparsers.add_parser("run", help="Run a single config without wandb")
    sp_run.add_argument("--model", default="perplexity-ai/pplx-embed-v1-late-0.6b")
    sp_run.add_argument("--strategy", required=True, choices=STRATEGIES)
    sp_run.add_argument("--num-layers", type=int, required=True)
    sp_run.add_argument("--device", default=None)
    sp_run.add_argument("--batch-size", type=int, default=32)
    sp_run.add_argument("--datasets", nargs="+", default=None)
    sp_run.add_argument("--output", default="results/layer_pruning_results.json")

    args = parser.parse_args()

    if args.command == "create-sweep":
        sweep_config = build_sweep_config(args.model)
        sweep_config["program"] = "scripts/layer_pruning_experiment.py"
        # Pass model as a fixed parameter so agents can read it
        sweep_config["parameters"]["model"] = {
            "value": args.model,
        }
        sweep_config["parameters"]["batch_size"] = {"value": 32}
        sweep_id = wandb.sweep(
            sweep_config,
            project=args.wandb_project,
            entity=args.wandb_entity,
        )
        # Resolve entity (may come from default login if not specified)
        entity = args.wandb_entity or wandb.Api().default_entity
        full_id = f"{entity}/{args.wandb_project}/{sweep_id}"
        print(f"\nSweep created: {sweep_id}")
        print(f"SWEEP_PATH={full_id}")
        print(f"URL: https://wandb.ai/{full_id}")
        print(f"\nTo launch agents:")
        for i in range(8):
            print(f"  CUDA_VISIBLE_DEVICES={i} python scripts/layer_pruning_experiment.py agent {full_id} &")

    elif args.command == "agent":
        wandb.agent(args.sweep_id, function=sweep_agent_fn, count=args.count)

    elif args.command == "run":
        run_standalone(args)


if __name__ == "__main__":
    main()
