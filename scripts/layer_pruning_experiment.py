"""Layer pruning experiment for ColBERT models.

Evaluates how many transformer layers can be removed from a ColBERT model
while maintaining retrieval quality on NanoBEIR tasks.

Usage:
    # Run a single strategy
    uv run python scripts/layer_pruning_experiment.py --strategies uniform

    # Run specific layer counts with a specific strategy
    uv run python scripts/layer_pruning_experiment.py --strategies tail --keep-layers 24 20 14

    # Run explicit layer indices
    uv run python scripts/layer_pruning_experiment.py --config-name custom_12L --layer-indices 0 1 2 3 24 25 26 27 --skip-baseline

    # Results append to the same JSON file across runs
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from pylate import evaluation, models


def get_transformer_layers(model: models.ColBERT) -> nn.ModuleList:
    """Get the transformer layer list from the model backbone."""
    auto_model = model[0].auto_model
    # Try common attribute names for the layer list
    for attr in ["layers", "encoder.layer", "layer"]:
        obj = auto_model
        for part in attr.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and isinstance(obj, nn.ModuleList):
            return obj
    # Fallback: search named modules for a ModuleList with many children
    for name, module in auto_model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 4:
            return module
    raise RuntimeError(
        f"Could not find transformer layers. "
        f"Top-level attributes: {[n for n, _ in auto_model.named_children()]}"
    )


def set_transformer_layers(
    model: models.ColBERT, layers: nn.ModuleList
) -> None:
    """Set the transformer layer list and update config."""
    auto_model = model[0].auto_model
    # Find and replace
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
    # Fallback
    for name, module in auto_model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 4:
            parent_name = ".".join(name.split(".")[:-1]) if "." in name else ""
            child_name = name.split(".")[-1]
            parent = auto_model if not parent_name else dict(auto_model.named_modules())[parent_name]
            setattr(parent, child_name, layers)
            auto_model.config.num_hidden_layers = len(layers)
            return
    raise RuntimeError("Could not set transformer layers")


def prune_layers(
    model: models.ColBERT, keep_indices: list[int]
) -> models.ColBERT:
    """Create a pruned copy of the model keeping only specified layer indices."""
    original_layers = get_transformer_layers(model)
    new_layers = nn.ModuleList([original_layers[i] for i in keep_indices])
    set_transformer_layers(model, new_layers)
    return model


def restore_layers(
    model: models.ColBERT, original_layers: nn.ModuleList
) -> None:
    """Restore the original layers after pruning."""
    set_transformer_layers(model, original_layers)


def load_existing_results(output_path: str) -> dict:
    """Load existing results file if it exists."""
    path = Path(output_path)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_results(output_path: str, results: dict) -> None:
    """Save results, merging with any existing file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


def print_summary(all_results: dict) -> None:
    """Print a comparison table of all results."""
    print(f"\n{'='*90}")
    print("SUMMARY (all experiments)")
    print(f"{'='*90}")
    baseline_ndcg = all_results.get("full", {}).get("mean_ndcg@10")
    print(
        f"{'Config':<25} {'Layers':>6} {'Params':>8} "
        f"{'nDCG@10':>10} {'Δ nDCG':>10} {'MRR@10':>10} {'Recall@10':>10} {'Time':>8}"
    )
    print("-" * 90)
    # Sort by num_layers descending
    sorted_results = sorted(all_results.items(), key=lambda x: -x[1]["num_layers"])
    for name, r in sorted_results:
        if baseline_ndcg is not None:
            delta = r["mean_ndcg@10"] - baseline_ndcg
            delta_str = f"{delta:+.4f}" if name != "full" else "baseline"
        else:
            delta_str = "n/a"
        print(
            f"{name:<25} {r['num_layers']:>6} {r['params_M']:>7.1f}M "
            f"{r['mean_ndcg@10']:>10.4f} {delta_str:>10} "
            f"{r['mean_mrr@10']:>10.4f} {r['mean_recall@10']:>10.4f} {r['time_s']:>7.1f}s"
        )


def generate_pruning_configs(
    num_layers: int, strategies: list[str], keep_layers: list[int] | None = None
) -> dict[str, list[int]]:
    """Generate layer index configs for different pruning strategies."""
    if keep_layers is None:
        keep_layers = [24, 20, 16, 14, 10, 7]

    configs = {"full": list(range(num_layers))}

    if "uniform" in strategies:
        # Remove layers uniformly spaced, at various reduction levels
        for keep in keep_layers:
            if keep >= num_layers:
                continue
            step = num_layers / keep
            indices = [int(i * step) for i in range(keep)]
            # Always keep first and last layer
            if 0 not in indices:
                indices[0] = 0
            if num_layers - 1 not in indices:
                indices[-1] = num_layers - 1
            configs[f"uniform_{keep}L"] = sorted(set(indices))

    if "tail" in strategies:
        # Keep last N layers (top layers are most semantic)
        for keep in keep_layers:
            if keep >= num_layers:
                continue
            configs[f"tail_{keep}L"] = list(range(num_layers - keep, num_layers))

    if "head_tail" in strategies:
        # Keep first few + last few, drop middle
        for keep in keep_layers:
            if keep >= num_layers:
                continue
            head = keep // 4  # 25% from start
            tail = keep - head  # 75% from end
            indices = list(range(head)) + list(range(num_layers - tail, num_layers))
            configs[f"head_tail_{keep}L"] = sorted(set(indices))

    if "middle_out" in strategies:
        # Remove from middle, keep edges
        for keep in keep_layers:
            if keep >= num_layers:
                continue
            half = keep // 2
            indices = list(range(half)) + list(range(num_layers - half, num_layers))
            configs[f"middle_out_{keep}L"] = sorted(set(indices))

    return configs


def evaluate_config(
    model: models.ColBERT,
    dataset_names: list[str],
    batch_size: int,
) -> dict:
    """Run NanoBEIR evaluation and return metrics."""
    evaluator = evaluation.NanoBEIREvaluator(
        dataset_names=dataset_names,
        batch_size=batch_size,
    )
    results = evaluator(model)
    return results


def run_experiment(
    model_name: str,
    datasets: list[str],
    strategies: list[str],
    keep_layers: list[int] | None,
    batch_size: int,
    output_path: str,
    device: str | None,
    skip_baseline: bool = False,
    config_name: str | None = None,
    layer_indices: list[int] | None = None,
):
    # Load existing results for appending
    all_results = load_existing_results(output_path)
    if all_results:
        print(f"Loaded {len(all_results)} existing results from {output_path}")

    print(f"Loading model: {model_name}")
    model = models.ColBERT(
        model_name_or_path=model_name,
        trust_remote_code=True,
        device=device,
        document_length=512,
    )

    original_layers = get_transformer_layers(model)
    num_layers = len(original_layers)
    print(f"Model has {num_layers} transformer layers")

    # Keep a reference to original layer list
    original_layer_list = list(original_layers)

    # Build configs to run
    if layer_indices is not None:
        # Explicit layer indices mode
        name = config_name or f"custom_{len(layer_indices)}L"
        configs = {name: sorted(layer_indices)}
    else:
        configs = generate_pruning_configs(num_layers, strategies, keep_layers)

    if skip_baseline:
        configs.pop("full", None)

    # Skip configs already evaluated
    new_configs = {k: v for k, v in configs.items() if k not in all_results}
    skipped = len(configs) - len(new_configs)
    if skipped:
        print(f"Skipping {skipped} already-evaluated configs")
    configs = new_configs

    if not configs:
        print("Nothing new to evaluate.")
        print_summary(all_results)
        return

    print(f"\nWill evaluate {len(configs)} configurations:")
    for name, indices in configs.items():
        print(f"  {name}: {len(indices)} layers")

    for config_name_iter, keep_indices in configs.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {config_name_iter} ({len(keep_indices)}/{num_layers} layers)")
        print(f"Layers kept: {keep_indices}")
        print(f"{'='*60}")

        # Restore original layers then prune
        full_layers = nn.ModuleList(original_layer_list)
        set_transformer_layers(model, full_layers)
        if config_name_iter != "full":
            prune_layers(model, keep_indices)

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters()) / 1e6

        start_time = time.time()
        metrics = evaluate_config(model, datasets, batch_size)
        elapsed = time.time() - start_time

        # Extract key metrics
        ndcg10 = metrics.get("NanoBEIR_mean_MaxSim_ndcg@10", 0)
        mrr10 = metrics.get("NanoBEIR_mean_MaxSim_mrr@10", 0)
        recall10 = metrics.get("NanoBEIR_mean_MaxSim_recall@10", 0)

        result = {
            "config": config_name_iter,
            "num_layers": len(keep_indices),
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
        all_results[config_name_iter] = result

        # Save after each config (incremental, crash-safe)
        save_results(output_path, all_results)

        print(f"  nDCG@10: {ndcg10:.4f} | MRR@10: {mrr10:.4f} | Recall@10: {recall10:.4f}")
        print(f"  Params: {n_params:.1f}M | Time: {elapsed:.1f}s")

    print(f"\nResults saved to {output_path}")
    print_summary(all_results)


def main():
    parser = argparse.ArgumentParser(description="ColBERT layer pruning experiment")
    parser.add_argument(
        "--model",
        default="perplexity-ai/pplx-embed-v1-late-0.6b",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["scifact", "nfcorpus", "fiqa2018", "scidocs", "arguana"],
        help="NanoBEIR dataset names to evaluate on",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["uniform", "tail", "head_tail", "middle_out"],
        choices=["uniform", "tail", "head_tail", "middle_out"],
        help="Pruning strategies to try",
    )
    parser.add_argument(
        "--keep-layers",
        nargs="+",
        type=int,
        default=None,
        help="Layer counts to try (default: 24 20 16 14 10 7)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output",
        default="results/layer_pruning_results.json",
        help="Output JSON path (results are appended across runs)",
    )
    parser.add_argument("--device", default=None, help="Device (cuda, mps, cpu)")
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip full model baseline evaluation",
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help="Name for a custom config (use with --layer-indices)",
    )
    parser.add_argument(
        "--layer-indices",
        nargs="+",
        type=int,
        default=None,
        help="Explicit layer indices to keep (overrides --strategies)",
    )
    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        datasets=args.datasets,
        strategies=args.strategies,
        keep_layers=args.keep_layers,
        batch_size=args.batch_size,
        output_path=args.output,
        device=args.device,
        skip_baseline=args.skip_baseline,
        config_name=args.config_name,
        layer_indices=args.layer_indices,
    )


if __name__ == "__main__":
    main()
