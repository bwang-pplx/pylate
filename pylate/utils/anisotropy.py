from __future__ import annotations

import torch
from transformers import TrainerCallback

__all__ = ["token_anisotropy", "AnisotropyCallback"]


def token_anisotropy(token_embeddings: torch.Tensor) -> dict[str, float]:
    """Anisotropy statistics for a set of (token) embeddings.

    Anisotropy measures how much the embeddings collapse onto a shared
    direction (a narrow cone) rather than spreading over the space. For
    multi-vector / late-interaction models it is measured at the **token**
    level, since that is what MaxSim (and sparsification) operate on: pool all
    token vectors, L2-normalize each, and summarize the direction cloud.

    Parameters
    ----------
    token_embeddings
        A ``(num_tokens, dim)`` tensor of token embeddings (already pooled
        across the batch; padding / special tokens should be excluded by the
        caller).

    Returns
    -------
    dict with:
        - ``mean_norm``: norm of the mean of the L2-normalized embeddings, in
          ``[0, 1]``. ``0`` = isotropic, ``1`` = all aligned (anisotropic).
        - ``eff_rank``: participation ratio of the covariance spectrum (higher =
          more dimensions effectively used = more isotropic).
        - ``mean_pair_cos``: mean pairwise cosine over a sample of tokens.

    Examples
    --------
    >>> import torch
    >>> _ = torch.manual_seed(0)
    >>> isotropic = torch.randn(300, 32)
    >>> anisotropic = torch.randn(300, 32) + 4.0  # shared offset -> narrow cone
    >>> iso = token_anisotropy(isotropic)["mean_norm"]
    >>> ani = token_anisotropy(anisotropic)["mean_norm"]
    >>> bool(iso < ani)
    True
    """
    unit = torch.nn.functional.normalize(token_embeddings.float(), p=2, dim=-1)
    mean_norm = float(unit.mean(dim=0).norm())

    centered = unit - unit.mean(dim=0)
    variance = torch.linalg.svdvals(centered) ** 2
    denominator = float((variance**2).sum())
    eff_rank = float(variance.sum() ** 2 / denominator) if denominator > 0 else 0.0

    count = min(2000, unit.shape[0])
    sample = unit[torch.randperm(unit.shape[0])[:count]]
    mean_pair_cos = float((sample @ sample.t()).mean())

    return {
        "mean_norm": mean_norm,
        "eff_rank": eff_rank,
        "mean_pair_cos": mean_pair_cos,
    }


class AnisotropyCallback(TrainerCallback):
    """Trainer callback that logs token-level anisotropy on a fixed probe set.

    Every ``every_n_steps`` it encodes ``probe_texts`` with the model and logs
    :func:`token_anisotropy` of the pooled token embeddings (to Weights & Biases
    if a run is active, and to stdout). Encodes documents (``is_query=False``) by
    default to avoid the near-constant query expansion ``[MASK]`` tokens
    contaminating the estimate.

    Parameters
    ----------
    model
        The :class:`~pylate.models.ColBERT` being trained (used for ``encode``).
    probe_texts
        A fixed list of texts to measure anisotropy on (e.g. a sample of the
        training corpus). Kept fixed so the metric is comparable across steps.
    every_n_steps
        Logging cadence in optimizer steps. Defaults to ``100``.
    batch_size
        Encoding batch size. Defaults to ``32``.
    is_query
        Whether to encode as queries. Defaults to ``False`` (documents).
    prefix
        Metric-name prefix. Defaults to ``"anisotropy"``.
    """

    def __init__(
        self,
        model,
        probe_texts: list[str],
        every_n_steps: int = 100,
        batch_size: int = 32,
        is_query: bool = False,
        prefix: str = "anisotropy",
    ) -> None:
        self.model = model
        self.probe_texts = probe_texts
        self.every_n_steps = every_n_steps
        self.batch_size = batch_size
        self.is_query = is_query
        self.prefix = prefix

    @torch.no_grad()
    def measure(self) -> dict[str, float]:
        """Encode the probe set and return prefixed anisotropy metrics."""
        was_training = self.model.training
        self.model.eval()
        embeddings = self.model.encode(
            self.probe_texts,
            is_query=self.is_query,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        self.model.train(was_training)
        tokens = torch.cat([e.float().cpu() for e in embeddings], dim=0)
        return {f"{self.prefix}/{k}": v for k, v in token_anisotropy(tokens).items()}

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.every_n_steps:
            return
        metrics = self.measure()
        print(f"[step {state.global_step}] {metrics}", flush=True)
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(metrics, step=state.global_step)
        except ImportError:
            pass
