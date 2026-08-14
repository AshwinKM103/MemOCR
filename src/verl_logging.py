"""VERL-aware experiment logging: extends `src.experiment.ExperimentManager`
with multi-GPU rank coordination, VERL trainer tags, throughput/FLOPs metric
helpers, and FSDP-aware checkpoint metadata.

This manager runs *alongside* VERL's own `verl.utils.tracking.Tracking`
(which `SFTTrainer.fit()` / `RayPPOTrainer` already drive from
`config.trainer.logger`), not in place of it. VERL's own Tracking owns the
WandB run that receives per-step training metrics logged inside the trainer
loop (see `config/verl/*.yaml`, which sets `trainer.logger: [console, wandb]`
so that happens with zero trainer code changes). `VERLExperimentManager`
instead owns:

- The MemOCR-side run registry entry (`registry.jsonl`) and checkpoint
  metadata sidecars, consistent with the recipe/memory_train runs.
- Rank-0-gated summary/benchmark logging from `scripts/train_verl.py` and
  `scripts/benchmark_verl.py`, which run outside the trainer's per-step loop.
- Multi-GPU metric aggregation for values collected *by our own wrapper
  scripts* (e.g. benchmark throughput samples gathered per-rank).

Usage:
    from src.verl_logging import VERLExperimentManager, is_rank_zero

    exp = VERLExperimentManager(config=cfg, trainer_type="sft")
    if is_rank_zero():
        exp.start_run(name=cfg.run.name)
    exp.log_training_step(step=10, metrics={"loss": 0.42}, tokens_per_sec=1200.0)
    exp.log_checkpoint_metadata(step=10, checkpoint_dir=ckpt_dir, world_size=8,
                                 sharding_strategy="FULL_SHARD")
    if is_rank_zero():
        exp.end_run()
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from omegaconf import DictConfig, OmegaConf, open_dict

from src.experiment import ExperimentManager

# Trainer types this bridge understands; kept in sync with config/verl/*.yaml
# `trainer_type` values.
_KNOWN_TRAINER_TYPES = frozenset({"sft", "ppo", "rl"})


def get_rank() -> int:
    """Current distributed rank, or 0 if `torch.distributed` is unavailable
    or not yet initialized (e.g. single-process runs, unit tests).

    Deferred `torch` import so this module (and `is_rank_zero`) can be used
    in CPU-only test environments without torch installed with CUDA/NCCL
    support.
    """
    try:
        import torch.distributed as dist
    except ImportError:
        return 0
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Current distributed world size, or 1 if not running distributed."""
    try:
        import torch.distributed as dist
    except ImportError:
        return 1
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_rank_zero() -> bool:
    """True if this process should perform WandB/registry/log I/O.

    Every call site that writes to WandB, the run registry, or checkpoint
    metadata must gate on this (or pass through `VERLExperimentManager`,
    which gates internally) — logging from every rank would duplicate
    WandB series and interleave writes to the same registry.jsonl file.
    """
    return get_rank() == 0


def aggregate_metrics_across_ranks(local_metrics: dict[str, float]) -> dict[str, float]:
    """Average scalar metrics across all distributed ranks.

    Uses `torch.distributed.all_reduce` (AVG) when running distributed;
    returns `local_metrics` unchanged for single-process/non-distributed
    callers (including unit tests), so this is safe to call unconditionally
    from a trainer wrapper.

    Args:
        local_metrics: Scalar metrics computed on the calling rank only,
            e.g. `{"throughput/tokens_per_sec": 1187.3}`.

    Returns:
        A new dict with the same keys, each averaged across ranks.
    """
    if not local_metrics:
        return {}
    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        return dict(local_metrics)
    if not dist.is_available() or not dist.is_initialized():
        return dict(local_metrics)

    keys = sorted(local_metrics)
    values = torch.tensor([float(local_metrics[k]) for k in keys], dtype=torch.float64)
    if torch.cuda.is_available():
        values = values.cuda()
    dist.all_reduce(values, op=dist.ReduceOp.AVG)
    return {k: v.item() for k, v in zip(keys, values.cpu(), strict=True)}


class VERLExperimentManager(ExperimentManager):
    """`ExperimentManager` subclass for VERL trainer runs (SFT/PPO/RL).

    Adds VERL-specific tracking on top of the base `ExperimentManager`:

    - `trainer_type` tag ("sft" | "ppo" | "rl") merged into WandB tags.
    - `log_training_step`: rank-0-gated metric logging with throughput/memory
      metric namespacing (`throughput/*`, `memory/*` prefixes) so dashboards
      group VERL performance metrics separately from task metrics.
    - `log_metrics_aggregated`: cross-rank averaging via
      `aggregate_metrics_across_ranks()`, then rank-0-gated logging.
    - `log_checkpoint_metadata`: writes a `verl_metadata.json` sidecar with
      FSDP context (world_size, sharding_strategy) alongside VERL's own
      `CheckpointHandler` checkpoint directory.
    - `log_config`: logs the fully-resolved config as structured output.

    All logging methods are rank-0-gated and safe to call unconditionally
    from distributed training code.

    Example:
        >>> from src.verl_logging import VERLExperimentManager, is_rank_zero
        >>> exp = VERLExperimentManager(config=cfg, trainer_type="sft")
        >>> if is_rank_zero():
        ...     exp.start_run(name="my_training_run")
        ...     exp.log_config()
        >>> exp.log_training_step(
        ...     step=100,
        ...     metrics={"loss": 0.42},
        ...     tokens_per_sec=1200.0,
        ...     peak_memory_gb=24.5
        ... )
        >>> if is_rank_zero():
        ...     exp.end_run(status="completed")
    """

    def __init__(self, config: DictConfig, trainer_type: str) -> None:
        """Initialize a VERL experiment manager.

        Merges the trainer type tag into WandB tags and initializes the
        base `ExperimentManager` with the enhanced config.

        Args:
            config: Hydra OmegaConf config with `logging.wandb` section.
            trainer_type: One of "sft", "ppo", "rl". Used for WandB tagging
                and internal tracking.

        Raises:
            ValueError: If `trainer_type` is not one of the known types.
        """
        if trainer_type not in _KNOWN_TRAINER_TYPES:
            raise ValueError(
                f"Unknown VERL trainer_type '{trainer_type}'; expected one of {sorted(_KNOWN_TRAINER_TYPES)}"
            )
        merged_config = self._with_trainer_tag(config, trainer_type)
        super().__init__(config=merged_config)
        self.trainer_type = trainer_type

    @staticmethod
    def _with_trainer_tag(config: DictConfig, trainer_type: str) -> DictConfig:
        """Merge a `verl:<trainer_type>` WandB tag into `logging.wandb.tags`.

        Follows the same pattern as `RecipeExperimentManager`. No-op if
        `logging.wandb` is not present in the config. Handles struct-locked
        configs by using `open_dict` context.

        Args:
            config: Hydra OmegaConf config with optional `logging.wandb` section.
            trainer_type: One of "sft", "ppo", "rl". Tag added as `verl:<trainer_type>`.

        Returns:
            A modified config with the trainer tag appended to `logging.wandb.tags`.
            The original `config` is unchanged if `logging.wandb` is not present.
        """
        if OmegaConf.select(config, "logging.wandb") is None:
            return config
        existing = list(OmegaConf.select(config, "logging.wandb.tags", default=[]))
        tag = f"verl:{trainer_type}"
        if tag not in existing:
            existing.append(tag)
        # `logging.wandb` may be struct-locked with no `tags` key (e.g.
        # config/logging/console.yaml, which only sets `enabled: false`);
        # OmegaConf.merge alone raises ConfigKeyError in that case, so the
        # new key is added under an explicit `open_dict` context.
        merged = OmegaConf.merge(config, {})
        assert isinstance(merged, DictConfig)
        with open_dict(merged.logging.wandb):
            merged.logging.wandb.tags = existing
        return merged

    def log_training_step(
        self,
        step: int,
        metrics: dict[str, float],
        tokens_per_sec: float | None = None,
        samples_per_sec: float | None = None,
        peak_memory_gb: float | None = None,
        reserved_memory_gb: float | None = None,
    ) -> None:
        """Log a training step's task metrics plus optional performance metrics.

        Performance metrics are namespaced (`throughput/*`, `memory/*`) so they
        group separately from task metrics (loss, accuracy, ...) in the WandB UI.
        Silently no-ops on non-zero ranks so trainer wrapper code can call this
        unconditionally.

        Args:
            step: Global training step counter.
            metrics: Task metrics as a dict (e.g., `{"loss": 0.42}`).
            tokens_per_sec: Throughput in tokens per second. If provided, logged
                as `throughput/tokens_per_sec`.
            samples_per_sec: Throughput in samples per second. If provided, logged
                as `throughput/samples_per_sec`.
            peak_memory_gb: Peak GPU memory used (GB). If provided, logged as
                `memory/peak_gb`.
            reserved_memory_gb: Reserved GPU memory (GB). If provided, logged as
                `memory/reserved_gb`.

        Note:
            Only rank 0 performs the actual WandB/registry write. Non-zero ranks
            return immediately without error.
        """
        if not is_rank_zero():
            return
        enriched = dict(metrics)
        if tokens_per_sec is not None:
            enriched["throughput/tokens_per_sec"] = tokens_per_sec
        if samples_per_sec is not None:
            enriched["throughput/samples_per_sec"] = samples_per_sec
        if peak_memory_gb is not None:
            enriched["memory/peak_gb"] = peak_memory_gb
        if reserved_memory_gb is not None:
            enriched["memory/reserved_gb"] = reserved_memory_gb
        self.log_metrics(step=step, metrics=enriched)

    def log_metrics_aggregated(self, step: int, local_metrics: dict[str, float]) -> dict[str, float]:
        """Average metrics across all ranks, then log once from rank 0.

        Uses `torch.distributed.all_reduce` with AVG mode when running distributed.
        Returns the aggregated dict on every rank so callers can assert on it
        in tests without rank-0 gating, but only rank 0 performs the actual
        WandB/registry write.

        Args:
            step: Global training step counter.
            local_metrics: Scalar metrics computed on the calling rank (e.g.,
                `{"reward": 0.9}`).

        Returns:
            A new dict with the same keys, each value averaged across all ranks.
            For single-process runs, returns a copy of `local_metrics` unchanged.
        """
        aggregated = aggregate_metrics_across_ranks(local_metrics)
        if is_rank_zero():
            self.log_metrics(step=step, metrics=aggregated)
        return aggregated

    def log_checkpoint_metadata(
        self,
        step: int,
        checkpoint_dir: Path | str,
        world_size: int | None = None,
        sharding_strategy: str | None = None,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        """Write a `verl_metadata.json` sidecar alongside a VERL checkpoint.

        Records distributed-training context (world size, sharding strategy,
        step, timestamp, metrics) that VERL's own `CheckpointHandler` does not
        persist. Does not manage checkpoint files themselves — only creates the
        metadata sidecar.

        This method no-ops (returns the path without writing) on non-zero ranks,
        so it is safe to call unconditionally from distributed training code.

        Args:
            step: Global training step this checkpoint corresponds to.
            checkpoint_dir: Path to the VERL checkpoint directory for this step.
                Must already exist (created by `verl.utils.checkpoint.CheckpointHandler`).
            world_size: Distributed world size at save time. Defaults to
                `get_world_size()` if not provided.
            sharding_strategy: FSDP sharding strategy name (e.g., "FULL_SHARD").
                Optional; stored as-is in metadata.
            metrics: Metrics to embed alongside checkpoint metadata (e.g.,
                eval loss at this step). Optional; stored as empty dict if not provided.

        Returns:
            Path to `verl_metadata.json`. On non-zero ranks, returns the expected
            path without writing. On rank 0, the file is written if
            `self._run_id` is not None.

        Example:
            >>> exp = VERLExperimentManager(config=cfg, trainer_type="sft")
            >>> exp.start_run(name="my_run")
            >>> checkpoint_dir = Path("./checkpoints/step_00010")
            >>> checkpoint_dir.mkdir(parents=True, exist_ok=True)
            >>> metadata_path = exp.log_checkpoint_metadata(
            ...     step=10,
            ...     checkpoint_dir=checkpoint_dir,
            ...     world_size=8,
            ...     sharding_strategy="FULL_SHARD",
            ...     metrics={"val_loss": 0.21}
            ... )
            >>> # On rank 0, metadata_path now exists with full metadata
        """
        checkpoint_dir = Path(checkpoint_dir)
        if not is_rank_zero():
            return checkpoint_dir / "verl_metadata.json"

        metadata = {
            "step": step,
            "trainer_type": self.trainer_type,
            "run_id": self._run_id,
            "run_name": self._run_name,
            "timestamp": time.time(),
            "world_size": world_size if world_size is not None else get_world_size(),
            "sharding_strategy": sharding_strategy,
            "metrics": metrics or {},
        }
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = checkpoint_dir / "verl_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
        if self._run_id is not None:
            self.logger.info("verl_checkpoint_metadata_saved", step=step, path=str(metadata_path))
        return metadata_path

    def log_config(self) -> None:
        """Emit the fully-resolved config as a structured log line.

        Gated to rank 0 (silent no-op on other ranks). Mirrors
        `RecipeExperimentManager.log_config`. Resolves all interpolations
        and outputs the config as a container (dict) in the log record.

        Note:
            Only rank 0 performs the actual write. Safe to call unconditionally
            from distributed code.
        """
        if not is_rank_zero():
            return
        resolved = OmegaConf.to_container(self.config, resolve=True)
        self.logger.info("verl_config", trainer_type=self.trainer_type, config=resolved)


def estimate_flops_per_second(estimated_flops: float, elapsed_seconds: float) -> float:
    """Compute FLOPs per second from estimated FLOPs and elapsed time.

    Given VERL's `FlopsCounter.estimate_flops()` output (verl/utils/flops_counter.py)
    and the measured wall-clock time for a training step, returns the throughput
    in FLOPs/sec.

    Kept as a standalone helper (not a method) so `scripts/benchmark_verl.py`
    can use it without constructing a full trainer/engine.

    Args:
        estimated_flops: Total FLOPs computed for the step (from VERL's FLOPs
            counter or a manual estimation).
        elapsed_seconds: Wall-clock time in seconds for the step.

    Returns:
        FLOPs per second (estimated_flops / elapsed_seconds).

    Raises:
        ValueError: If `elapsed_seconds` is not positive.
    """
    if elapsed_seconds <= 0:
        raise ValueError(f"elapsed_seconds must be positive, got {elapsed_seconds}")
    return estimated_flops / elapsed_seconds
