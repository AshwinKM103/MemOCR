#!/usr/bin/env python
"""Throughput/memory/FLOPs benchmarking for VERL trainers.

Runs a fixed number of dummy training-shaped iterations (no real model
required by default; pass `--engine-config` to profile a real
`verl.workers.engine` instance) and logs aggregated performance metrics to
WandB via `VERLExperimentManager`.

Usage:
    python scripts/benchmark_verl.py verl=sft +experiment=baseline \\
        benchmark.num_steps=50 benchmark.batch_size=8 benchmark.seq_len=2048

Metrics logged (namespaced under `throughput/` and `memory/`, see
`VERLExperimentManager.log_training_step`):
    - throughput/tokens_per_sec, throughput/samples_per_sec
    - memory/peak_gb, memory/reserved_gb (CUDA only; 0 on CPU)
    - throughput/estimated_flops_per_sec (only if `--flops` estimate given)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.verl_logging import (
    VERLExperimentManager,
    estimate_flops_per_second,
    get_world_size,
    is_rank_zero,
)

CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "config")
CONFIG_NAME = "verl_config"


@dataclass
class BenchmarkResult:
    """Summary of a benchmark run for programmatic use and WandB logging.

    Attributes:
        num_steps: Number of dummy training steps executed.
        total_seconds: Wall-clock time in seconds for all steps.
        tokens_per_sec: Throughput in tokens per second.
        samples_per_sec: Throughput in samples per second.
        peak_memory_gb: Peak GPU memory used (GB). 0.0 on CPU-only runs.
        reserved_memory_gb: Reserved GPU memory (GB). 0.0 on CPU-only runs.
        world_size: Distributed world size (1 for single-process).

    Optional attributes (set dynamically by callers):
        flops_per_sec: FLOPs per second (if `--flops` estimate was provided).
    """

    num_steps: int
    total_seconds: float
    tokens_per_sec: float
    samples_per_sec: float
    peak_memory_gb: float
    reserved_memory_gb: float
    world_size: int


def _peak_and_reserved_memory_gb() -> tuple[float, float]:
    """Get CUDA peak and reserved memory for the current device.

    Returns peak and reserved GPU memory in GB. On CPU-only systems or when
    CUDA is unavailable, returns (0.0, 0.0).

    Returns:
        A tuple of (peak_memory_gb, reserved_memory_gb). Peak is the maximum
        memory allocated across all tensors on device since the last reset.
        Reserved is the total memory allocated by the CUDA memory allocator.
    """
    try:
        import torch
    except ImportError:
        return 0.0, 0.0
    if not torch.cuda.is_available():
        return 0.0, 0.0
    peak_bytes = torch.cuda.max_memory_allocated()
    reserved_bytes = torch.cuda.max_memory_reserved()
    return peak_bytes / 1e9, reserved_bytes / 1e9


def run_dummy_step(batch_size: int, seq_len: int) -> None:
    """Execute one dummy compute step simulating a training iteration.

    Allocates and immediately frees a tensor shaped like an activation batch
    (batch_size, seq_len, hidden_dim) so peak-memory measurement is meaningful
    without loading a real model. On CPU-only systems, sleeps briefly instead.

    Args:
        batch_size: Batch size for the dummy tensor.
        seq_len: Sequence length for the dummy tensor.

    Note:
        The hidden dimension is hardcoded to 4096 to simulate a typical LLM
        activation batch.
    """
    try:
        import torch
    except ImportError:
        time.sleep(0.001)
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy = torch.randn(batch_size, seq_len, 4096, device=device, dtype=torch.float32)
    _ = (dummy * 2.0).sum()
    del dummy


def run_benchmark(
    num_steps: int,
    batch_size: int,
    seq_len: int,
    flops_per_step: float | None = None,
) -> BenchmarkResult:
    """Run dummy training steps and measure throughput and memory metrics.

    Executes `num_steps` dummy steps, measuring wall-clock throughput in
    tokens/sec and samples/sec. If CUDA is available, also measures peak and
    reserved GPU memory. Optionally estimates FLOPs/sec if `flops_per_step`
    is provided.

    Args:
        num_steps: Number of dummy steps to execute. Must be positive.
        batch_size: Batch size per step. Must be positive.
        seq_len: Sequence length per step. Must be positive.
        flops_per_step: Optional FLOPs estimate per step. If provided, the
            result includes a `flops_per_sec` attribute.

    Returns:
        BenchmarkResult with throughput and memory metrics.

    Raises:
        ValueError: If num_steps, batch_size, or seq_len are not positive.

    Example:
        >>> result = run_benchmark(num_steps=50, batch_size=8, seq_len=2048)
        >>> print(f"Throughput: {result.tokens_per_sec:.0f} tokens/sec")
        >>> print(f"Memory: {result.peak_memory_gb:.1f} GB peak")
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if batch_size <= 0 or seq_len <= 0:
        raise ValueError(f"batch_size and seq_len must be positive, got batch_size={batch_size}, seq_len={seq_len}")

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

    start = time.perf_counter()
    for _ in range(num_steps):
        run_dummy_step(batch_size=batch_size, seq_len=seq_len)
    total_seconds = time.perf_counter() - start

    total_tokens = num_steps * batch_size * seq_len
    total_samples = num_steps * batch_size
    peak_gb, reserved_gb = _peak_and_reserved_memory_gb()

    result = BenchmarkResult(
        num_steps=num_steps,
        total_seconds=total_seconds,
        tokens_per_sec=total_tokens / total_seconds,
        samples_per_sec=total_samples / total_seconds,
        peak_memory_gb=peak_gb,
        reserved_memory_gb=reserved_gb,
        world_size=get_world_size(),
    )
    if flops_per_step is not None:
        result.flops_per_sec = estimate_flops_per_second(  # type: ignore[attr-defined]
            flops_per_step * num_steps, total_seconds
        )
    return result


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for VERL trainer benchmarking.

    Composes the config, runs a benchmark with parameters from cfg.benchmark.*,
    prints results to stdout, and logs metrics to WandB via
    `VERLExperimentManager`.

    Args:
        cfg: Hydra OmegaConf config with `benchmark` and `verl` sections.

    Example CLI usage:
        python scripts/benchmark_verl.py verl=sft experiment=baseline \\
            benchmark.num_steps=50 benchmark.batch_size=8 benchmark.seq_len=2048
    """
    num_steps = int(OmegaConf.select(cfg, "benchmark.num_steps", default=20))
    batch_size = int(OmegaConf.select(cfg, "benchmark.batch_size", default=4))
    seq_len = int(OmegaConf.select(cfg, "benchmark.seq_len", default=1024))

    result = run_benchmark(num_steps=num_steps, batch_size=batch_size, seq_len=seq_len)

    if is_rank_zero():
        print(f"=== Benchmark result ({cfg.verl.trainer_type}) ===")
        print(result)

    exp = VERLExperimentManager(config=cfg, trainer_type=cfg.verl.trainer_type)
    if is_rank_zero():
        exp.start_run(name=f"benchmark_{cfg.run.name}")
    exp.log_training_step(
        step=0,
        metrics={},
        tokens_per_sec=result.tokens_per_sec,
        samples_per_sec=result.samples_per_sec,
        peak_memory_gb=result.peak_memory_gb,
        reserved_memory_gb=result.reserved_memory_gb,
    )
    if is_rank_zero():
        exp.end_run(status="completed")


if __name__ == "__main__":
    main()
