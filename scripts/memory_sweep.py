#!/usr/bin/env python
"""Grid-search sweep driver over memory density / budget / model variants.

Usage:
    python scripts/memory_sweep.py -m +experiment=ablation_density
    python scripts/memory_sweep.py -m +experiment=ablation_density \\
        --sweep-values memory/ablations=low,medium,high

This wraps `memory_train.main` for each point in the grid defined by
`experiment.sweep` (see `config/experiment/ablation_density.yaml`), running
trials in parallel processes (default: `multiprocessing`) or via Ray if
`sweep.backend=ray` and a Ray cluster is reachable.
"""

from __future__ import annotations

import multiprocessing as mp
import subprocess
import sys
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logging import get_logger  # noqa: E402

CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "config")
TRAIN_SCRIPT = Path(__file__).resolve().parent / "memory_train.py"

logger = get_logger(__name__)


def build_trial_overrides(cfg: DictConfig) -> list[list[str]]:
    """Expand `experiment.sweep` into a list of Hydra CLI override lists.

    e.g. sweep.param="memory/ablations", sweep.values=[low, medium, high]
    yields [["+memory/ablations=low"], ["+memory/ablations=medium"], ...].
    """
    sweep_cfg = OmegaConf.select(cfg, "experiment.sweep", default=None)
    if sweep_cfg is None:
        raise ValueError("experiment.sweep must be set (param + values) to run a sweep.")

    param = sweep_cfg["param"]
    values = sweep_cfg["values"]
    return [[f"+{param}={value}"] for value in values]


def run_trial(overrides: list[str], base_overrides: list[str]) -> int:
    """Run one training trial as a subprocess; returns its exit code.

    Subprocess isolation (rather than in-process Hydra re-init) avoids
    Hydra's global-state singleton clashing across trials run in the same
    process, and matches how `memory_train.py` is invoked standalone.
    """
    cmd = [sys.executable, str(TRAIN_SCRIPT), *base_overrides, *overrides]
    logger.info("sweep_trial_start overrides=%s", overrides)
    result = subprocess.run(cmd, capture_output=False)
    logger.info("sweep_trial_end overrides=%s returncode=%s", overrides, result.returncode)
    return result.returncode


def run_sweep(
    trial_override_sets: list[list[str]], base_overrides: list[str], max_workers: Optional[int] = None
) -> list[int]:
    """Run all trials in parallel using a process pool.

    `max_workers` defaults to `min(len(trials), os.cpu_count())` so a small
    sweep does not over-subscribe the machine.
    """
    max_workers = max_workers or min(len(trial_override_sets), mp.cpu_count())
    with mp.Pool(processes=max_workers) as pool:
        return pool.starmap(run_trial, [(overrides, base_overrides) for overrides in trial_override_sets])


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    trial_override_sets = build_trial_overrides(cfg)
    experiment_name = OmegaConf.select(cfg, "experiment.name", default="sweep")
    base_overrides = [f"+experiment={experiment_name}"]

    logger.info("sweep_start experiment=%s num_trials=%d", experiment_name, len(trial_override_sets))
    return_codes = run_sweep(trial_override_sets, base_overrides)

    failures = [rc for rc in return_codes if rc != 0]
    if failures:
        logger.error("sweep_completed_with_failures num_failed=%d/%d", len(failures), len(return_codes))
        sys.exit(1)
    logger.info("sweep_completed_successfully num_trials=%d", len(return_codes))


if __name__ == "__main__":
    main()
