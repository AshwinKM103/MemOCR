#!/usr/bin/env python
"""Grid-search sweep driver over recipe x memory-density ablations.

Usage:
    python scripts/sweep_recipe.py +recipe=memory_ablation +experiment=ablation_density

This is the recipe-aware sibling of `scripts/memory_sweep.py`: it wraps
`train_recipe.main` for each (recipe, memory_ablation) point in
`recipe.sweep` (see `config/recipe/memory_ablation.yaml`), running trials in
parallel subprocesses so Hydra's per-process global state does not clash
across trials -- same isolation strategy as `memory_sweep.py`.
"""

from __future__ import annotations

import itertools
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
TRAIN_RECIPE_SCRIPT = Path(__file__).resolve().parent / "train_recipe.py"

logger = get_logger(__name__)


def build_trial_overrides(cfg: DictConfig) -> list[list[str]]:
    """Expand `recipe.sweep` into a Hydra CLI override list per trial: the
    cartesian product of `recipe.sweep.recipe` and
    `recipe.sweep.memory_ablation`.
    """
    sweep_cfg = OmegaConf.select(cfg, "recipe.sweep", default=None)
    if sweep_cfg is None:
        raise ValueError("recipe.sweep must be set (recipe + memory_ablation) to run a sweep.")

    recipes = sweep_cfg["recipe"]
    ablations = sweep_cfg["memory_ablation"]
    return [
        [f"recipe={recipe_name}", f"+memory/ablations={ablation}"]
        for recipe_name, ablation in itertools.product(recipes, ablations)
    ]


def run_trial(overrides: list[str], base_overrides: list[str]) -> int:
    """Run one recipe training trial as a subprocess; returns its exit code."""
    cmd = [sys.executable, str(TRAIN_RECIPE_SCRIPT), *base_overrides, *overrides]
    logger.info("recipe_sweep_trial_start overrides=%s", overrides)
    result = subprocess.run(cmd, capture_output=False)
    logger.info("recipe_sweep_trial_end overrides=%s returncode=%s", overrides, result.returncode)
    return result.returncode


def run_sweep(
    trial_override_sets: list[list[str]], base_overrides: list[str], max_workers: Optional[int] = None
) -> list[int]:
    """Run all trials in parallel using a process pool."""
    max_workers = max_workers or min(len(trial_override_sets), mp.cpu_count())
    with mp.Pool(processes=max_workers) as pool:
        return pool.starmap(run_trial, [(overrides, base_overrides) for overrides in trial_override_sets])


def aggregate_return_codes(return_codes: list[int]) -> dict[str, int]:
    """Summarize trial outcomes for the final log line / exit decision."""
    failures = [rc for rc in return_codes if rc != 0]
    return {"num_trials": len(return_codes), "num_failed": len(failures)}


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="recipe_config")
def main(cfg: DictConfig) -> None:
    trial_override_sets = build_trial_overrides(cfg)
    experiment_name = OmegaConf.select(cfg, "experiment.name", default="sweep")
    base_overrides = [f"+experiment={experiment_name}"]

    logger.info("recipe_sweep_start experiment=%s num_trials=%d", experiment_name, len(trial_override_sets))
    return_codes = run_sweep(trial_override_sets, base_overrides)
    summary = aggregate_return_codes(return_codes)

    if summary["num_failed"]:
        logger.error(
            "recipe_sweep_completed_with_failures num_failed=%d/%d",
            summary["num_failed"],
            summary["num_trials"],
        )
        sys.exit(1)
    logger.info("recipe_sweep_completed_successfully num_trials=%d", summary["num_trials"])


if __name__ == "__main__":
    main()
