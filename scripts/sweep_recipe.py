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

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logging import get_logger

CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "config")
TRAIN_RECIPE_SCRIPT = Path(__file__).resolve().parent / "train_recipe.py"

logger = get_logger(__name__)


def build_trial_overrides(cfg: DictConfig) -> list[list[str]]:
    """Build a grid of Hydra CLI overrides from the sweep configuration.

    Computes the cartesian product of:
    - recipe.sweep.recipe: list of recipe names (e.g., ["spin", "langgraph_agent"])
    - recipe.sweep.memory_ablation: list of memory ablation variants (e.g., ["low", "medium", "high"])

    Each (recipe, ablation) pair becomes one trial with a list of CLI overrides
    (e.g., ["recipe=spin", "+memory/ablations=low"]).

    Args:
        cfg: A Hydra-composed DictConfig with recipe.sweep containing
            'recipe' and 'memory_ablation' lists.

    Returns:
        A list of override lists, one per trial. Each inner list is suitable
        for passing to train_recipe.py as CLI arguments.

    Raises:
        ValueError: If cfg.recipe.sweep is not set (missing sweep config).

    Example:
        >>> cfg = OmegaConf.create({
        ...     "recipe": {
        ...         "sweep": {
        ...             "recipe": ["spin", "langgraph_agent"],
        ...             "memory_ablation": ["low", "high"]
        ...         }
        ...     }
        ... })
        >>> overrides = build_trial_overrides(cfg)
        >>> assert len(overrides) == 4  # 2 recipes x 2 ablations
        >>> assert ["recipe=spin", "+memory/ablations=low"] in overrides
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
    """Execute a single trial of recipe training in a subprocess.

    Runs `python scripts/train_recipe.py <base_overrides> <overrides>` in
    a subprocess and returns its exit code. The trial runs in isolation
    (separate process) to avoid global state clashes between trials
    (e.g., Hydra's global singleton, OmegaConf caches).

    Args:
        overrides: Trial-specific Hydra CLI overrides (e.g.,
            ["recipe=spin", "+memory/ablations=low"]).
        base_overrides: Shared overrides applied to all trials (e.g.,
            ["+experiment=ablation_density"]).

    Returns:
        The exit code of the subprocess (0 for success, non-zero for failure).

    Example:
        >>> returncode = run_trial(
        ...     ["recipe=spin", "+memory/ablations=low"],
        ...     ["+experiment=ablation_density"]
        ... )
        >>> assert returncode == 0  # Training succeeded
    """
    cmd = [sys.executable, str(TRAIN_RECIPE_SCRIPT), *base_overrides, *overrides]
    logger.info("recipe_sweep_trial_start overrides=%s", overrides)
    result = subprocess.run(cmd, capture_output=False)
    logger.info("recipe_sweep_trial_end overrides=%s returncode=%s", overrides, result.returncode)
    return result.returncode


def run_sweep(
    trial_override_sets: list[list[str]], base_overrides: list[str], max_workers: int | None = None
) -> list[int]:
    """Execute all trials in parallel using a process pool.

    Submits each trial to a multiprocessing.Pool for parallel execution.
    The pool size defaults to min(num_trials, cpu_count()) to maximize
    parallelism while respecting system resources.

    Each trial runs in isolation, so Hydra and other global state do not
    clash between trials.

    Args:
        trial_override_sets: A list of override lists, one per trial (as
            produced by build_trial_overrides).
        base_overrides: Shared overrides applied to all trials (e.g.,
            ["+experiment=ablation_density"]).
        max_workers: Optional maximum number of parallel workers. If None,
            defaults to min(len(trial_override_sets), mp.cpu_count()).

    Returns:
        A list of exit codes, one per trial, in the same order as
        trial_override_sets. Non-zero codes indicate failure.

    Example:
        >>> trial_overrides = [
        ...     ["recipe=spin", "+memory/ablations=low"],
        ...     ["recipe=spin", "+memory/ablations=high"],
        ... ]
        >>> base = ["+experiment=ablation_density"]
        >>> return_codes = run_sweep(trial_overrides, base, max_workers=2)
        >>> assert len(return_codes) == 2
    """
    max_workers = max_workers or min(len(trial_override_sets), mp.cpu_count())
    with mp.Pool(processes=max_workers) as pool:
        return pool.starmap(run_trial, [(overrides, base_overrides) for overrides in trial_override_sets])


def aggregate_return_codes(return_codes: list[int]) -> dict[str, int]:
    """Summarize trial exit codes into pass/fail counts.

    Args:
        return_codes: A list of exit codes (0 for success, non-zero for
            failure), one per trial.

    Returns:
        A dict with keys:
        - 'num_trials': Total number of trials
        - 'num_failed': Number of trials with non-zero exit codes

    Example:
        >>> summary = aggregate_return_codes([0, 0, 1, 0])
        >>> assert summary == {"num_trials": 4, "num_failed": 1}
    """
    failures = [rc for rc in return_codes if rc != 0]
    return {"num_trials": len(return_codes), "num_failed": len(failures)}


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="recipe_config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for memory ablation sweep over recipes.

    Orchestrates a grid search over recipe x memory-density combinations.
    Flow:
    1. Load the sweep configuration (recipe.sweep.recipe and
       recipe.sweep.memory_ablation lists)
    2. Expand the cartesian product into trial-specific Hydra overrides
    3. Run all trials in parallel subprocesses (pool size = min(num_trials, cpu_count))
    4. Aggregate results and exit with code 0 (all pass) or 1 (any failure)

    Each trial is isolated in its own subprocess to avoid global state
    clashes (Hydra singleton, OmegaConf caches, etc.).

    Args:
        cfg: A Hydra-composed DictConfig loaded from config/recipe_config.yaml
            with recipe=memory_ablation (or another preset that defines
            recipe.sweep).

    Example:
        ```bash
        # Run a 2x3 grid: [spin, langgraph_agent] x [low, medium, high]
        python scripts/sweep_recipe.py recipe=memory_ablation experiment=ablation_density
        ```

    Raises:
        SystemExit: With code 1 if any trial fails; 0 if all succeed.
        ValueError: If recipe.sweep is not configured.
    """
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
