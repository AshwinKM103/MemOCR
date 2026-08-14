#!/usr/bin/env python
"""Train a recipe (SPIN, LangGraph agent, ...) with a Hydra-composed config.

Usage:
    python scripts/train_recipe.py recipe=spin +experiment=baseline
    python scripts/train_recipe.py recipe=langgraph_agent +experiment=baseline

This is the recipe-aware sibling of `scripts/memory_train.py`: it wires
`RecipeExperimentManager` (WandB tags for recipe/memory-density/token-budget)
around each recipe's *existing* entrypoint rather than reimplementing
FSDP/PPO or agent-loop training here.

- If `recipe.launch.command` is set (e.g. SPIN's `run_spin.sh`, which owns
  its own Ray/FSDP dispatch), that command is subprocess-run with
  MEMOCR_RUN_DIR/MEMOCR_RUN_NAME/MEMOCR_RECIPE exported so its own logs can
  be correlated with this run's WandB run and registry entry.
- Otherwise `recipe.module` must resolve to an in-process entrypoint
  (currently unimplemented per-recipe; raises `NotImplementedError` naming
  the missing recipe so the failure is diagnosable, not silent).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment import ExperimentError  # noqa: E402
from src.recipe_logging import RecipeExperimentManager  # noqa: E402

CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "config")


def validate_recipe_config(cfg: DictConfig) -> None:
    """Fail fast if the composed config has no recipe selected, or the
    selected recipe has neither a launch command nor an in-process module.
    """
    recipe_name = OmegaConf.select(cfg, "recipe.name")
    if not recipe_name or recipe_name == "base_recipe":
        raise ExperimentError("No recipe selected. Pass `recipe=<name>` on the CLI, e.g. `recipe=spin`.")
    launch_command = OmegaConf.select(cfg, "recipe.launch.command")
    module = OmegaConf.select(cfg, "recipe.module")
    if not launch_command and not module:
        raise ExperimentError(f"Recipe '{recipe_name}' defines neither recipe.launch.command nor recipe.module.")


def run_launch_command(cfg: DictConfig, exp: RecipeExperimentManager) -> int:
    """Subprocess-run an existing recipe launcher (e.g. `run_spin.sh`),
    exporting run identity so its logs can be correlated with this run.
    """
    command = cfg.recipe.launch.command
    cwd = OmegaConf.select(cfg, "recipe.launch.cwd", default=".")
    env = {
        **os.environ,
        "MEMOCR_RUN_DIR": str(exp.run_dir),
        "MEMOCR_RUN_NAME": exp._run_name or "",
        "MEMOCR_RECIPE": exp.recipe_name,
    }
    exp.logger.info("recipe_launch_start", command=command, cwd=str(cwd))
    result = subprocess.run(command, shell=True, cwd=cwd, env=env, check=False)
    exp.logger.info("recipe_launch_end", command=command, returncode=result.returncode)
    return result.returncode


def run_in_process_module(cfg: DictConfig, exp: RecipeExperimentManager) -> None:
    """In-process recipes (e.g. `recipe.module` pointing at an
    `AgentLoopBase` subclass) are not yet wired to a generic driver loop --
    each agent-loop recipe has a distinct call signature. Raise clearly
    rather than silently no-op, per `.claude/rules/error-handling.md`.
    """
    raise NotImplementedError(
        f"In-process training for recipe.module='{cfg.recipe.module}' is not implemented. "
        "Add a recipe-specific driver in scripts/train_recipe.py, or set recipe.launch.command."
    )


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="recipe_config")
def main(cfg: DictConfig) -> None:
    validate_recipe_config(cfg)
    exp = RecipeExperimentManager(config=cfg)
    exp.start_run(name=cfg.run.name)
    exp.log_config()
    try:
        if cfg.recipe.launch.command:
            returncode = run_launch_command(cfg, exp)
            if returncode != 0:
                raise ExperimentError(f"Recipe launch command exited with code {returncode}")
        else:
            run_in_process_module(cfg, exp)
    except Exception:
        exp.end_run(status="failed")
        raise
    else:
        exp.end_run(status="completed")


if __name__ == "__main__":
    main()
