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

from src.experiment import ExperimentError
from src.recipe_logging import RecipeExperimentManager

CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "config")


def validate_recipe_config(cfg: DictConfig) -> None:
    """Validate that the composed config has a recipe selected and a valid
    entrypoint.

    Checks that:
    - A recipe is selected (recipe.name is not None or 'base_recipe')
    - The recipe defines either recipe.launch.command (for external launchers)
      or recipe.module (for in-process entrypoints)

    Fails immediately if validation fails, before any training process is
    spawned.

    Args:
        cfg: A Hydra-composed DictConfig.

    Raises:
        ExperimentError: If recipe.name is missing or 'base_recipe', or if
            neither recipe.launch.command nor recipe.module is defined.

    Example:
        >>> # Valid config
        >>> cfg = OmegaConf.create({"recipe": {"name": "spin", "launch": {"command": "bash run.sh"}}})
        >>> validate_recipe_config(cfg)  # Passes
        >>> # Invalid: no recipe
        >>> cfg2 = OmegaConf.create({})
        >>> validate_recipe_config(cfg2)  # Raises ExperimentError
    """
    recipe_name = OmegaConf.select(cfg, "recipe.name")
    if not recipe_name or recipe_name == "base_recipe":
        raise ExperimentError("No recipe selected. Pass `recipe=<name>` on the CLI, e.g. `recipe=spin`.")
    launch_command = OmegaConf.select(cfg, "recipe.launch.command")
    module = OmegaConf.select(cfg, "recipe.module")
    if not launch_command and not module:
        raise ExperimentError(f"Recipe '{recipe_name}' defines neither recipe.launch.command nor recipe.module.")


def run_launch_command(cfg: DictConfig, exp: RecipeExperimentManager) -> int:
    """Execute a recipe's external launcher in a subprocess.

    Runs the command specified in cfg.recipe.launch.command (e.g., SPIN's
    run_spin.sh), exporting environment variables so the launcher can
    correlate its logs and outputs with this WandB run and registry entry:
    - MEMOCR_RUN_DIR: The run directory created by the ExperimentManager
    - MEMOCR_RUN_NAME: The WandB run name
    - MEMOCR_RECIPE: The recipe name

    The subprocess is run in the directory specified by
    cfg.recipe.launch.cwd (default: '.', the repo root).

    Args:
        cfg: A Hydra-composed DictConfig with recipe.launch.command and
            optional recipe.launch.cwd.
        exp: A RecipeExperimentManager instance with run_dir and _run_name
            attributes already set (by start_run).

    Returns:
        The exit code of the subprocess (0 for success, non-zero for failure).

    Example:
        >>> cfg = OmegaConf.create({
        ...     "recipe": {
        ...         "launch": {
        ...             "command": "bash recipe/spin/run_spin.sh",
        ...             "cwd": "."
        ...         }
        ...     }
        ... })
        >>> returncode = run_launch_command(cfg, exp)
        >>> if returncode != 0:
        ...     raise ExperimentError(f"Recipe launcher failed with code {returncode}")
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
    """Placeholder for in-process recipe execution.

    In-process recipes (e.g., AgentLoopBase subclasses accessed via
    cfg.recipe.module) each have distinct call signatures and initialization
    requirements, so a single generic driver loop is not yet implemented.

    Currently raises NotImplementedError with a clear message identifying
    the missing recipe driver. This follows error-handling best practices
    (explicit failure rather than silent no-op) and guides the developer
    toward the next step: adding a recipe-specific driver to this function
    or to scripts/train_recipe.py.

    Args:
        cfg: A Hydra-composed DictConfig with recipe.module set.
        exp: A RecipeExperimentManager instance (not yet used, but available
            for logging/checkpointing once drivers are implemented).

    Raises:
        NotImplementedError: Always, with a message naming the recipe.module
            that needs a driver.

    Note:
        To add support for a new in-process recipe:
        1. Implement a driver function for the recipe (import its module,
           call its entrypoint with appropriate args)
        2. Add logic to run_in_process_module to dispatch to the driver
           based on cfg.recipe.module
        3. Ensure the driver uses exp.log_agent_loop_output() to record
           metrics from each training step.
    """
    raise NotImplementedError(
        f"In-process training for recipe.module='{cfg.recipe.module}' is not implemented. "
        "Add a recipe-specific driver in scripts/train_recipe.py, or set recipe.launch.command."
    )


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="recipe_config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for recipe-aware training.

    Orchestrates end-to-end training for a recipe (SPIN, LangGraph agent, etc.)
    using a Hydra-composed config. Flow:
    1. Validate that the config has a valid recipe selected
    2. If dry_run=true, print the resolved config and exit
    3. Create a RecipeExperimentManager (WandB + file logging with recipe tags)
    4. Start a run and log the full config
    5. Execute the recipe via either:
       - An external launcher (recipe.launch.command, e.g., SPIN's run_spin.sh)
       - An in-process module (recipe.module, currently raises NotImplementedError)
    6. Mark the run as completed or failed based on exit code

    Catches all exceptions during training, marks the run as failed, and
    re-raises the exception for proper exit code handling.

    Args:
        cfg: A Hydra-composed DictConfig loaded from config/recipe_config.yaml.
            Must include recipe, experiment, logging, memory, and model groups.
            Override via CLI: `python scripts/train_recipe.py recipe=<name>
            experiment=<name>`.

    Example:
        ```bash
        python scripts/train_recipe.py recipe=spin experiment=baseline
        python scripts/train_recipe.py recipe=langgraph_agent experiment=baseline \\
            +memory/ablations=high
        python scripts/train_recipe.py recipe=spin dry_run=true
        ```

    Raises:
        ExperimentError: If validation fails (invalid recipe or missing
            entrypoint) or if the recipe launcher exits with a non-zero code.
        Other exceptions may be raised by the recipe itself, caught, logged,
            and re-raised.
    """
    validate_recipe_config(cfg)
    if cfg.recipe.dry_run:
        print("=== Resolved Config ===")
        print(OmegaConf.to_yaml(cfg))
        return
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
