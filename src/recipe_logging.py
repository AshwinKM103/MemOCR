"""Recipe-aware experiment logging: extends `src.experiment.ExperimentManager`
with WandB tags/metadata derived from `recipe.*` and `memory.*` config, and
with the ability to record metrics coming out of `AgentLoopOutput.metrics`
(the dict every recipe/langgraph_agent loop already populates).

Usage:
    from src.recipe_logging import RecipeExperimentManager

    exp = RecipeExperimentManager(config=cfg)
    exp.start_run(name=cfg.run.name)
    exp.log_agent_loop_output(step=1, output=agent_loop_output)
    exp.save_checkpoint(step=1, metrics={"reward": 0.8})
    exp.end_run()
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.experiment import ExperimentManager

# Fields on AgentLoopOutput (verl.experimental.agent_loop.agent_loop) that are
# not scalar metrics and must not be forwarded to WandB as-is.
_AGENT_LOOP_NON_METRIC_FIELDS = frozenset({"metrics", "response_mask", "prompt_ids", "response_ids"})


def resolve_recipe_tags(config: DictConfig) -> list[str]:
    """Build the WandB tag list for a run: experiment name, recipe tags, and
    memory density / token budget so runs are filterable by ablation axis
    without parsing config.json after the fact.

    Computes tags from `experiment.name`, `recipe.name`, `recipe.tags`,
    `memory.density.target_tokens_per_1k_context`, and
    `memory.max_memorization_length`. Missing fields are skipped (not
    errored). Duplicates are deduplicated while preserving order.

    Args:
        config: A Hydra DictConfig containing experiment, recipe, and memory
            configuration. Field names are resolved using OmegaConf.select(),
            so interpolations and defaults are expanded.

    Returns:
        A list of tag strings ready to pass to WandB. Tags follow the format:
        - Experiment name (if present)
        - recipe:<name> (if recipe.name is set)
        - Each tag in recipe.tags (if present)
        - memory_density:<value> (if density is set)
        - token_budget:<value> (if budget is set)
        Returns an empty list if no tags are found.

    Example:
        >>> cfg = OmegaConf.create({
        ...     "experiment": {"name": "baseline_hotpotqa"},
        ...     "recipe": {"name": "spin_fsdp", "tags": ["spin", "fsdp"]},
        ...     "memory": {
        ...         "density": {"target_tokens_per_1k_context": 64},
        ...         "max_memorization_length": 256,
        ...     }
        ... })
        >>> tags = resolve_recipe_tags(cfg)
        >>> assert "recipe:spin_fsdp" in tags
        >>> assert "memory_density:64" in tags
    """
    tags: list[str] = []

    experiment_name = OmegaConf.select(config, "experiment.name")
    if experiment_name:
        tags.append(str(experiment_name))

    recipe_name = OmegaConf.select(config, "recipe.name")
    if recipe_name:
        tags.append(f"recipe:{recipe_name}")

    recipe_tags = OmegaConf.select(config, "recipe.tags", default=[])
    tags.extend(str(t) for t in recipe_tags)

    density = OmegaConf.select(config, "memory.density.target_tokens_per_1k_context")
    if density is not None:
        tags.append(f"memory_density:{density}")

    token_budget = OmegaConf.select(config, "memory.max_memorization_length")
    if token_budget is not None:
        tags.append(f"token_budget:{token_budget}")

    # Preserve order, drop duplicates.
    seen: set[str] = set()
    deduped = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            deduped.append(tag)
    return deduped


def agent_loop_output_to_metrics(output: Any) -> dict[str, float]:
    """Flatten `AgentLoopOutput.metrics` (and any top-level scalar attributes)
    into a `{metric_name: value}` dict with structured `agent/` field names,
    ready for `ExperimentManager.log_metrics` / WandB.

    Accepts either an `AgentLoopOutput` instance or a plain dict (as produced
    by tests / non-agent recipes), so callers do not need to import
    `verl.experimental.agent_loop.agent_loop` just to log metrics.

    Args:
        output: Either an AgentLoopOutput instance or a dict. If a dict, looks
            for a 'metrics' key; if not found, treats the entire dict as metrics.
            If an object, extracts the 'metrics' attribute if present, and also
            iterates over all attributes looking for scalar (int/float) values.

    Returns:
        A flat dict of {metric_name: float} pairs with 'agent/' prefix applied
        to all keys. Non-scalar values and special fields
        (response_mask, prompt_ids, response_ids, etc.) are excluded.
        Returns an empty dict if no scalar metrics are found.

    Example:
        >>> output = {"metrics": {"reward": 0.8, "turns": 3, "loss": 0.1}}
        >>> metrics = agent_loop_output_to_metrics(output)
        >>> assert metrics == {"agent/reward": 0.8, "agent/turns": 3.0, "agent/loss": 0.1}
        >>> # Non-scalar values are dropped:
        >>> output2 = {"metrics": {"reward": 0.5, "trace": ["a", "b"]}}
        >>> metrics2 = agent_loop_output_to_metrics(output2)
        >>> assert metrics2 == {"agent/reward": 0.5}
    """
    raw_metrics: dict[str, Any] = {}
    if isinstance(output, dict):
        raw_metrics = dict(output.get("metrics", output))
    else:
        metrics_attr = getattr(output, "metrics", None)
        if metrics_attr is not None:
            raw_metrics = dict(metrics_attr) if not isinstance(metrics_attr, dict) else metrics_attr
        for field_name in vars(output) if hasattr(output, "__dict__") else []:
            if field_name in _AGENT_LOOP_NON_METRIC_FIELDS:
                continue
            value = getattr(output, field_name)
            if isinstance(value, (int, float)):
                raw_metrics.setdefault(field_name, value)

    flattened: dict[str, float] = {}
    for key, value in raw_metrics.items():
        if isinstance(value, (int, float)):
            flattened[f"agent/{key}"] = float(value)
    return flattened


class RecipeExperimentManager(ExperimentManager):
    """`ExperimentManager` subclass with recipe-aware WandB tags and agent-loop
    metrics ingestion.

    Extends the base `ExperimentManager` to automatically compute WandB tags
    from recipe configuration (recipe name, tags, memory density, token budget)
    and merge them into the run config before `start_run` is called. This
    allows all recipe runs to be filtered in the WandB UI by recipe and memory
    ablation without manual dashboard configuration.

    Also provides `log_agent_loop_output`, which flattens `AgentLoopOutput.metrics`
    dicts into WandB-ready scalar metrics with 'agent/' prefix.

    Attributes:
        recipe_name (str): The recipe name from the config, used for logging
            and correlation with external processes (e.g., SPIN's ray launcher).

    Example:
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.create({
        ...     "recipe": {"name": "spin_fsdp", "tags": ["fsdp"]},
        ...     "memory": {"max_memorization_length": 256},
        ...     "experiment": {"name": "baseline"},
        ...     "run": {"name": "my_run"},
        ...     "logging": {"wandb": {"enabled": True}}
        ... })
        >>> exp = RecipeExperimentManager(config=cfg)
        >>> exp.start_run(name=cfg.run.name)
        >>> exp.log_agent_loop_output(step=1, output={"metrics": {"reward": 0.9}})
        >>> exp.end_run()
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize the manager with recipe-aware config.

        Computes WandB tags from the config (experiment name, recipe tags,
        memory density, token budget) and merges them into
        `config.logging.wandb.tags` before calling the parent `__init__`.
        The merged config is then passed to the parent class.

        Args:
            config: A fully-composed Hydra DictConfig containing recipe,
                memory, experiment, and logging sections. The config is
                expected to have recipe.name, and may have optional
                recipe.tags, memory.density, and memory.max_memorization_length.
        """
        merged_config = self._with_recipe_tags(config)
        super().__init__(config=merged_config)
        self.recipe_name: str = OmegaConf.select(merged_config, "recipe.name", default="unknown_recipe")

    @staticmethod
    def _with_recipe_tags(config: DictConfig) -> DictConfig:
        """Merge computed recipe tags into the WandB config.

        Calls `resolve_recipe_tags` to compute the tag list, then merges it
        into `config.logging.wandb.tags`. If `logging.wandb` is not present
        in the config, returns the config unchanged (no error).

        Args:
            config: A Hydra DictConfig to merge tags into.

        Returns:
            A new DictConfig with recipe tags merged into
            config.logging.wandb.tags. The original config is not modified.
        """
        tags = resolve_recipe_tags(config)
        if OmegaConf.select(config, "logging.wandb") is None:
            return config
        merged = OmegaConf.merge(config, OmegaConf.create({"logging": {"wandb": {"tags": tags}}}))
        assert isinstance(merged, DictConfig)
        return merged

    def log_agent_loop_output(self, step: int, output: Any) -> None:
        """Log an AgentLoopOutput (or dict with a metrics key) as WandB metrics.

        Flattens the output's metrics into scalar key-value pairs prefixed
        with 'agent/', then logs them via the parent's `log_metrics` method
        (which forwards to WandB, JSON sink, or both depending on config).

        Args:
            step: The training step or iteration number to associate with
                these metrics.
            output: An AgentLoopOutput instance, or a dict containing a
                'metrics' key. Passed to `agent_loop_output_to_metrics` for
                flattening. Non-scalar values are dropped.

        Example:
            >>> exp.log_agent_loop_output(step=1, output={
            ...     "metrics": {"reward": 0.8, "turns": 3}
            ... })
            # Logs: {"agent/reward": 0.8, "agent/turns": 3.0} at step 1.
        """
        metrics = agent_loop_output_to_metrics(output)
        if metrics:
            self.log_metrics(step=step, metrics=metrics)

    def log_config(self) -> None:
        """Emit the fully-resolved config as a structured log line.

        Converts the entire composed Hydra config to a container (resolving
        all interpolations), then logs it as a structured JSON entry via
        the experiment logger. This is useful for audit trails and debugging
        when the WandB `config=` snapshot (set on `wandb.init`) is not
        sufficient, or for console/file-only logging without WandB.

        The log entry includes the recipe name as a top-level field for easy
        filtering and correlation.

        Example:
            >>> exp.start_run(name="my_run")
            >>> exp.log_config()
            # Logs: {"level": "info", "message": "recipe_config", "recipe": "spin_fsdp", "config": {...}}
        """
        resolved = OmegaConf.to_container(self.config, resolve=True)
        self.logger.info("recipe_config", recipe=self.recipe_name, config=resolved)
