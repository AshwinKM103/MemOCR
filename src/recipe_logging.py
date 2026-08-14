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
    """`ExperimentManager` with recipe-aware WandB tags and an
    `AgentLoopOutput`-shaped metrics ingestion path.

    Tags are computed once at construction (config is immutable for the life
    of a run) and merged into `logging.wandb.tags` before `start_run` creates
    the WandB run, so every recipe run is filterable by recipe name, memory
    density, and token budget in the WandB UI without extra dashboard setup.
    """

    def __init__(self, config: DictConfig) -> None:
        merged_config = self._with_recipe_tags(config)
        super().__init__(config=merged_config)
        self.recipe_name: str = OmegaConf.select(merged_config, "recipe.name", default="unknown_recipe")

    @staticmethod
    def _with_recipe_tags(config: DictConfig) -> DictConfig:
        tags = resolve_recipe_tags(config)
        if OmegaConf.select(config, "logging.wandb") is None:
            return config
        merged = OmegaConf.merge(config, OmegaConf.create({"logging": {"wandb": {"tags": tags}}}))
        assert isinstance(merged, DictConfig)
        return merged

    def log_agent_loop_output(self, step: int, output: Any) -> None:
        """Log an `AgentLoopOutput` (or dict with a `metrics` key) as a
        metrics dict with `agent/`-prefixed field names.
        """
        metrics = agent_loop_output_to_metrics(output)
        if metrics:
            self.log_metrics(step=step, metrics=metrics)

    def log_config(self) -> None:
        """Emit the fully-resolved config as a single structured log line,
        for runs where the WandB `config=` snapshot (set on `wandb.init`) is
        not sufficient, e.g. console/file-only logging.
        """
        resolved = OmegaConf.to_container(self.config, resolve=True)
        self.logger.info("recipe_config", recipe=self.recipe_name, config=resolved)
