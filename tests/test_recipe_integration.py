"""Tests for the recipe Hydra/WandB integration: config composition,
recipe-aware WandB tags, agent-loop metric flattening, and the sweep-trial
override grid built by scripts/sweep_recipe.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from scripts.sweep_recipe import build_trial_overrides
from src.recipe_logging import (
    RecipeExperimentManager,
    agent_loop_output_to_metrics,
    resolve_recipe_tags,
)

CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "config")


class TestRecipeConfigComposition:
    """Hydra `compose()` calls double as config validation: an unresolvable
    interpolation or missing config group raises here, not at training time.
    """

    def test_recipe_config_composes_with_spin(self) -> None:
        with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
            cfg = compose(config_name="recipe_config", overrides=["recipe=spin", "experiment=baseline"])
        assert cfg.recipe.name == "spin_fsdp"
        assert cfg.recipe.launch.command == "bash recipe/spin/run_spin.sh"
        assert "spin" in cfg.recipe.tags

    def test_recipe_config_composes_with_langgraph_agent(self) -> None:
        with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
            cfg = compose(config_name="recipe_config", overrides=["recipe=langgraph_agent", "experiment=baseline"])
        assert cfg.recipe.name == "langgraph_agent"
        assert cfg.recipe.module == "recipe.langgraph_agent.react_agent_loop"
        assert cfg.recipe.launch.command is None

    def test_default_config_yaml_unaffected_by_recipe_group(self) -> None:
        """scripts/memory_train.py's `config.yaml` must keep composing without
        a `recipe` key -- this is the backward-compatibility guarantee.
        """
        with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
            cfg = compose(config_name="config", overrides=["experiment=baseline"])
        assert "recipe" not in cfg

    def test_memory_ablation_sweep_grid_composes(self) -> None:
        with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
            cfg = compose(
                config_name="recipe_config",
                overrides=["recipe=memory_ablation", "experiment=ablation_density"],
            )
        assert cfg.recipe.sweep.recipe == ["spin", "langgraph_agent"]
        assert cfg.recipe.sweep.memory_ablation == ["low", "medium", "high"]


class TestResolveRecipeTags:
    def test_tags_include_experiment_recipe_density_and_budget(self) -> None:
        cfg = OmegaConf.create(
            {
                "experiment": {"name": "baseline_hotpotqa"},
                "recipe": {"name": "spin_fsdp", "tags": ["spin", "fsdp"]},
                "memory": {
                    "density": {"target_tokens_per_1k_context": 64},
                    "max_memorization_length": 256,
                },
            }
        )
        tags = resolve_recipe_tags(cfg)
        assert tags == [
            "baseline_hotpotqa",
            "recipe:spin_fsdp",
            "spin",
            "fsdp",
            "memory_density:64",
            "token_budget:256",
        ]

    def test_missing_fields_are_skipped_not_errored(self) -> None:
        cfg = OmegaConf.create({"experiment": {"name": "solo"}})
        assert resolve_recipe_tags(cfg) == ["solo"]

    def test_duplicate_tags_are_deduplicated(self) -> None:
        cfg = OmegaConf.create({"experiment": {"name": "x"}, "recipe": {"tags": ["x"]}})
        assert resolve_recipe_tags(cfg) == ["x"]


class TestAgentLoopOutputToMetrics:
    def test_flattens_metrics_dict_with_agent_prefix(self) -> None:
        output = {"metrics": {"reward": 0.8, "turns": 3, "tool_name": "search"}}
        metrics = agent_loop_output_to_metrics(output)
        assert metrics == {"agent/reward": 0.8, "agent/turns": 3.0}

    def test_non_scalar_values_are_dropped(self) -> None:
        output = {"metrics": {"reward": 0.5, "trace": ["a", "b"]}}
        metrics = agent_loop_output_to_metrics(output)
        assert metrics == {"agent/reward": 0.5}

    def test_empty_metrics_returns_empty_dict(self) -> None:
        assert agent_loop_output_to_metrics({"metrics": {}}) == {}


class TestRecipeExperimentManager:
    def test_start_run_tags_wandb_config_with_recipe_metadata(
        self, base_config: DictConfig, tmp_output_dir: Path
    ) -> None:
        cfg = OmegaConf.merge(
            base_config,
            {
                "recipe": {"name": "spin_fsdp", "tags": ["spin"]},
                "logging": {"wandb": {"enabled": False, "tags": []}},
            },
        )
        exp = RecipeExperimentManager(config=cfg)
        assert exp.recipe_name == "spin_fsdp"
        # base_config fixture sets memory.max_memorization_length=32, so a
        # token_budget tag is expected alongside the recipe/experiment tags.
        assert exp.config.logging.wandb.tags == [
            "unit_test_experiment",
            "recipe:spin_fsdp",
            "spin",
            "token_budget:32",
        ]
        exp.start_run(name="run1")
        exp.end_run()

    def test_log_agent_loop_output_forwards_to_metric_aggregator(self, base_config: DictConfig) -> None:
        cfg = OmegaConf.merge(base_config, {"recipe": {"name": "langgraph_agent"}})
        exp = RecipeExperimentManager(config=cfg)
        exp.start_run(name="run1")
        exp.log_agent_loop_output(step=1, output={"metrics": {"reward": 1.0}})
        summary = exp.end_run()
        assert summary["agent/reward_last"] == 1.0

    def test_manager_without_wandb_config_skips_tag_merge(self, base_config: DictConfig) -> None:
        cfg = OmegaConf.create(OmegaConf.to_container(base_config, resolve=False))
        del cfg["logging"]["wandb"]
        cfg["recipe"] = {"name": "spin_fsdp"}
        exp = RecipeExperimentManager(config=cfg)
        exp.start_run(name="run1")
        exp.end_run()


class TestSweepTrialOverrides:
    def test_builds_cartesian_product_of_recipe_and_ablation(self) -> None:
        cfg = OmegaConf.create(
            {"recipe": {"sweep": {"recipe": ["spin", "langgraph_agent"], "memory_ablation": ["low", "high"]}}}
        )
        overrides = build_trial_overrides(cfg)
        assert overrides == [
            ["recipe=spin", "+memory/ablations=low"],
            ["recipe=spin", "+memory/ablations=high"],
            ["recipe=langgraph_agent", "+memory/ablations=low"],
            ["recipe=langgraph_agent", "+memory/ablations=high"],
        ]

    def test_missing_sweep_config_raises(self) -> None:
        with pytest.raises(ValueError):
            build_trial_overrides(OmegaConf.create({}))
