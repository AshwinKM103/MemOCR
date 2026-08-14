"""Tests for the VERL-Hydra integration: config composition (MemOCR bridge +
VERL native schema), WandB tag/rank gating, multi-GPU metric aggregation
(non-distributed passthrough), and checkpoint metadata.

No torch/Ray/GPU dependency: `src.verl_logging` degrades to single-process
behavior when `torch.distributed` is unavailable or uninitialized, which is
exactly the path these tests exercise (mirrors how CI runs it).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from scripts.train_verl import build_native_config, validate_verl_config
from src.experiment import ExperimentError
from src.verl_logging import (
    VERLExperimentManager,
    aggregate_metrics_across_ranks,
    estimate_flops_per_second,
    get_rank,
    get_world_size,
    is_rank_zero,
)

MEMOCR_ROOT = Path(__file__).resolve().parent.parent
VERL_CONFIG_DIR = str(MEMOCR_ROOT / "config")


class TestVerlConfigComposition:
    """Test Hydra config composition for VERL trainer integration.

    Hydra's `compose()` function validates interpolations and config group
    references at composition time, so these tests catch configuration errors
    early rather than at training time. Covers trainer-type selection (sft/ppo/rl),
    native config bridge merging, and backward compatibility.
    """

    @pytest.mark.parametrize("trainer_type", ["sft", "ppo", "rl"])
    def test_verl_group_composes_for_each_trainer_type(self, trainer_type: str) -> None:
        """Verify verl_config composes with each trainer type."""
        with initialize_config_dir(version_base=None, config_dir=VERL_CONFIG_DIR):
            cfg = compose(config_name="verl_config", overrides=[f"verl={trainer_type}", "experiment=baseline"])
        assert cfg.verl.trainer_type == trainer_type
        assert cfg.verl.trainer.project_name == cfg.logging.wandb.project
        assert cfg.verl.trainer.experiment_name == cfg.run.name

    def test_sft_native_config_bridge_merges_without_error(self) -> None:
        """Verify SFT bridge merges correctly and preserves VERL's native fields."""
        with initialize_config_dir(version_base=None, config_dir=VERL_CONFIG_DIR):
            cfg = compose(config_name="verl_config", overrides=["verl=sft", "experiment=baseline"])
        native = build_native_config(cfg)
        assert native.trainer.project_name == cfg.logging.wandb.project
        assert native.trainer.experiment_name == cfg.run.name
        assert native.trainer.logger == ["console", "wandb"]
        # Fields VERL's native schema owns and this bridge does not touch
        # (e.g. model/engine/optim) must survive the merge untouched.
        assert "model" in native
        assert "engine" in native

    def test_ppo_and_rl_bridge_to_the_same_native_schema(self) -> None:
        """Verify PPO and RL both use the same native ppo_trainer schema."""
        with initialize_config_dir(version_base=None, config_dir=VERL_CONFIG_DIR):
            ppo_cfg = compose(config_name="verl_config", overrides=["verl=ppo", "experiment=baseline"])
            rl_cfg = compose(config_name="verl_config", overrides=["verl=rl", "experiment=baseline"])
        assert ppo_cfg.verl.native_config_name == rl_cfg.verl.native_config_name == "ppo_trainer"

    def test_validate_verl_config_raises_on_unknown_trainer_type(self) -> None:
        """Verify validation rejects unknown trainer types."""
        cfg = OmegaConf.create({"verl": {"trainer_type": "not_a_real_trainer"}})
        with pytest.raises(ExperimentError):
            validate_verl_config(cfg)

    def test_validate_verl_config_raises_on_missing_native_config(self) -> None:
        """Verify validation rejects missing native config files."""
        cfg = OmegaConf.create({"verl": {"trainer_type": "sft", "native_config_name": "does_not_exist"}})
        with pytest.raises(ExperimentError):
            validate_verl_config(cfg)

    def test_backward_compatible_config_yaml_unaffected(self) -> None:
        """Verify memory_train.py config composes without verl group."""
        """`config/config.yaml` (memory_train.py) must compose without the
        `verl` group ever being selected — this bridge is additive.
        """
        with initialize_config_dir(version_base=None, config_dir=str(MEMOCR_ROOT / "config")):
            cfg = compose(config_name="config", overrides=["experiment=baseline"])
        assert "verl" not in cfg


class TestRankHelpers:
    """Test distributed rank helpers and metric aggregation.

    CI and this test suite run single-process (non-distributed). torch.distributed
    is either absent or uninitialized, so every helper must degrade gracefully
    to rank 0 / world size 1.
    """

    def test_get_rank_defaults_to_zero(self) -> None:
        """Verify get_rank() returns 0 when torch.distributed is unavailable."""
        assert get_rank() == 0

    def test_get_world_size_defaults_to_one(self) -> None:
        """Verify get_world_size() returns 1 when torch.distributed is unavailable."""
        assert get_world_size() == 1

    def test_is_rank_zero_true_when_not_distributed(self) -> None:
        """Verify is_rank_zero() returns True in non-distributed context."""
        assert is_rank_zero() is True

    def test_aggregate_metrics_passthrough_when_not_distributed(self) -> None:
        """Verify aggregation passes through metrics unchanged when single-process."""
        local = {"loss": 0.5, "throughput/tokens_per_sec": 1200.0}
        assert aggregate_metrics_across_ranks(local) == local

    def test_aggregate_metrics_empty_input_returns_empty(self) -> None:
        """Verify aggregation handles empty input gracefully."""
        assert aggregate_metrics_across_ranks({}) == {}


class TestEstimateFlopsPerSecond:
    """Test FLOPs/sec estimation utility."""

    def test_computes_flops_over_elapsed_time(self) -> None:
        """Verify FLOPs/sec is computed correctly."""
        assert estimate_flops_per_second(estimated_flops=1e12, elapsed_seconds=2.0) == 5e11

    def test_raises_on_non_positive_elapsed_time(self) -> None:
        """Verify estimation rejects non-positive elapsed time."""
        with pytest.raises(ValueError):
            estimate_flops_per_second(estimated_flops=1e12, elapsed_seconds=0.0)


class TestVERLExperimentManager:
    """Test VERLExperimentManager initialization, config tagging, and logging."""

    def test_rejects_unknown_trainer_type(self, base_config: DictConfig) -> None:
        """Verify manager rejects unknown trainer types."""
        with pytest.raises(ValueError):
            VERLExperimentManager(config=base_config, trainer_type="not_a_trainer")

    def test_trainer_tag_merged_into_wandb_tags(self, base_config: DictConfig) -> None:
        """Verify trainer type is merged into WandB tags."""
        cfg = OmegaConf.merge(base_config, {"logging": {"wandb": {"enabled": False, "tags": []}}})
        exp = VERLExperimentManager(config=cfg, trainer_type="sft")
        assert "verl:sft" in exp.config.logging.wandb.tags
        assert exp.trainer_type == "sft"

    def test_manager_without_wandb_config_skips_tag_merge(self, base_config: DictConfig) -> None:
        """Verify manager handles missing WandB config gracefully."""
        cfg = OmegaConf.create(OmegaConf.to_container(base_config, resolve=False))
        del cfg["logging"]["wandb"]
        exp = VERLExperimentManager(config=cfg, trainer_type="ppo")
        exp.start_run(name="run1")
        exp.end_run()

    def test_log_training_step_records_throughput_and_memory_metrics(self, base_config: DictConfig) -> None:
        """Verify training step logging includes throughput and memory metrics."""
        exp = VERLExperimentManager(config=base_config, trainer_type="sft")
        exp.start_run(name="run1")
        exp.log_training_step(
            step=1,
            metrics={"loss": 0.3},
            tokens_per_sec=1000.0,
            samples_per_sec=8.0,
            peak_memory_gb=12.5,
            reserved_memory_gb=16.0,
        )
        summary = exp.end_run()
        assert summary["loss_last"] == 0.3
        assert summary["throughput/tokens_per_sec_last"] == 1000.0
        assert summary["memory/peak_gb_last"] == 12.5

    def test_log_metrics_aggregated_returns_dict_and_logs(self, base_config: DictConfig) -> None:
        """Verify aggregated metrics are returned and logged correctly."""
        exp = VERLExperimentManager(config=base_config, trainer_type="ppo")
        exp.start_run(name="run1")
        aggregated = exp.log_metrics_aggregated(step=1, local_metrics={"reward": 0.9})
        assert aggregated == {"reward": 0.9}
        summary = exp.end_run()
        assert summary["reward_last"] == 0.9

    def test_log_checkpoint_metadata_writes_sidecar_json(self, base_config: DictConfig, tmp_output_dir: Path) -> None:
        """Verify checkpoint metadata is written as verl_metadata.json sidecar."""
        exp = VERLExperimentManager(config=base_config, trainer_type="sft")
        exp.start_run(name="run1")
        checkpoint_dir = tmp_output_dir / "checkpoints" / "step_00000010"
        metadata_path = exp.log_checkpoint_metadata(
            step=10,
            checkpoint_dir=checkpoint_dir,
            world_size=8,
            sharding_strategy="FULL_SHARD",
            metrics={"val_loss": 0.21},
        )
        assert metadata_path.exists()
        payload = json.loads(metadata_path.read_text())
        assert payload["step"] == 10
        assert payload["world_size"] == 8
        assert payload["sharding_strategy"] == "FULL_SHARD"
        assert payload["metrics"] == {"val_loss": 0.21}
        exp.end_run()

    def test_log_config_emits_structured_log_without_error(self, base_config: DictConfig) -> None:
        """Verify config logging does not raise errors."""
        exp = VERLExperimentManager(config=base_config, trainer_type="rl")
        exp.start_run(name="run1")
        exp.log_config()  # no assertion needed beyond "does not raise"
        exp.end_run()
