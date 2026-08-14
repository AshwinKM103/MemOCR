"""Tests for src/experiment.py: config-driven lifecycle, checkpoints, registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import DictConfig

from src.experiment import ExperimentError, ExperimentManager, MetricAggregator


class TestMetricAggregator:
    def test_summary_computes_mean_min_max_last(self) -> None:
        agg = MetricAggregator()
        agg.add({"em": 0.5})
        agg.add({"em": 0.7})
        agg.add({"em": 0.9})
        summary = agg.summary()
        assert summary["em_mean"] == pytest.approx(0.7)
        assert summary["em_min"] == 0.5
        assert summary["em_max"] == 0.9
        assert summary["em_last"] == 0.9

    def test_summary_empty_returns_empty_dict(self) -> None:
        assert MetricAggregator().summary() == {}

    def test_latest_returns_most_recent_value_per_key(self) -> None:
        agg = MetricAggregator()
        agg.add({"loss": 1.0, "acc": 0.1})
        agg.add({"loss": 0.5})
        latest = agg.latest()
        assert latest["loss"] == 0.5
        assert latest["acc"] == 0.1


class TestExperimentManagerLifecycle:
    def test_start_run_creates_output_dirs(self, base_config: DictConfig) -> None:
        exp = ExperimentManager(config=base_config)
        exp.start_run(name="run1")
        assert exp.run_dir.exists()
        assert (exp.run_dir / "checkpoints").exists()
        exp.end_run()

    def test_double_start_run_raises(self, base_config: DictConfig) -> None:
        exp = ExperimentManager(config=base_config)
        exp.start_run(name="run1")
        with pytest.raises(ExperimentError):
            exp.start_run(name="run2")
        exp.end_run()

    def test_end_run_without_start_raises(self, base_config: DictConfig) -> None:
        exp = ExperimentManager(config=base_config)
        with pytest.raises(ExperimentError):
            exp.end_run()

    def test_logger_before_start_run_raises(self, base_config: DictConfig) -> None:
        exp = ExperimentManager(config=base_config)
        with pytest.raises(ExperimentError):
            _ = exp.logger

    def test_end_run_returns_metric_summary(self, base_config: DictConfig) -> None:
        exp = ExperimentManager(config=base_config)
        exp.start_run(name="run1")
        exp.log_metrics(step=1, metrics={"em": 0.6})
        exp.log_metrics(step=2, metrics={"em": 0.8})
        summary = exp.end_run()
        assert summary["em_mean"] == pytest.approx(0.7)

    def test_registry_appends_one_line_per_run(self, base_config: DictConfig) -> None:
        exp = ExperimentManager(config=base_config)
        exp.start_run(name="run1")
        exp.end_run()
        exp.start_run(name="run2")
        exp.end_run()

        lines = exp.registry_path.read_text().strip().splitlines()
        # start_run + end_run each write a registry record -> 4 lines total
        assert len(lines) == 4
        statuses = [json.loads(line)["status"] for line in lines]
        assert statuses == ["running", "completed", "running", "completed"]


class TestCheckpoints:
    def test_save_checkpoint_writes_metadata(self, base_config: DictConfig) -> None:
        exp = ExperimentManager(config=base_config)
        exp.start_run(name="run1")
        ckpt_dir = exp.save_checkpoint(step=10, metrics={"em": 0.55})
        metadata_path = ckpt_dir / "metadata.json"
        assert metadata_path.exists()

        metadata = json.loads(metadata_path.read_text())
        assert metadata["step"] == 10
        assert metadata["metrics"] == {"em": 0.55}
        assert metadata["run_name"] == "run1"
        exp.end_run()

    def test_save_checkpoint_copies_state_files(self, base_config: DictConfig, tmp_path: Path) -> None:
        exp = ExperimentManager(config=base_config)
        exp.start_run(name="run1")

        weights_file = tmp_path / "model.bin"
        weights_file.write_bytes(b"fake-weights")

        ckpt_dir = exp.save_checkpoint(step=5, state={"model.bin": weights_file})
        assert (ckpt_dir / "model.bin").read_bytes() == b"fake-weights"
        exp.end_run()

    def test_load_checkpoint_round_trips_metadata(self, base_config: DictConfig) -> None:
        exp = ExperimentManager(config=base_config)
        exp.start_run(name="run1")
        ckpt_dir = exp.save_checkpoint(step=20, metrics={"f1": 0.72})
        exp.end_run()

        loaded = exp.load_checkpoint(ckpt_dir)
        assert loaded.step == 20
        assert loaded.metrics == {"f1": 0.72}

    def test_load_checkpoint_missing_raises(self, base_config: DictConfig, tmp_path: Path) -> None:
        exp = ExperimentManager(config=base_config)
        with pytest.raises(FileNotFoundError):
            exp.load_checkpoint(tmp_path / "does_not_exist")

    def test_rollback_restores_previous_checkpoint_metadata(self, base_config: DictConfig) -> None:
        """Simulates the rollback procedure: load an older checkpoint after a newer one exists."""
        exp = ExperimentManager(config=base_config)
        exp.start_run(name="run1")
        old_ckpt = exp.save_checkpoint(step=1, metrics={"em": 0.4})
        new_ckpt = exp.save_checkpoint(step=2, metrics={"em": 0.3})  # regression
        exp.end_run()

        rolled_back = exp.load_checkpoint(old_ckpt)
        assert rolled_back.step == 1
        assert rolled_back.metrics["em"] == 0.4
        assert new_ckpt != old_ckpt
