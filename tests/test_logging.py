"""Tests for src/logging.py: structured console/file logging, WandB mocking."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig, OmegaConf

from src.logging import (
    ExperimentLogger,
    JsonFormatter,
    LoggingSettings,
    experiment_run,
    get_logger,
    setup_experiment_logging,
)


class TestLoggingSettings:
    def test_from_config_uses_defaults_when_logging_group_absent(self) -> None:
        cfg = OmegaConf.create({})
        settings = LoggingSettings.from_config(cfg)
        assert settings.console_enabled is True
        assert settings.wandb_enabled is False

    def test_from_config_reads_all_sinks(self, file_logging_config: DictConfig) -> None:
        settings = LoggingSettings.from_config(file_logging_config)
        assert settings.file_enabled is True
        assert settings.file_path.endswith("test_run.jsonl")
        assert settings.wandb_enabled is False


class TestJsonFormatter:
    def test_format_includes_required_fields(self) -> None:
        formatter = JsonFormatter(service="memocr")
        record = logging.LogRecord(
            name="memocr.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="something happened",
            args=(),
            exc_info=None,
        )
        record.request_id = "req-123"
        payload = json.loads(formatter.format(record))
        assert payload["level"] == "INFO"
        assert payload["service"] == "memocr"
        assert payload["requestId"] == "req-123"
        assert payload["message"] == "something happened"
        assert "timestamp" in payload

    def test_format_merges_extra_context(self) -> None:
        formatter = JsonFormatter(service="memocr")
        record = logging.LogRecord(
            name="memocr.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="step done",
            args=(),
            exc_info=None,
        )
        record.step = 3
        record.loss = 0.42
        payload = json.loads(formatter.format(record))
        assert payload["step"] == 3
        assert payload["loss"] == 0.42

    def test_format_handles_non_json_safe_values(self) -> None:
        formatter = JsonFormatter(service="memocr")
        record = logging.LogRecord(
            name="memocr.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="weird payload",
            args=(),
            exc_info=None,
        )
        record.weird = object()
        payload = json.loads(formatter.format(record))
        assert isinstance(payload["weird"], str)


class TestSetupExperimentLogging:
    def test_console_only_creates_logger_with_one_handler(self, base_config: DictConfig) -> None:
        logger = setup_experiment_logging(base_config, "exp", "run_console_only")
        assert isinstance(logger, ExperimentLogger)
        underlying = logging.getLogger("memocr.exp.run_console_only")
        assert len(underlying.handlers) == 1

    def test_file_sink_writes_json_lines(self, file_logging_config: DictConfig, tmp_output_dir: Path) -> None:
        logger = setup_experiment_logging(file_logging_config, "exp", "run_file_sink")
        logger.info("training_step", step=1, loss=0.5)

        log_file = tmp_output_dir / "logs" / "test_run.jsonl"
        assert log_file.exists()
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["message"] == "training_step"
        assert payload["step"] == 1
        assert payload["loss"] == 0.5

    def test_repeated_setup_does_not_duplicate_handlers(self, base_config: DictConfig) -> None:
        setup_experiment_logging(base_config, "exp", "run_dedup")
        setup_experiment_logging(base_config, "exp", "run_dedup")
        underlying = logging.getLogger("memocr.exp.run_dedup")
        assert len(underlying.handlers) == 1

    def test_wandb_disabled_skips_wandb_init(self, base_config: DictConfig) -> None:
        with patch("src.logging.wandb") as mock_wandb:
            setup_experiment_logging(base_config, "exp", "run_no_wandb")
            mock_wandb.init.assert_not_called()

    def test_wandb_enabled_calls_init_and_logs_metrics(self, base_config: DictConfig) -> None:
        cfg = OmegaConf.merge(base_config, {"logging": {"wandb": {"enabled": True, "project": "memocr-test"}}})
        mock_run = MagicMock()
        with patch("src.logging.wandb") as mock_wandb:
            mock_wandb.init.return_value = mock_run
            logger = setup_experiment_logging(cfg, "exp", "run_wandb")
            logger.info("step", step=1, loss=0.3)
            mock_wandb.init.assert_called_once()
            mock_run.log.assert_called_once()


class TestExperimentRunContextManager:
    def test_context_manager_logs_start_and_end(self, base_config: DictConfig) -> None:
        with experiment_run(base_config, "exp", "run_ctx") as logger:
            logger.info("mid_run_event")
        # no exception means logging + WandB finish path completed cleanly

    def test_context_manager_reraises_and_logs_exception(self, base_config: DictConfig) -> None:
        with pytest.raises(ValueError):
            with experiment_run(base_config, "exp", "run_ctx_fail"):
                raise ValueError("boom")


def test_get_logger_returns_stdlib_logger() -> None:
    logger = get_logger("memocr.unit_test")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "memocr.unit_test"
