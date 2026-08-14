"""Shared fixtures for the src/ and scripts/ test suite.

These tests exercise the MLOps scaffolding (`src/`) in isolation from torch,
Ray, and the QA datasets, so they run fast and without GPUs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    output_dir = tmp_path / "runs" / "test_run"
    output_dir.mkdir(parents=True)
    return output_dir


@pytest.fixture
def base_config(tmp_output_dir: Path) -> DictConfig:
    return OmegaConf.create(
        {
            "run": {"name": "test_run", "output_dir": str(tmp_output_dir), "seed": 42},
            "experiment": {"name": "unit_test_experiment"},
            "logging": {
                "console": {"enabled": True, "level": "DEBUG", "json": False},
                "file": {"enabled": False},
                "wandb": {"enabled": False},
            },
            "memory": {
                "context_key": "context",
                "max_prompt_length": 128,
                "chunk_size": 64,
                "max_memorization_length": 32,
                "max_chunks": 2,
                "max_final_response_length": 32,
            },
            "model": {
                "checkpoint": "unit-test/checkpoint",
                "inference": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 64, "seed": None},
            },
        }
    )


@pytest.fixture
def file_logging_config(base_config: DictConfig, tmp_output_dir: Path) -> DictConfig:
    cfg = OmegaConf.merge(
        base_config,
        {
            "logging": {
                "console": {"enabled": True, "level": "DEBUG", "json": False},
                "file": {
                    "enabled": True,
                    "level": "DEBUG",
                    "json": True,
                    "path": str(tmp_output_dir / "logs" / "test_run.jsonl"),
                    "rotate_mb": 10,
                    "backup_count": 1,
                },
                "wandb": {"enabled": False},
            }
        },
    )
    return cfg
