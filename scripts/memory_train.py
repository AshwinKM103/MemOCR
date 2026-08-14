#!/usr/bin/env python
"""Train a memory agent with a Hydra-composed config.

Usage:
    python scripts/memory_train.py +experiment=baseline
    python scripts/memory_train.py +experiment=baseline memory.chunk_size=512

This script wires together config loading, structured logging, checkpointing
via `ExperimentManager`, and a training loop over `LLMGenerationManager`. It
does not implement the PPO update itself (that lives in
`verl/trainer/ppo/ray_trainer.py`); it demonstrates the MLOps scaffolding
(config validation, logging, checkpoints, metric aggregation) a full
training entrypoint would sit inside.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment import ExperimentError, ExperimentManager  # noqa: E402
from src.generation import GenerationConfig  # noqa: E402

CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "config")


def validate_config(cfg: DictConfig) -> GenerationConfig:
    """Fail fast on a malformed config before touching GPUs.

    Raises pydantic.ValidationError if `model.inference.*` is out of range,
    and ExperimentError if required experiment fields are missing.
    """
    required = ["experiment.name", "experiment.dataset", "memory.chunk_size", "model.checkpoint"]
    missing = [key for key in required if OmegaConf.select(cfg, key) is None]
    if missing:
        raise ExperimentError(f"Missing required config keys: {missing}")

    inference_cfg = OmegaConf.select(cfg, "model.inference", default={})
    return GenerationConfig(
        temperature=inference_cfg.get("temperature", 0.7),
        top_p=inference_cfg.get("top_p", 0.9),
        max_tokens=inference_cfg.get("max_tokens", 512),
        seed=inference_cfg.get("seed"),
    )


def build_dataset(cfg: DictConfig):
    """Construct the training dataset from `experiment.data_files.train`.

    Deferred import: `recurrent.impls.memory` pulls in torch/transformers,
    which we want to avoid loading just to validate a config (e.g. in tests).
    """
    from transformers import AutoTokenizer

    from recurrent.impls.memory import MemoryConfig, MemoryDataset

    memory_cfg = MemoryConfig(
        context_key=cfg.memory.context_key,
        max_prompt_length=cfg.memory.max_prompt_length,
        chunk_size=cfg.memory.chunk_size,
        max_memorization_length=cfg.memory.max_memorization_length,
        max_chunks=cfg.memory.max_chunks,
        max_final_response_length=cfg.memory.max_final_response_length,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.checkpoint)
    data_config = OmegaConf.create({"truncation": "middle", "prompt_key": "prompt"})
    return MemoryDataset(
        recurrent_config=memory_cfg,
        data_files=cfg.experiment.data_files.train,
        tokenizer=tokenizer,
        data_config=data_config,
    )


def train_loop(cfg: DictConfig, exp: ExperimentManager, generation_config: GenerationConfig) -> dict[str, Any]:
    """Minimal training loop skeleton: iterates epochs/steps, checkpoints,
    and logs metrics. The actual policy-optimization step is delegated to
    `verl.trainer.ppo.ray_trainer.RayPPOTrainer`, which is out of scope for
    this MLOps scaffolding pass.
    """
    total_epochs = cfg.experiment.train.total_epochs
    save_freq = cfg.experiment.train.save_freq
    step = 0

    for epoch in range(total_epochs):
        exp.logger.info("epoch_start", epoch=epoch, total_epochs=total_epochs)
        # Placeholder for the real per-batch loop; a full implementation
        # would iterate the DataLoader built from `build_dataset(cfg)` and
        # call `LLMGenerationManager.run_llm_loop` + a PPO update per batch.
        step += 1
        exp.log_metrics(step=step, metrics={"epoch": epoch, "generation/max_tokens": generation_config.max_tokens})

        if step % save_freq == 0:
            exp.save_checkpoint(step=step, metrics={"epoch": epoch})

    return exp.end_run()


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    generation_config = validate_config(cfg)
    exp = ExperimentManager(config=cfg)
    exp.start_run(name=cfg.run.name)
    try:
        summary = train_loop(cfg, exp, generation_config)
        exp.logger.info("train_complete", **summary)
    except Exception:
        exp.end_run(status="failed")
        raise


if __name__ == "__main__":
    main()
