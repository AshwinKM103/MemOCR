#!/usr/bin/env python
"""Hydra entrypoint for VERL trainers (SFT, PPO, RL), bridging Phase 1/2
MLOps config groups (experiment, logging, memory, model) into VERL's native
OmegaConf trainer schema.

Composition strategy:
1. Hydra composes MemOCR's own config tree (config/verl/defaults.yaml +
   experiment/logging/memory/model groups), selecting a trainer via
   `verl=sft|ppo|rl` (config/verl/{sft,ppo,rl}.yaml).
2. VERL's *native* trainer config (verl/trainer/config/<native_config_name>.yaml)
   is loaded separately with `OmegaConf.load` and used as the base — this
   is VERL's own schema, unmodified.
3. The bridge fields from step 1 (`cfg.verl.trainer.*`, `cfg.verl.data.*`)
   are merged on top of VERL's native config via `OmegaConf.merge`, so VERL
   keeps every field it doesn't know about MemOCR (model configs, engine
   strategy, optimizer, etc.) and only the explicitly-bridged fields
   (project_name, experiment_name, logger, checkpoint dir, batch size,
   epoch/save/eval frequency) are overridden.
4. The merged native config is passed to VERL's own `run_sft` / `run_ppo`
   entrypoint unchanged — no VERL trainer code is modified or subclassed.

Usage:
    python scripts/train_verl.py verl=sft +experiment=baseline
    python scripts/train_verl.py verl=ppo +experiment=baseline
    python scripts/train_verl.py verl=sft dry_run=true
    python scripts/train_verl.py --cfg job --resolve verl=sft +experiment=baseline
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Callable

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment import ExperimentError
from src.verl_logging import VERLExperimentManager, is_rank_zero

CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "config")
CONFIG_NAME = "verl_config"
VERL_NATIVE_CONFIG_DIR = Path(__file__).resolve().parent.parent / "verl" / "trainer" / "config"

# Bridge fields lifted from our composed cfg.verl.* onto VERL's native
# config namespace. Keeping this as an explicit allowlist (rather than
# merging all of cfg.verl) means a typo in config/verl/*.yaml fails loudly
# via KeyError instead of silently injecting an unrecognized field into
# VERL's schema.
_BRIDGE_TRAINER_FIELDS = (
    "project_name",
    "experiment_name",
    "logger",
    "default_local_dir",
    "total_epochs",
    "save_freq",
    "test_freq",
)
_BRIDGE_DATA_FIELDS = ("train_batch_size",)

_ENTRYPOINTS = {
    "sft": ("verl.trainer.sft_trainer", "run_sft"),
    "ppo": ("verl.trainer.main_ppo", "run_ppo"),
    "rl": ("verl.trainer.main_ppo", "run_ppo"),
}


def validate_verl_config(cfg: DictConfig) -> None:
    """Validate that the composed config has required VERL bridge fields.

    Fails fast if trainer_type, native_config_name, or the native config file
    are missing, before importing torch/ray or touching the filesystem.

    Args:
        cfg: Hydra OmegaConf config with `verl` section.

    Raises:
        ExperimentError: If trainer_type is unknown, native_config_name is
            not set, or the native config file does not exist.
    """
    trainer_type = OmegaConf.select(cfg, "verl.trainer_type")
    if trainer_type not in _ENTRYPOINTS:
        raise ExperimentError(
            f"Unknown or missing verl.trainer_type='{trainer_type}'. Select one of {sorted(_ENTRYPOINTS)} "
            "via `verl=sft|ppo|rl` on the CLI."
        )
    native_config_name = OmegaConf.select(cfg, "verl.native_config_name")
    if not native_config_name:
        raise ExperimentError(f"verl.native_config_name is not set for trainer_type='{trainer_type}'.")
    native_path = VERL_NATIVE_CONFIG_DIR / f"{native_config_name}.yaml"
    if not native_path.exists():
        raise ExperimentError(f"VERL native config not found: {native_path}")


def _compose_native_config(native_config_name: str) -> DictConfig:
    """Fully compose VERL's native trainer config with all defaults resolved.

    VERL's trainer configs (e.g., `sft_trainer_engine.yaml`, `ppo_trainer.yaml`)
    are themselves Hydra entrypoint configs with their own `defaults:` list
    (e.g., `model@model: hf_model`). Plain `OmegaConf.load` returns the
    unexpanded defaults list, not the composed `model`/`engine`/`optim`
    sections VERL's trainers actually read. This function uses Hydra's
    `compose()` to fully resolve the config tree.

    Since `scripts/train_verl.py`'s `main()` is a `@hydra.main` entrypoint,
    `GlobalHydra` is already initialized. Hydra does not support nested
    `initialize_config_dir` contexts, so the outer Hydra state is cleared
    for this nested compose and not restored (the outer config is already
    captured at this point).

    Args:
        native_config_name: Base name of VERL's native config file (e.g.,
            "sft_trainer_engine", "ppo_trainer").

    Returns:
        The fully composed native config as a DictConfig.

    Raises:
        ConfigCompositionException: If the native config or its defaults
            cannot be found or composed.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    was_initialized = GlobalHydra.instance().is_initialized()
    if was_initialized:
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(VERL_NATIVE_CONFIG_DIR)):
        return compose(config_name=native_config_name)


def build_native_config(cfg: DictConfig) -> DictConfig:
    """Compose VERL's native trainer config and merge MemOCR bridge fields.

    Loads and fully composes VERL's native trainer config (e.g.,
    `sft_trainer_engine.yaml`), then merges an allowlist of MemOCR bridge
    fields (project_name, experiment_name, logger, batch size, etc.) on top.
    The resulting config is in VERL's own schema and ready to pass directly
    to `run_sft()` or `run_ppo()` without modification.

    Args:
        cfg: Hydra OmegaConf config with `verl` section containing
            `native_config_name` and bridge fields.

    Returns:
        A fully merged DictConfig in VERL's native schema, with MemOCR bridge
        fields overlaid on top of VERL's native defaults.

    Raises:
        ExperimentError: If the native config file does not exist.
    """
    native_config_name = cfg.verl.native_config_name
    native_path = VERL_NATIVE_CONFIG_DIR / f"{native_config_name}.yaml"
    if not native_path.exists():
        raise ExperimentError(f"VERL native config not found: {native_path}")
    native_cfg = _compose_native_config(native_config_name)

    trainer_overrides = {
        field: cfg.verl.trainer[field] for field in _BRIDGE_TRAINER_FIELDS if field in cfg.verl.trainer
    }
    data_overrides = {field: cfg.verl.data[field] for field in _BRIDGE_DATA_FIELDS if field in cfg.verl.data}

    overrides = OmegaConf.create({"trainer": trainer_overrides, "data": data_overrides})
    merged = OmegaConf.merge(native_cfg, overrides)
    assert isinstance(merged, DictConfig)
    return merged


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint: compose configs, start tracking, and dispatch to trainer.

    Validates the composed config, optionally prints a dry-run summary, then
    constructs a `VERLExperimentManager`, builds the merged native config,
    and dispatches to the selected VERL trainer entrypoint (run_sft or run_ppo).

    Only rank 0 drives `VERLExperimentManager` (registry writes, config
    logging). Every rank calls the VERL entrypoint so distributed
    initialization proceeds normally.

    Args:
        cfg: Hydra OmegaConf config with both MemOCR and VERL groups.

    Raises:
        ExperimentError: If config validation fails.
        Any exception from the VERL trainer is re-raised after cleanup.

    Example CLI usage:
        # SFT training
        torchrun --nproc_per_node=8 scripts/train_verl.py verl=sft experiment=baseline

        # PPO training
        torchrun --nproc_per_node=8 scripts/train_verl.py verl=ppo experiment=baseline

        # Dry run (print resolved configs, exit)
        python scripts/train_verl.py verl=sft experiment=baseline dry_run=true
    """
    validate_verl_config(cfg)
    if cfg.dry_run:
        if is_rank_zero():
            print("=== Resolved MemOCR Config ===")
            print(OmegaConf.to_yaml(cfg))
            native_cfg = build_native_config(cfg)
            print("=== Resolved VERL Native Config (post-bridge) ===")
            print(OmegaConf.to_yaml(native_cfg))
        return

    trainer_type = cfg.verl.trainer_type
    module_name, entrypoint_name = _ENTRYPOINTS[trainer_type]
    native_cfg = build_native_config(cfg)

    exp = VERLExperimentManager(config=cfg, trainer_type=trainer_type)
    if is_rank_zero():
        exp.start_run(name=cfg.run.name)
        exp.log_config()

    try:
        module = importlib.import_module(module_name)
        entrypoint: Callable[[DictConfig], None] = getattr(module, entrypoint_name)
        entrypoint(native_cfg)
    except Exception:
        if is_rank_zero():
            exp.end_run(status="failed")
        raise
    else:
        if is_rank_zero():
            exp.end_run(status="completed")


if __name__ == "__main__":
    main()
