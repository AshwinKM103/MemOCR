# VERL-Hydra Integration Guide

Bridges the Phase 1/2 MLOps infrastructure (`config/`, `src/experiment.py`,
`src/logging.py`, `src/recipe_logging.py`) into VERL's trainer system, so
VERL runs get the same run registry, checkpoint metadata, and WandB tagging
conventions as `scripts/memory_train.py` and `scripts/train_recipe.py`.

## Design

VERL trainers already read a `default_backend`/`logger` list
(`verl/utils/tracking.py: Tracking`) that includes a native `"wandb"`
backend — `SFTTrainer.fit()` and `RayPPOTrainer` drive that `Tracking`
instance directly from `config.trainer.logger`/`project_name`/
`experiment_name`. This means the integration does **not** need to
subclass or monkeypatch VERL trainers to get WandB output; it only needs to
put the right values into VERL's own config namespace before the trainer is
constructed.

Two config trees compose independently and are merged by
`scripts/train_verl.py`:

1. **MemOCR's own tree** (`config/verl_config.yaml` + `config/verl/{sft,ppo,rl}.yaml`,
   reusing `config/experiment/`, `config/logging/`, `config/memory/`,
   `config/model/` unchanged) — the Hydra-composed side, selected with
   `verl=sft|ppo|rl` on the CLI.
2. **VERL's native tree** (`verl/trainer/config/sft_trainer_engine.yaml` /
   `ppo_trainer.yaml`, with their own `model@model`, `engine@engine`,
   `optim@optim` sub-group defaults) — composed via a nested
   `hydra.compose()` call, unmodified.

`scripts/train_verl.py:build_native_config()` composes (2) in full (not a
plain `OmegaConf.load` — VERL's native trainer configs are themselves Hydra
entrypoints with their own `defaults:` list, so a plain YAML load would
leave `model`/`engine`/`optim` unexpanded), then merges an explicit
allowlist of bridge fields from (1) onto it:

| MemOCR field (`cfg.verl.trainer.*`) | VERL native field           |
| ----------------------------------- | --------------------------- |
| `project_name`                      | `trainer.project_name`      |
| `experiment_name`                   | `trainer.experiment_name`   |
| `logger`                            | `trainer.logger`            |
| `default_local_dir`                 | `trainer.default_local_dir` |
| `total_epochs`                      | `trainer.total_epochs`      |
| `save_freq`                         | `trainer.save_freq`         |
| `test_freq`                         | `trainer.test_freq`         |
| `data.train_batch_size`             | `data.train_batch_size`     |

Everything else in VERL's native config (`model`, `engine`, `optim`,
`checkpoint`, `data.*` beyond batch size) is untouched — override it the
normal VERL way (CLI overrides against the native config, or edit the
`native_config_name` yaml) if needed. No file under `verl/` is modified by
this integration.

## Files

- `config/verl_config.yaml` — root Hydra config for `scripts/train_verl.py`.
- `config/verl/{sft,ppo,rl}.yaml` — trainer-specific bridge fields
  (`@package verl`). `rl.yaml` currently resolves to the same native
  `ppo_trainer` config as `ppo.yaml`: this vendored VERL checkout does not
  ship a distinct `rl_trainer.py` — `RayPPOTrainer` (parameterized by
  `config.algorithm.*`) is VERL's RL trainer.
- `src/verl_logging.py` — `VERLExperimentManager(ExperimentManager)`, rank
  helpers (`get_rank`, `get_world_size`, `is_rank_zero`),
  `aggregate_metrics_across_ranks`, `estimate_flops_per_second`.
- `scripts/train_verl.py` — Hydra entrypoint; validates config, builds the
  merged native config, starts/ends a `VERLExperimentManager` run (rank-0
  gated), dispatches to `verl.trainer.sft_trainer.run_sft` or
  `verl.trainer.main_ppo.run_ppo`.
- `scripts/benchmark_verl.py` — throughput (tokens/sec, samples/sec) and
  peak/reserved GPU memory profiling, logged via `VERLExperimentManager`.
- `tests/test_verl_integration.py` — config composition, rank/aggregation
  helpers, `VERLExperimentManager` behavior, checkpoint metadata.

## CLI examples

```bash
# SFT: print the resolved MemOCR + VERL native config, do not train
python scripts/train_verl.py verl=sft experiment=baseline dry_run=true

# SFT: actual run (requires torch.distributed init; run under torchrun)
torchrun --nproc_per_node=8 scripts/train_verl.py verl=sft experiment=baseline

# PPO
torchrun --nproc_per_node=8 scripts/train_verl.py verl=ppo experiment=baseline

# Inspect only the composed MemOCR-side config (no VERL native compose)
python scripts/train_verl.py --cfg job --resolve verl=sft experiment=baseline

# Benchmark throughput/memory (no real model needed by default)
python scripts/benchmark_verl.py verl=sft experiment=baseline \
    benchmark.num_steps=50 benchmark.batch_size=8 benchmark.seq_len=2048
```

Large artifacts (`env.paths.output_root`, `checkpoint_root`) default to
`/mnt/ssd/users/durgesh/memocr/...` per the repo's storage invariants;
override with `MEMOCR_OUTPUT_ROOT`/`MEMOCR_CHECKPOINT_ROOT` env vars for
local testing (see `config/env.yaml`, `.env.example`).

## Multi-GPU / rank coordination

- `src/verl_logging.get_rank()` / `is_rank_zero()` wrap
  `torch.distributed`, degrading to rank 0 / world size 1 when torch or
  distributed init is unavailable (CPU-only tests, single-process runs).
- `VERLExperimentManager.log_training_step`, `.log_checkpoint_metadata`,
  and `.log_config` all no-op on non-zero ranks — safe to call
  unconditionally from wrapper code without an `if is_rank_zero()` guard at
  every call site.
- `aggregate_metrics_across_ranks` averages scalar metrics with
  `torch.distributed.all_reduce(..., op=ReduceOp.AVG)` when distributed,
  and passes values through unchanged otherwise (used by
  `VERLExperimentManager.log_metrics_aggregated`, and available directly
  for `scripts/benchmark_verl.py`-style per-rank sampling).
- `VERLExperimentManager.log_checkpoint_metadata` writes a
  `verl_metadata.json` sidecar (world_size, sharding_strategy, step,
  metrics) next to VERL's own `CheckpointHandler`-managed checkpoint
  directory. It does not manage the checkpoint files themselves — VERL's
  checkpoint handler already does that.

## Known limitations

- PPO/RL `dry_run` compose currently fails in this vendored VERL checkout:
  `verl/trainer/config/ppo_trainer.yaml` declares a `data@data: legacy_data`
  default pointing at `verl/trainer/config/data/legacy_data.yaml`, which
  does not exist in this checkout (pre-existing gap in the vendored VERL
  tree, unrelated to this bridge — confirmed via `find verl/trainer/config
-iname legacy_data*` returning nothing). SFT composes and dry-runs
  cleanly; PPO/RL will need that file restored (or `native_config_name`
  repointed at a working PPO schema) before `verl=ppo|rl` dry-run or train
  will succeed. `config/verl/ppo.yaml`/`rl.yaml` and the bridge logic are
  otherwise identical to the working SFT path and exercised by
  `tests/test_verl_integration.py::test_ppo_and_rl_bridge_to_the_same_native_schema`
  (native-config-name equality only, since full PPO compose is currently
  blocked by the missing file above).
- `run_in_process_module`-style per-recipe drivers (as in
  `scripts/train_recipe.py`) are out of scope here; VERL trainers are
  invoked directly via their existing `run_sft`/`run_ppo` functions.
