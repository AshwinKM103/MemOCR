# Migration: from `scripts/train.sh`/`scripts/eval.sh` to Hydra-based scripts

The repo's existing `scripts/train.sh` and `scripts/eval.sh` invoke the
`verl` PPO trainer directly with shell-level environment variables and
inline Hydra overrides. The new `scripts/memory_{train,eval,sweep}.py`
entrypoints wrap that same underlying stack (`LLMGenerationManager`,
`RayPPOTrainer`) with the config/logging/checkpoint scaffolding under
`config/` and `src/`. Nothing under `recurrent/` or `verl/` changed
behavior — the migration is additive.

## What changed

| Before                                                                                                          | After                                                                                                                             |
| --------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Shell script sets env vars, calls `python -m verl.trainer.main_ppo ...` with inline `key=value` Hydra overrides | `python scripts/memory_train.py +experiment=<name>` composes the same overrides from versioned YAML in `config/`                  |
| No structured logging; stdout only                                                                              | `src/logging.py` — console + JSON file + optional WandB, one call site                                                            |
| No checkpoint metadata beyond what `verl`'s checkpoint manager writes                                           | `src/experiment.py::ExperimentManager` adds a `metadata.json` sidecar (git SHA, resolved config, metrics) next to each checkpoint |
| No experiment registry                                                                                          | `<output_root>/registry.jsonl` — one line per run start/end                                                                       |
| Sweeps done by hand (multiple shell invocations)                                                                | `scripts/memory_sweep.py -m +experiment=<name>` drives a `experiment.sweep` grid in parallel                                      |

## What did not change

- `recurrent/interface.py` (`RConfig`, `RAgent`, `RDataset`, `RRegister`) —
  unchanged; `RConfig` subclasses (e.g. `MemoryConfig`) keep working as-is.
- `recurrent/generation_manager.py` / `recurrent/async_generation_manager.py`
  — only type hints and structured `logger.info(..., extra={...})` calls were
  added; control flow, return values, and public method signatures are
  unchanged. `AsyncLLMGenerationManager` still raises `NotImplementedError`
  at construction — that guard was preserved, not touched.
- `recurrent/impls/*` agent implementations (`memory.py`, `memory_img.py`,
  etc.) — untouched by this pass.

## Migration steps for an existing shell-script run

1. Find the equivalent `key=value` overrides your `train.sh` passes to
   `verl.trainer.main_ppo` (e.g. `data.train_files=...`,
   `actor_rollout_ref.model.path=...`).
2. Create (or reuse) a `config/experiment/<name>.yaml` with those values
   under `experiment.*` / cross-referenced `model.checkpoint` /
   `memory.*`.
3. Replace the shell invocation with
   `python scripts/memory_train.py +experiment=<name>`.
4. Add `.env` (from `.env.example`) so host-specific paths resolve without
   editing YAML.

## Known gaps (call out explicitly, do not paper over)

- `scripts/memory_train.py::train_loop` is a scaffolding skeleton: it
  demonstrates checkpointing/logging/metric-aggregation wiring but does not
  yet call `RayPPOTrainer` or `LLMGenerationManager.run_llm_loop` inline —
  that integration is the next step, not done in this pass.
- `scripts/memory_eval.py::run_dataset_predictions` returns empty
  predictions until wired to a live Ray rollout worker group; the
  EM/F1/token-budget scoring functions themselves are fully implemented and
  unit-tested independent of that wiring.
- The async generation manager (`AsyncLLMGenerationManager`) is disabled
  upstream (`raise NotImplementedError`) and was left disabled; the sweep
  and train scripts only exercise the synchronous path.
