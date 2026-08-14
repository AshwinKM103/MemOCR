# Recipe Config Guide

Recipe training (`recipe/spin`, `recipe/langgraph_agent`, ...) composes on top
of the same Hydra groups as `config/config.yaml` (`memory`, `model`,
`experiment`, `logging`), plus a new `recipe` group. The composition root is
`config/recipe_config.yaml`; it is a sibling of `config/config.yaml`, not a
replacement -- `scripts/memory_train.py` is unaffected.

## Quick start

```bash
# SPIN (FSDP + DataParallel PPO), delegates to recipe/spin/run_spin.sh
python scripts/train_recipe.py recipe=spin experiment=baseline

# LangGraph ReactAgentLoop
python scripts/train_recipe.py recipe=langgraph_agent experiment=baseline

# Override memory density for one run
python scripts/train_recipe.py recipe=spin experiment=baseline \
    +memory/ablations=high

# Memory-density x recipe ablation sweep (parallel subprocess trials)
python scripts/sweep_recipe.py recipe=memory_ablation experiment=ablation_density

# Print the fully-resolved config without running anything
python scripts/train_recipe.py recipe=spin experiment=baseline --cfg job --resolve
```

Note: `experiment` (and `recipe`, `logging`, `memory`, `model`) are already
selected by `config/recipe_config.yaml`'s `defaults:` list, so overriding
them on the CLI uses `key=value`, not `+key=value` -- the `+` prefix is only
for keys _not_ already in the defaults list (e.g. `+memory/ablations=high`,
which selects a config _inside_ the already-selected `memory` group).

## The `recipe` group

`config/recipe/defaults.yaml` defines the schema every concrete recipe fills in:

| Field                   | Meaning                                                                                                                       |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `recipe.name`           | Run identity; folded into `run.name` and WandB tags.                                                                          |
| `recipe.description`    | Human-readable summary, logged with the config snapshot.                                                                      |
| `recipe.module`         | Dotted path to an in-process entrypoint (agent-loop recipes).                                                                 |
| `recipe.launch.command` | Shell command for recipes with their own launcher (SPIN's Ray/FSDP `run_spin.sh`). Preferred over `module` when both are set. |
| `recipe.launch.cwd`     | Working directory for `launch.command`; defaults to the repo root.                                                            |
| `recipe.tags`           | Extra WandB tags (e.g. `[spin, fsdp]`).                                                                                       |

`scripts/train_recipe.py` requires exactly one of `recipe.launch.command` or
`recipe.module` to be set; `validate_recipe_config` raises
`src.experiment.ExperimentError` otherwise, before any process is spawned.

### `config/recipe/spin.yaml`

SPIN owns its own Ray + FSDP dispatch (`recipe/spin/fsdp_workers.py`,
`SPINRolloutRefWorker`). `train_recipe.py` does not reimplement or import
that dispatch path; it subprocess-runs `recipe/spin/run_spin.sh` with
`MEMOCR_RUN_DIR`, `MEMOCR_RUN_NAME`, and `MEMOCR_RECIPE` exported, so the
launcher's own logs can be correlated with the WandB run and the
`registry.jsonl` entry `ExperimentManager` writes.

### `config/recipe/langgraph_agent.yaml`

Points `recipe.module` at `recipe.langgraph_agent.react_agent_loop`. In-process
driving of arbitrary `AgentLoopBase` subclasses is not yet implemented in
`train_recipe.py` (each agent loop has a distinct call signature); calling
`train_recipe.py recipe=langgraph_agent` currently raises `NotImplementedError`
naming the missing driver, rather than silently no-op-ing. Wiring a
recipe-specific driver into `run_in_process_module` is the next step for that
recipe (out of scope for the Hydra/WandB scaffolding pass).

### `config/recipe/memory_ablation.yaml`

Not selected as a recipe to _train_ -- it defines the sweep grid consumed by
`scripts/sweep_recipe.py`: the cartesian product of `recipe.sweep.recipe`
(which recipes to run) and `recipe.sweep.memory_ablation` (which
`config/memory/ablations.yaml` variant: `low` / `medium` / `high`). Each grid
point is one `train_recipe.py` subprocess.

## WandB tags and metrics

`src.recipe_logging.RecipeExperimentManager` (a drop-in `ExperimentManager`
subclass) computes WandB tags at construction time from:

- `experiment.name`
- `recipe:{recipe.name}` and each entry of `recipe.tags`
- `memory_density:{memory.density.target_tokens_per_1k_context}`
- `token_budget:{memory.max_memorization_length}`

so every recipe run is filterable in the WandB UI by recipe and by memory
ablation axis without extra dashboard setup. See `resolve_recipe_tags` in
`src/recipe_logging.py` for the exact merge logic.

`RecipeExperimentManager.log_agent_loop_output(step, output)` flattens an
`AgentLoopOutput.metrics` dict (or a plain `{"metrics": {...}}` dict) into
`agent/`-prefixed scalar metrics and forwards them through
`ExperimentManager.log_metrics`, which is the same path that already writes
to the JSON file sink and (if enabled) WandB.

## Checkpoint metadata

No new checkpoint format is introduced. `ExperimentManager.save_checkpoint`
already embeds `OmegaConf.to_container(self.config, resolve=True)` in
`metadata.json`; because the composed config now includes `recipe.*`, every
checkpoint saved from a recipe run is self-describing about which recipe and
recipe tags produced it, with no changes to `CheckpointMetadata` or the
on-disk checkpoint format.

## Known pre-existing gaps (not introduced by this change, flagged for follow-up)

- `config/config.yaml`'s own docstring and `docs/config-guide.md` document
  `+experiment=baseline` as the CLI invocation, but `experiment` is already
  in the `defaults:` list, so Hydra raises
  `experiment appears more than once in the final defaults list`. Verified:
  `python scripts/memory_train.py +experiment=baseline` fails; dropping the
  `+` (`experiment=baseline`) works. `config/recipe_config.yaml` and this
  guide use the working form.
- `config/env.yaml` is not wired into the Hydra `defaults:` list of either
  `config.yaml` or `config/recipe_config.yaml` (only referenced via
  `env_config_path` for manual loading). Any interpolation touching `env.*`
  (`env.paths.cache_root`, etc.) fails to resolve under `--cfg job --resolve`
  or `OmegaConf.to_container(..., resolve=True)`. Reproduced on the
  pre-existing `scripts/memory_train.py` as well as `train_recipe.py`, so
  this is a baseline gap, not a regression from the recipe integration.
