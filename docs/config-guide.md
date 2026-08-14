# Config Guide

MemOCR experiments are configured with [Hydra](https://hydra.cc). The
composition root is `config/config.yaml`; groups live under `config/<group>/`.
See `config/defaults/README.md` for the terse schema table.

## Quick start

```bash
# Baseline run, all defaults
python scripts/memory_train.py +experiment=baseline

# Override a leaf value
python scripts/memory_train.py +experiment=baseline memory.chunk_size=512

# Swap a config group entirely
python scripts/memory_train.py +experiment=baseline model=qwen logging=wandb

# Print the fully-resolved config without running anything
python scripts/memory_train.py +experiment=baseline --cfg job --resolve
```

## Groups

- `memory` — `MemoryConfig` fields consumed by `recurrent.impls.memory.MemoryAgent`
  (`chunk_size`, `max_chunks`, `max_memorization_length`, ...).
- `model` — Qwen-VL checkpoint/inference params, embedding model, tokenizer policy.
- `experiment` — top-level run definition: dataset, data file paths, train/eval settings.
- `logging` — which sinks are active: `console` (default), `file` (+JSON lines), `wandb` (+metrics).
- `env` — host-specific paths and credentials, sourced from `.env` (see `.env.example`).

## Environment variables

`config/env.yaml` reads from the process environment via
`${oc.env:VAR_NAME,default}`. Copy `.env.example` to `.env` and source it (or
export the variables in your shell/launcher) before running any script:

```bash
cp .env.example .env
# edit .env with real paths
set -a; source .env; set +a
python scripts/memory_train.py +experiment=baseline
```

## Adding a memory-density ablation value

1. Add a new key under `config/memory/ablations.yaml` (e.g. `very_high`).
2. Add the value to `experiment.sweep.values` in
   `config/experiment/ablation_density.yaml`.
3. Run `python scripts/memory_sweep.py -m +experiment=ablation_density`.

## Validation

Hydra fails fast on missing required keys (`ExperimentError` from
`scripts/memory_train.py::validate_config`) and on out-of-range generation
parameters (`src.generation.GenerationConfig`, a pydantic model) before any
model weights are loaded. If you see a `ConfigCompositionException` or a
pydantic `ValidationError` at startup, fix the YAML — do not catch and
suppress it in the script.
