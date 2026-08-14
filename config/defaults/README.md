# Config Schema

This directory documents the Hydra config groups under `config/`. See
`docs/config-guide.md` for usage examples; this file is the terse schema
reference.

## Groups

| Group        | File(s)                                    | `@package`         | Purpose                                             |
| ------------ | ------------------------------------------ | ------------------ | --------------------------------------------------- |
| (root)       | `config/config.yaml`                       | —                  | Composition root, `run.*`, hydra job/sweep dirs     |
| `env`        | `config/env.yaml`                          | `env`              | Host paths, GPU, WandB creds — from `.env`          |
| `memory`     | `config/memory/default.yaml`               | `memory`           | `MemoryConfig` fields (chunk_size, max_chunks, ...) |
| `memory`     | `config/memory/ablations.yaml`             | `memory.ablations` | Named density presets: low/medium/high              |
| `memory`     | `config/memory/datasets.yaml`              | `memory.datasets`  | Per-dataset chunking overrides                      |
| `model`      | `config/model/qwen.yaml`                   | `model`            | Qwen-VL checkpoint + inference params               |
| `model`      | `config/model/embedding.yaml`              | `model.embedding`  | Sentence-embedding model for retrieval              |
| `model`      | `config/model/tokenizer.yaml`              | `model.tokenizer`  | Tokenizer padding/truncation policy                 |
| `experiment` | `config/experiment/baseline.yaml`          | `experiment`       | Single-dataset training/eval run                    |
| `experiment` | `config/experiment/ablation_density.yaml`  | `experiment`       | Density sweep over `memory.ablations`               |
| `experiment` | `config/experiment/multi_dataset.yaml`     | `experiment`       | Fixed-checkpoint multi-dataset eval                 |
| `logging`    | `config/logging/{console,file,wandb}.yaml` | `logging`          | Sink selection: console / +file / +wandb            |

## Field ownership

- `memory.*` fields map 1:1 onto `recurrent.impls.memory.MemoryConfig` (a
  subclass of `recurrent.interface.RConfig`). Do not add fields here that
  `MemoryConfig` does not accept — Pydantic/dataclass validation will fail
  fast in `src/experiment.py::ExperimentManager._validate_config`.
- `model.*` fields feed `LLMGenerationManager` / `AsyncLLMGenerationManager`
  via `src.generation.GenerationConfig` (pydantic), not directly — this
  keeps the YAML schema and the runtime validation schema in sync in one
  place.
- `run.output_dir` is the single source of truth for where checkpoints,
  logs, and WandB run metadata land for a given invocation.

## Adding a new experiment

1. Add `config/experiment/<name>.yaml` with at least `name`, `dataset`,
   `data_files`.
2. Run `python scripts/memory_train.py +experiment=<name>` — Hydra will
   fail fast with a clear `ConfigCompositionException` if a required key
   is missing, before any model is loaded.
