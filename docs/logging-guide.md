# Logging Guide

`src/logging.py` provides structured logging (JSON to file, human-readable
to console) with an optional WandB metric mirror. It follows
`.claude/rules/monitoring.md`: every log entry carries `timestamp`, `level`,
`service`, `requestId`, `message`, plus caller-supplied context.

## Basic usage

```python
from src.logging import setup_experiment_logging

logger = setup_experiment_logging(config=cfg, experiment_name="baseline_hotpotqa", run_name="run_001")
logger.info("training_step", step=1, loss=0.5, lr=2e-5)
logger.warning("dataset_skipped_no_data", dataset="nq")
logger.exception("checkpoint_save_failed", step=100)  # includes traceback
```

Each call accepts arbitrary keyword context. Numeric values (`int`/`float`)
are additionally forwarded to WandB as metrics when the `wandb` sink is
enabled (`config/logging/wandb.yaml`).

## Context manager for a full run

```python
from src.logging import experiment_run

with experiment_run(cfg, experiment_name="baseline_hotpotqa", run_name="run_001") as logger:
    logger.info("run_start")
    ...  # training/eval code
# WandB run is finished and a run_end/run_failed log line is emitted automatically,
# even if the block raises.
```

## Choosing a sink profile

| Profile   | Config            | Use for                                                                            |
| --------- | ----------------- | ---------------------------------------------------------------------------------- |
| `console` | `logging=console` | Local debugging, quick smoke tests                                                 |
| `file`    | `logging=file`    | CI runs, unattended jobs — JSON lines at `${run.output_dir}/logs/<run_name>.jsonl` |
| `wandb`   | `logging=wandb`   | Real experiments — console + file + WandB metrics                                  |

## Log levels

- `DEBUG` — verbose per-batch detail (padding, tensor shapes).
- `INFO` — step/epoch boundaries, checkpoint saves, run start/end.
- `WARNING` — recoverable issues (e.g. a dataset file missing during eval).
- `ERROR` — failures requiring attention; use `logger.exception(...)` to
  capture the traceback.

## What not to log

Per `.claude/rules/monitoring.md`: never log passwords, tokens, or full
request bodies. Model checkpoints paths and dataset names are fine; raw
user-submitted prompts should be truncated or hashed if they may contain
sensitive content.

## Testing

`tests/test_logging.py` mocks `wandb` via `unittest.mock.patch("src.logging.wandb")`
so tests never hit the network. Follow that pattern for any new code that
calls `wandb.init`/`wandb.log` directly.
