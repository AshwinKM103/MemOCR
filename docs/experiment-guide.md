# Experiment Guide

Three entrypoints under `scripts/` cover the experiment lifecycle:
training (`memory_train.py`), evaluation (`memory_eval.py`), and sweeps
(`memory_sweep.py`). All three are Hydra apps rooted at `config/config.yaml`.

## Running the baseline

```bash
python scripts/memory_train.py +experiment=baseline
```

This:

1. Validates the composed config (`scripts/memory_train.py::validate_config`)
   — fails fast if required keys are missing or `model.inference.*` is out
   of range.
2. Starts a run via `ExperimentManager` (`src/experiment.py`), which creates
   `${run.output_dir}` and appends a `running` record to `registry.jsonl`.
3. Runs the training loop, checkpointing every `experiment.train.save_freq`
   steps with embedded metadata (git commit SHA, resolved config, metrics).
4. Ends the run, flushing an aggregated metric summary and a `completed`
   (or `failed`) registry record.

## Running evaluation

```bash
python scripts/memory_eval.py +experiment=baseline model.checkpoint=/path/to/ckpt
```

Computes EM/F1/token-budget per dataset listed in `experiment.eval.datasets`,
writes `<run_dir>/eval_results.json`, and logs metrics through the same
`ExperimentManager`/`src.logging` stack as training.

For a full multi-dataset sweep:

```bash
python scripts/memory_eval.py +experiment=multi_dataset model.checkpoint=/path/to/ckpt
```

## Running a density sweep

```bash
python scripts/memory_sweep.py -m +experiment=ablation_density
```

Expands `experiment.sweep.values` (see
`config/experiment/ablation_density.yaml`) into one `memory_train.py`
subprocess per value, run in parallel via a `multiprocessing.Pool`. Each
trial gets its own `run.output_dir` (Hydra's `hydra.sweep.subdir`), so
results do not collide.

## Rollback / recovery

Every checkpoint is self-describing: `<checkpoint_dir>/metadata.json`
embeds the step, run id, git commit SHA, resolved config, and reported
metrics. To roll back to a previous checkpoint:

```python
from src.experiment import ExperimentManager
exp = ExperimentManager(config=cfg)
metadata = exp.load_checkpoint("/path/to/checkpoints/step_00000010")
```

`tests/test_experiment_manager.py::TestCheckpoints::test_rollback_restores_previous_checkpoint_metadata`
exercises this path — verify it before trusting a rollback in production,
per `.claude/rules/monitoring.md`'s "test the rollback procedure" guidance
(mirrored from the MLOps agent's pre-completion checklist).

## Registry

`ExperimentManager` appends one JSON line per lifecycle transition to
`<output_root>/registry.jsonl` (sibling of the per-run output dirs). Use it
to answer "what ran, when, with what config, and what were the final
metrics" without digging through individual run directories:

```bash
cat runs/registry.jsonl | python3 -c "import sys, json; [print(json.loads(l)) for l in sys.stdin]"
```
