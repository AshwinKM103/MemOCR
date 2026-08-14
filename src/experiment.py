"""Experiment lifecycle management: checkpoints, metrics, and a run registry.

Usage:
    from src.experiment import ExperimentManager

    exp = ExperimentManager(config=cfg)
    exp.start_run(name="baseline")
    # ... train/eval code ...
    exp.save_checkpoint(step=100, metrics={"em": 0.62, "f1": 0.71})
    exp.end_run()
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from src.logging import ExperimentLogger, setup_experiment_logging


class ExperimentError(Exception):
    """Raised for experiment lifecycle misuse (e.g. checkpoint before start_run)."""


@dataclass
class CheckpointMetadata:
    """Metadata embedded alongside every checkpoint for reproducibility."""

    step: int
    run_id: str
    run_name: str
    timestamp: float
    git_commit_sha: Optional[str]
    config: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class RunRecord:
    """One entry in the experiment registry (`registry.jsonl`)."""

    run_id: str
    run_name: str
    experiment_name: str
    started_at: float
    ended_at: Optional[float] = None
    git_commit_sha: Optional[str] = None
    output_dir: str = ""
    final_metrics: dict[str, float] = field(default_factory=dict)
    status: str = "running"  # running | completed | failed


def _current_git_commit_sha() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


class MetricAggregator:
    """Running aggregation of scalar metrics across steps.

    Tracks token budget, info density, and QA accuracy (or any scalar metric
    the caller reports) with mean/min/max, so callers do not need to keep
    their own accumulators.
    """

    def __init__(self) -> None:
        self._history: dict[str, list[float]] = {}

    def add(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self._history.setdefault(key, []).append(float(value))

    def summary(self) -> dict[str, float]:
        summary: dict[str, float] = {}
        for key, values in self._history.items():
            if not values:
                continue
            summary[f"{key}_mean"] = sum(values) / len(values)
            summary[f"{key}_min"] = min(values)
            summary[f"{key}_max"] = max(values)
            summary[f"{key}_last"] = values[-1]
        return summary

    def latest(self) -> dict[str, float]:
        return {key: values[-1] for key, values in self._history.items() if values}


class ExperimentManager:
    """Config-driven experiment lifecycle: run start/end, checkpoints, registry.

    One `ExperimentManager` is created per Hydra config; `start_run` /
    `end_run` may be called multiple times sequentially (e.g. across sweep
    trials sharing a manager), but not re-entered — call `end_run` before the
    next `start_run`.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.experiment_name: str = OmegaConf.select(config, "experiment.name", default="unnamed_experiment")
        self.output_root = Path(OmegaConf.select(config, "run.output_dir", default="./runs"))
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.output_root.parent / "registry.jsonl"

        self._run_id: Optional[str] = None
        self._run_name: Optional[str] = None
        self._run_dir: Optional[Path] = None
        self._logger: Optional[ExperimentLogger] = None
        self._metrics = MetricAggregator()
        self._started_at: Optional[float] = None

    @property
    def logger(self) -> ExperimentLogger:
        if self._logger is None:
            raise ExperimentError("start_run() must be called before accessing the logger.")
        return self._logger

    @property
    def run_dir(self) -> Path:
        if self._run_dir is None:
            raise ExperimentError("start_run() must be called before accessing run_dir.")
        return self._run_dir

    def start_run(self, name: str) -> str:
        """Begin a new run under this experiment. Returns the generated run_id."""
        if self._run_id is not None:
            raise ExperimentError(f"Run '{self._run_name}' already active; call end_run() first.")

        self._run_id = uuid.uuid4().hex[:12]
        self._run_name = name
        self._run_dir = self.output_root / name
        self._run_dir.mkdir(parents=True, exist_ok=True)
        (self._run_dir / "checkpoints").mkdir(exist_ok=True)
        self._metrics = MetricAggregator()
        self._started_at = time.time()

        self._logger = setup_experiment_logging(self.config, self.experiment_name, name)
        self._write_registry_record(status="running")
        self.logger.info(
            "experiment_run_started",
            run_id=self._run_id,
            run_name=name,
            experiment=self.experiment_name,
            git_commit_sha=_current_git_commit_sha() or "unknown",
        )
        return self._run_id

    def log_metrics(self, step: int, metrics: dict[str, float]) -> None:
        """Record metrics for aggregation and forward to the logger/WandB."""
        self._metrics.add(metrics)
        self.logger.log_metrics(metrics, step=step)

    def save_checkpoint(
        self, step: int, metrics: Optional[dict[str, float]] = None, state: Optional[dict[str, Any]] = None
    ) -> Path:
        """Save a checkpoint directory with embedded Hydra + git metadata.

        `state` is caller-provided (e.g. model/optimizer state_dict paths
        already written by the training loop); this method only writes the
        `metadata.json` sidecar so checkpoints stay self-describing.
        """
        if metrics:
            self._metrics.add(metrics)

        checkpoint_dir = self.run_dir / "checkpoints" / f"step_{step:08d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        metadata = CheckpointMetadata(
            step=step,
            run_id=self._run_id or "unknown",
            run_name=self._run_name or "unknown",
            timestamp=time.time(),
            git_commit_sha=_current_git_commit_sha(),
            config=OmegaConf.to_container(self.config, resolve=True),
            metrics=metrics or {},
        )
        (checkpoint_dir / "metadata.json").write_text(json.dumps(asdict(metadata), indent=2, default=str))

        if state is not None:
            for key, path_or_bytes in state.items():
                if isinstance(path_or_bytes, (str, Path)):
                    shutil.copy2(path_or_bytes, checkpoint_dir / Path(key).name)
                else:
                    (checkpoint_dir / f"{key}.json").write_text(json.dumps(path_or_bytes, default=str))

        self.logger.info("checkpoint_saved", step=step, path=str(checkpoint_dir))
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir: Path | str) -> CheckpointMetadata:
        """Load and validate a checkpoint's metadata sidecar.

        Raises FileNotFoundError if the checkpoint or its metadata is missing.
        """
        checkpoint_dir = Path(checkpoint_dir)
        metadata_path = checkpoint_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No metadata.json found under checkpoint dir: {checkpoint_dir}")
        raw = json.loads(metadata_path.read_text())
        return CheckpointMetadata(**raw)

    def end_run(self, status: str = "completed") -> dict[str, float]:
        """End the active run, flush the registry record, return the metric summary."""
        if self._run_id is None:
            raise ExperimentError("end_run() called with no active run.")

        summary = self._metrics.summary()
        self.logger.info("experiment_run_ended", run_id=self._run_id, status=status, **summary)
        self._write_registry_record(status=status, final_metrics=self._metrics.latest())

        if self._logger is not None and self._logger._wandb_run is not None:
            self._logger._wandb_run.finish()

        self._run_id = None
        self._run_name = None
        self._run_dir = None
        self._logger = None
        return summary

    def _write_registry_record(self, status: str, final_metrics: Optional[dict[str, float]] = None) -> None:
        record = RunRecord(
            run_id=self._run_id or "unknown",
            run_name=self._run_name or "unknown",
            experiment_name=self.experiment_name,
            started_at=self._started_at or time.time(),
            ended_at=time.time() if status != "running" else None,
            git_commit_sha=_current_git_commit_sha(),
            output_dir=str(self._run_dir) if self._run_dir else "",
            final_metrics=final_metrics or {},
            status=status,
        )
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with self.registry_path.open("a") as f:
            f.write(json.dumps(asdict(record), default=str) + "\n")
