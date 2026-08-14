"""Structured logging for MemOCR experiments.

Provides a dual-sink (console + JSON file) structured logger with optional
WandB metric mirroring, plus a context manager for experiment-run lifecycle
(start/end, guaranteed cleanup on exception).

Usage:
    from src.logging import setup_experiment_logging, get_logger

    logger = setup_experiment_logging(
        config=cfg, experiment_name="baseline_hotpotqa", run_name="run_001"
    )
    logger.info("training_step", step=1, loss=0.5)
    # -> console (human-readable), file (JSON lines), WandB (if enabled)
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from omegaconf import DictConfig, OmegaConf

try:
    import wandb
except ImportError:  # pragma: no cover - wandb is an optional runtime dep
    wandb = None


_STANDARD_LOG_RECORD_FIELDS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


class JsonFormatter(logging.Formatter):
    """Renders each log record as a single JSON line.

    Required fields per `.claude/rules/monitoring.md`: timestamp, level,
    service, requestId, message. Any additional keyword arguments passed to
    the logging call (via `extra=...`) are merged in verbatim.
    """

    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self.service,
            "requestId": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_LOG_RECORD_FIELDS and key != "request_id":
                payload[key] = _json_safe(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _json_safe(value: Any) -> Any:
    """Best-effort conversion of arbitrary log payload values to JSON-safe types."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


class ConsoleFormatter(logging.Formatter):
    """Human-readable single-line console format."""

    def __init__(self) -> None:
        super().__init__(fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s%(context)s")

    def format(self, record: logging.LogRecord) -> str:
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_LOG_RECORD_FIELDS and key != "request_id"
        }
        record.context = f" | {extras}" if extras else ""
        return super().format(record)


class ExperimentLogger:
    """Structured logger with an optional WandB metric sink.

    Wraps a stdlib `logging.Logger`. Log calls accept arbitrary keyword
    context (`logger.info("event", step=1, loss=0.5)`), which is attached to
    both the JSON file sink and, if enabled, forwarded to WandB as metrics.
    """

    def __init__(
        self,
        logger: logging.Logger,
        request_id: str,
        wandb_run: Optional[Any] = None,
        wandb_log_freq: int = 1,
    ) -> None:
        self._logger = logger
        self._request_id = request_id
        self._wandb_run = wandb_run
        self._wandb_log_freq = max(1, wandb_log_freq)
        self._call_count = 0

    def _log(self, level: int, message: str, **context: Any) -> None:
        self._logger.log(level, message, extra={"request_id": self._request_id, **context})
        numeric_context = {k: v for k, v in context.items() if isinstance(v, (int, float))}
        if self._wandb_run is not None and numeric_context:
            self._call_count += 1
            if self._call_count % self._wandb_log_freq == 0:
                self._wandb_run.log(numeric_context, step=context.get("step"))

    def debug(self, message: str, **context: Any) -> None:
        self._log(logging.DEBUG, message, **context)

    def info(self, message: str, **context: Any) -> None:
        self._log(logging.INFO, message, **context)

    def warning(self, message: str, **context: Any) -> None:
        self._log(logging.WARNING, message, **context)

    def error(self, message: str, **context: Any) -> None:
        self._log(logging.ERROR, message, **context)

    def exception(self, message: str, **context: Any) -> None:
        self._logger.error(message, extra={"request_id": self._request_id, **context}, exc_info=True)

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log a metrics dict to the file sink and WandB in one call."""
        self.info("metrics", step=step, **metrics)


@dataclass
class LoggingSettings:
    """Resolved view of the `logging.*` config block, with safe defaults."""

    console_enabled: bool = True
    console_level: str = "INFO"
    file_enabled: bool = False
    file_level: str = "DEBUG"
    file_path: Optional[str] = None
    file_rotate_mb: int = 100
    file_backup_count: int = 5
    wandb_enabled: bool = False
    wandb_entity: Optional[str] = None
    wandb_project: str = "memocr"
    wandb_mode: str = "online"
    wandb_tags: list[str] = field(default_factory=list)
    wandb_log_freq: int = 1

    @classmethod
    def from_config(cls, config: DictConfig) -> LoggingSettings:
        logging_cfg = config.get("logging") if hasattr(config, "get") else None
        if logging_cfg is None:
            return cls()
        as_container = OmegaConf.to_container(logging_cfg, resolve=True)
        console = as_container.get("console", {}) or {}
        file_cfg = as_container.get("file", {}) or {}
        wandb_cfg = as_container.get("wandb", {}) or {}
        return cls(
            console_enabled=console.get("enabled", True),
            console_level=console.get("level", "INFO"),
            file_enabled=file_cfg.get("enabled", False),
            file_level=file_cfg.get("level", "DEBUG"),
            file_path=file_cfg.get("path"),
            file_rotate_mb=file_cfg.get("rotate_mb", 100),
            file_backup_count=file_cfg.get("backup_count", 5),
            wandb_enabled=wandb_cfg.get("enabled", False),
            wandb_entity=wandb_cfg.get("entity"),
            wandb_project=wandb_cfg.get("project", "memocr"),
            wandb_mode=wandb_cfg.get("mode", "online"),
            wandb_tags=list(wandb_cfg.get("tags", [])),
            wandb_log_freq=wandb_cfg.get("log_freq", 1),
        )


def setup_experiment_logging(
    config: DictConfig,
    experiment_name: str,
    run_name: str,
    service: str = "memocr",
) -> ExperimentLogger:
    """Configure console/file/WandB sinks and return a structured logger.

    Idempotent per (experiment_name, run_name): calling this twice for the
    same run_name reuses the same underlying `logging.Logger` and does not
    duplicate handlers.
    """
    settings = LoggingSettings.from_config(config)
    logger_name = f"{service}.{experiment_name}.{run_name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        if settings.console_enabled:
            console_handler = logging.StreamHandler(stream=sys.stdout)
            console_handler.setLevel(settings.console_level)
            console_handler.setFormatter(ConsoleFormatter())
            logger.addHandler(console_handler)

        if settings.file_enabled and settings.file_path:
            file_path = Path(settings.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                filename=str(file_path),
                maxBytes=settings.file_rotate_mb * 1024 * 1024,
                backupCount=settings.file_backup_count,
            )
            file_handler.setLevel(settings.file_level)
            file_handler.setFormatter(JsonFormatter(service=service))
            logger.addHandler(file_handler)

    wandb_run = None
    if settings.wandb_enabled:
        wandb_run = _init_wandb(settings, experiment_name, run_name, config)

    request_id = str(uuid.uuid4())
    return ExperimentLogger(logger, request_id, wandb_run, settings.wandb_log_freq)


def _init_wandb(
    settings: LoggingSettings,
    experiment_name: str,
    run_name: str,
    config: DictConfig,
) -> Optional[Any]:
    if wandb is None:
        logging.getLogger(__name__).warning(
            "logging.wandb.enabled=true but the `wandb` package is not installed; skipping WandB sink."
        )
        return None
    if settings.wandb_mode == "disabled":
        return None
    return wandb.init(
        entity=settings.wandb_entity,
        project=settings.wandb_project,
        name=run_name,
        group=experiment_name,
        tags=settings.wandb_tags,
        mode=settings.wandb_mode,
        config=OmegaConf.to_container(config, resolve=True) if config is not None else None,
        reinit=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a plain stdlib logger for library code that does not need WandB."""
    return logging.getLogger(name)


@contextmanager
def experiment_run(
    config: DictConfig,
    experiment_name: str,
    run_name: str,
) -> Iterator[ExperimentLogger]:
    """Context manager wrapping the lifetime of one experiment run.

    Guarantees the WandB run is finished (even on exception) and emits a
    start/end log pair for auditability.
    """
    logger = setup_experiment_logging(config, experiment_name, run_name)
    logger.info("run_start", experiment=experiment_name, run=run_name)
    try:
        yield logger
    except Exception:
        logger.exception("run_failed", experiment=experiment_name, run=run_name)
        raise
    finally:
        logger.info("run_end", experiment=experiment_name, run=run_name)
        if logger._wandb_run is not None:
            logger._wandb_run.finish()
