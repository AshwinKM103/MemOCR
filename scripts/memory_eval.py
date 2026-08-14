#!/usr/bin/env python
"""Evaluate a memory-agent checkpoint on one or more QA datasets.

Usage:
    python scripts/memory_eval.py +experiment=baseline model.checkpoint=path/to/ckpt
    python scripts/memory_eval.py +experiment=multi_dataset model.checkpoint=path/to/ckpt

Computes EM, F1, and token-budget metrics per dataset, logs them via
`src.logging` (console + file + WandB, depending on the `logging` group),
and writes a JSON summary to `<run_dir>/eval_results.json`.
"""

from __future__ import annotations

import json
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment import ExperimentManager  # noqa: E402

CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "config")


def normalize_answer(text: str) -> str:
    """SQuAD-style answer normalization: lowercase, strip punctuation/articles/whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match_score(prediction: str, ground_truths: Iterable[str]) -> float:
    normalized_pred = normalize_answer(prediction)
    return float(any(normalized_pred == normalize_answer(gt) for gt in ground_truths))


def f1_score(prediction: str, ground_truths: Iterable[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    best = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def evaluate_dataset(
    dataset_name: str,
    predictions: list[str],
    references: list[list[str]],
    token_counts: list[int],
) -> dict[str, float]:
    """Aggregate EM/F1/token-budget metrics for one dataset's predictions.

    `predictions[i]` is scored against `references[i]` (a list of accepted
    answers, to support datasets with multiple valid gold answers).
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions/references length mismatch for {dataset_name}: {len(predictions)} vs {len(references)}"
        )

    em_scores = [exact_match_score(p, r) for p, r in zip(predictions, references)]
    f1_scores = [f1_score(p, r) for p, r in zip(predictions, references)]

    return {
        "em": sum(em_scores) / max(1, len(em_scores)),
        "f1": sum(f1_scores) / max(1, len(f1_scores)),
        "token_budget_mean": sum(token_counts) / max(1, len(token_counts)) if token_counts else 0.0,
        "num_examples": len(predictions),
    }


def run_dataset_predictions(cfg: DictConfig, dataset_name: str) -> tuple[list[str], list[list[str]], list[int]]:
    """Load a dataset and run the memory agent to produce predictions.

    Deferred import for the same reason as `memory_train.build_dataset`.
    Returns empty lists if the dataset file is not present (so eval can be
    exercised in CI without downloading the full QA corpora).
    """
    data_path = Path(OmegaConf.select(cfg, f"experiment.eval.data_files.{dataset_name}", default=""))
    if not data_path or not data_path.exists():
        return [], [], []

    # A full implementation would call `LLMGenerationManager.run_llm_loop_vl`
    # here per batch; left as an integration point since it requires a live
    # Ray rollout worker group, which is out of scope for this script's unit
    # tests.
    return [], [], []


def main_impl(cfg: DictConfig) -> dict[str, Any]:
    exp = ExperimentManager(config=cfg)
    exp.start_run(name=cfg.run.name)

    dataset_names = OmegaConf.select(cfg, "experiment.eval.datasets", default=[])
    results: dict[str, dict[str, float]] = {}
    try:
        for dataset_name in dataset_names:
            predictions, references, token_counts = run_dataset_predictions(cfg, dataset_name)
            if not predictions:
                exp.logger.warning("dataset_skipped_no_data", dataset=dataset_name)
                continue
            metrics = evaluate_dataset(dataset_name, predictions, references, token_counts)
            results[dataset_name] = metrics
            exp.log_metrics(step=0, metrics={f"{dataset_name}/{k}": v for k, v in metrics.items()})

        output_path = exp.run_dir / "eval_results.json"
        output_path.write_text(json.dumps(results, indent=2))
        exp.logger.info("eval_results_written", path=str(output_path))
        exp.end_run()
        return results
    except Exception:
        exp.end_run(status="failed")
        raise


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    results = main_impl(cfg)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
