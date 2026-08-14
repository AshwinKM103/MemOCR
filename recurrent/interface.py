# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import importlib.util
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.protocol import DataProto, DataProtoItem
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn


@dataclass
class RConfig:
    """
    Configuration for Multi-turn Policy Optimization.
    Just an interface. Add anything you need in a subclass of it.
    """

    pass


class RDataset(RLHFDataset):
    """
    Dataset for Multi-turn Policy Optimization.
    This class can be used directly as a subclass of RLHFDataset for RecurrentRL
    (if you do not need any new features)

    Overwritten Method:
        - __getitem__: get a single sample
        - get_batch_keys: tensor keys and non-tensor keys, should be contained in the batch.
        - get_collate_fn: collate function for dataloader, default to the same as RLHFDataset.

    The inherited methods are hdfs/parquet related methods.
    Make sure to call super().__init__() in your subclass to reuse RLHFDataset's initializer.
    """

    def __init__(
        self,
        recurrent_config: RConfig,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        data_config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        super().__init__(data_files=data_files, tokenizer=tokenizer, config=data_config, processor=processor)

    def __getitem__(self, item) -> dict:
        """
        Enforce subclass to override this method by declaring it as an abstract method.
        If you don't want to change its behavior, just return super().__getitem__(item).
        """
        row_dict = super().__getitem__(item)
        # used in validation metrics reduce
        row_dict["sample_uuid"] = str(uuid4())
        return row_dict

    def get_bactch_keys(self) -> tuple[list[str], list[str]]:
        return ["input_ids", "attention_mask", "position_ids"], []

    @staticmethod
    def get_collate_fn():
        return collate_fn


from .async_utils import ChatCompletionProxy


class AsyncOutput(ABC):
    """Output container for async LLM generation with timing and metrics."""

    def __init__(
        self,
        conversations: list[list[dict[str, str]]],
        sample_index: int,
        final_mask: bool,
        timing_raw: dict[str, float],
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Initialize async output.

        Args:
            conversations: List of conversation turns with messages.
            sample_index: Index of the sample in the original batch.
            final_mask: Whether this is the final turn.
            timing_raw: Timing information for profiling.
            metrics: Optional metrics dictionary.
        """
        self.conversations = conversations
        self.sample_index = sample_index
        self.final_mask = final_mask
        self.timing_raw = timing_raw
        if metrics is None:
            metrics = {}
        self.metrics = metrics
        if "workflow/num_conv" not in metrics:
            metrics["workflow/num_conv"] = len(conversations)


class AsyncRAgent(ABC):
    """Async recurrent agent interface.

    Manages multi-turn generation for a single sample:
    1. Initialize resources (__init__)
    2. Start generation (start)
    3. Rollout one turn with LLM (rollout)
    4. Cleanup (end)
    """

    def __init__(
        self,
        proxy: ChatCompletionProxy,
        tokenizer: PreTrainedTokenizer,
        config: RConfig,
        rollout_config: DictConfig,
    ) -> None:
        """Initialize async agent.

        Args:
            proxy: Chat completion proxy for LLM calls.
            tokenizer: Tokenizer for encoding/decoding.
            config: Agent configuration.
            rollout_config: Rollout-specific configuration.
        """
        self.proxy = proxy
        self.tokenizer = tokenizer
        self.config = config
        self.rollout_config = rollout_config
        self.timing_raw: dict[str, float] = {}

    def start(self, gen_batch: DataProto, timing_raw: dict[str, float]) -> None:
        """Initialize generation for a batch (optional override).

        Args:
            gen_batch: Batch of samples to generate from.
            timing_raw: Timing dictionary to accumulate timing data.
        """
        pass

    def end(self) -> None:
        """Cleanup after generation (optional override)."""
        pass

    @abstractmethod
    async def rollout(self, gen_item: DataProtoItem) -> AsyncOutput:
        """Rollout a single sample.

        Returns:
            AsyncOutput with conversations, sample index, final mask, timing, and metrics.
        """
        pass

    def sampling_params(self, meta_info: dict[str, Any]) -> dict[str, Any]:
        """Get sampling parameters for LLM generation.

        Adapted from works/rollout/vllm_spmd_rollout.
        Note: max_completion_tokens must be specified by subclass.
        Note: top_k is not supported in async mode.

        Args:
            meta_info: Metadata with do_sample and validate flags.

        Returns:
            Dictionary of sampling parameters (n, temperature, top_p, etc.).
        """
        kwargs: dict[str, Any] = {
            "n": 1,
            "temperature": self.rollout_config.temperature,
            "top_p": self.rollout_config.top_p,
        }
        do_sample = meta_info.get("do_sample", True)
        is_validate = meta_info.get("validate", False)
        if not do_sample:
            kwargs.update(
                {
                    "best_of": 1,
                    "top_p": 1.0,
                    "min_p": 0.0,
                    "temperature": 0,
                    "n": 1,
                }
            )
        elif is_validate:
            kwargs.update(
                {
                    "top_p": self.rollout_config.val_kwargs.top_p,
                    "temperature": self.rollout_config.val_kwargs.temperature,
                    "n": 1,
                }
            )

        return kwargs

    def reduce_timings(self, timing_raws: list[dict[str, float]]) -> dict[str, float]:
        """Reduce timing data from multiple agents.

        Averages async timings (can execute in parallel), sums sync timings (sequential).
        Naming convention: "async" in key name means the code is an await statement.

        Args:
            timing_raws: List of timing dictionaries from multiple agents.

        Returns:
            Reduced timing dictionary with aggregated timing data.
        """
        reduced: dict[str, float] = {}
        for k in timing_raws[0]:
            if "async" in k:
                # Async code can execute in parallel, so average
                reduced[k] = sum(timing_raw[k] for timing_raw in timing_raws) / len(timing_raws)
            else:
                # Sync code executes sequentially, so sum
                reduced[k] = sum(timing_raw[k] for timing_raw in timing_raws)
        return reduced

    def reduce_metrics(self, metrics: list[dict[str, Any]]) -> dict[str, Any]:
        """Reduce metrics from multiple agents.

        Computes mean, min, and max for each metric across agents.

        Args:
            metrics: List of metric dictionaries from multiple agents.

        Returns:
            Reduced metrics dictionary with _mean, _min, _max suffixes.
        """
        reduced: dict[str, Any] = {}
        for k in metrics[0]:
            reduced[k + "_mean"] = np.mean([m[k] for m in metrics])
            reduced[k + "_min"] = np.min([m[k] for m in metrics])
            reduced[k + "_max"] = np.max([m[k] for m in metrics])
        return reduced


class RAgent(ABC):
    """Recurrent agent interface for multi-turn generation.

    Subclasses should implement:
    1. __init__: Initialize agent with tokenizer and config
    2. start: Initialize generation for a batch
    3. action: Generate prompts/input_ids for current turn
    4. update: Process LLM output and update agent state
    5. done: Check if generation should stop
    6. end: Cleanup and return final masks/indices
    """

    @abstractmethod
    def __init__(self, tokenizer: PreTrainedTokenizer, config: RConfig) -> None:
        """Initialize recurrent agent.

        Args:
            tokenizer: Tokenizer for encoding/decoding.
            config: Agent configuration.
        """
        pass

    @abstractmethod
    def start(self, gen_batch: DataProto, timing_raw: dict[str, float]) -> None:
        """Initialize generation loop for a batch.

        Called once at the beginning. Initialize internal state.

        Args:
            gen_batch: Batch of samples with context.
            timing_raw: Dictionary to accumulate timing data.
        """
        self.gen_batch = gen_batch
        self.timing_raw = timing_raw
        self.step = 0
        self.final_mask_list: list[torch.Tensor] = []
        self.sample_index_list: list[torch.Tensor] = []
        pass

    @abstractmethod
    def action(self) -> tuple[list[torch.Tensor], dict[str, Any]]:
        """Generate prompts for current turn.

        Called once per rollout step. Return input_ids and metadata.
        Update sample_index and final_mask in internal state.

        Returns:
            Tuple of (input_ids list, meta_info dict).
        """
        sample_index = torch.arange(len(self.gen_batch), dtype=torch.long)
        self.sample_index_list.append(sample_index)
        self.final_mask_list.append(torch.full(sample_index.shape, False, dtype=torch.bool))
        pass

    @abstractmethod
    def update(self, gen_output: DataProto) -> DataProto:
        """Process LLM output and update agent state.

        Called after each rollout. Can execute tool calls or other custom logic.

        Args:
            gen_output: LLM-generated batch from generation.

        Returns:
            Updated generation output batch.
        """
        pass

    @abstractmethod
    def done(self) -> bool:
        """Check if generation loop should stop.

        Returns:
            True if generation should end, False otherwise.
        """
        return False

    @abstractmethod
    def end(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Cleanup and return final generation results.

        Called after done() returns True. Clean up internal state to save memory.

        Returns:
            Tuple of (final_mask: bool tensor, sample_index: long tensor).
        """
        del self.gen_batch
        del self.timing_raw
        self.step = 0
        sample_index = torch.cat(self.sample_index_list)
        final_mask = torch.cat(self.final_mask_list)
        del self.final_mask_list
        del self.sample_index_list
        return final_mask, sample_index


@dataclass
class RRegister:
    """Registry for custom recurrent implementation classes.

    Holds references to config, dataset, and agent classes for creating
    recurrent RL pipeline instances.
    """

    config_cls: type[RConfig]
    dataset_cls: type[RDataset]
    agent_cls: type[RAgent]

    @classmethod
    def from_filename(cls, file_path: str, obj_name: str) -> RRegister:
        """Load register from a Python file.

        Args:
            file_path: Path to Python file containing the register object.
            obj_name: Name of the RRegister object in the file.

        Returns:
            Loaded RRegister instance.

        Raises:
            FileNotFoundError: If file_path does not exist.
            AttributeError: If obj_name is not found in the module.
            TypeError: If obj_name is not an RRegister instance.
            RuntimeError: If module loading fails.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Recurrent implementation file '{file_path}' not found.")

        spec = importlib.util.spec_from_file_location("custom_module", file_path)
        if not spec:
            raise FileNotFoundError(f"Failed to create model spec for '{file_path}'.")
        module = importlib.util.module_from_spec(spec)
        try:
            if spec.loader is None:
                raise RuntimeError(f"Loader not found for '{file_path}'")
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Error loading module from '{file_path}': {e}")

        if not hasattr(module, obj_name):
            raise AttributeError(f"Register object '{obj_name}' not found in '{file_path}'.")

        obj = getattr(module, obj_name)
        if not isinstance(obj, cls):
            raise TypeError(f"Object '{obj_name}' in '{file_path}' is not an instance of {cls}.")
        print(f"[RECURRENT] recurrent enabled, using register '{obj_name}' from '{file_path}'.")
        return obj
