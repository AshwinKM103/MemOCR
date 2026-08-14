# Recurrent Agent Interface (`recurrent.interface`)

## Overview

Defines the core interfaces that all memory-augmented recurrent agents must implement. The recurrent agent interface enables multi-turn generation where agents iteratively generate responses, process environment feedback, and update internal state.

## Key Concepts

### Multi-Turn Generation Loop

The recurrent agent follows a loop pattern:

```python
agent.start(batch, timing_info)        # Initialize agent state
while not agent.done():
    prompts, meta = agent.action()     # Generate prompts for this turn
    llm_output = generate(prompts)     # Call LLM
    llm_output = agent.update(llm_output)  # Process output and update state
final_mask, indices = agent.end()      # Cleanup and return results
```

This allows agents to:

- Build context iteratively across turns
- Use LLM outputs to guide future prompts
- Implement search, planning, or reasoning strategies
- Track which samples have completed

## Core Classes

### `RConfig`

Base configuration dataclass for agents.

```python
from recurrent.interface import RConfig
from dataclasses import dataclass, field

@dataclass
class MyAgentConfig(RConfig):
    max_turns: int = 5
    temperature: float = 0.7
    top_p: float = 0.9
    system_prompt: str = "You are a helpful assistant."
```

**Key Points**:

- Subclass this for your agent's configuration
- All config should flow through this class
- Use dataclass `field()` for complex defaults

### `RAgent` (Abstract Base Class)

Abstract interface for synchronous (single-batch) agents.

```python
from recurrent.interface import RAgent, RConfig
from transformers import PreTrainedTokenizer
from verl import DataProto

class MyAgent(RAgent):
    def __init__(self, tokenizer: PreTrainedTokenizer, config: RConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.state = {}

    def start(self, gen_batch: DataProto, timing_raw: dict[str, float]):
        """Initialize agent for this batch."""
        self.gen_batch = gen_batch
        self.timing_raw = timing_raw
        self.step = 0
        self.final_mask_list = []
        self.sample_index_list = []
        # Custom initialization
        for idx in range(len(gen_batch)):
            self.state[idx] = {"history": []}

    def action(self) -> tuple[list[torch.Tensor], dict[str, Any]]:
        """Generate prompts for current turn."""
        prompts = []
        for idx in range(len(self.gen_batch)):
            # Build prompt using current state
            history = self.state[idx]["history"]
            prompt = format_prompt(history)
            prompt_ids = self.tokenizer.encode(prompt)
            prompts.append(torch.tensor(prompt_ids))

        # Track sample indices and termination
        sample_index = torch.arange(len(self.gen_batch))
        self.sample_index_list.append(sample_index)
        final_mask = torch.zeros(len(self.gen_batch), dtype=torch.bool)
        self.final_mask_list.append(final_mask)

        # Return prompts and metadata
        meta_info = {
            "input_pad_to": 512,  # Max sequence length
            "do_sample": True,
            "temperature": self.config.temperature,
        }
        return prompts, meta_info

    def update(self, gen_output: DataProto) -> DataProto:
        """Process LLM output and update agent state."""
        responses = gen_output.batch["responses"]
        for idx, response in enumerate(responses):
            # Decode response
            response_text = self.tokenizer.decode(response)
            # Update state
            self.state[idx]["history"].append(response_text)
            # Could also execute tools, compute rewards, etc.

        self.step += 1
        return gen_output

    def done(self) -> bool:
        """Check if generation should stop."""
        return self.step >= self.config.max_turns

    def end(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Cleanup and return results."""
        # Concatenate indices and masks across all turns
        final_mask = torch.cat(self.final_mask_list)
        sample_index = torch.cat(self.sample_index_list)

        # Cleanup
        del self.gen_batch
        del self.state

        return final_mask, sample_index
```

**Methods**:

- **`__init__(tokenizer, config)`**: Initialize agent. Store tokenizer and config as instance variables.
- **`start(gen_batch, timing_raw)`**: Called once at beginning. Initialize internal state, set `self.step = 0`, initialize `self.final_mask_list` and `self.sample_index_list`.
- **`action()`**: Generate prompts for current turn. Must:
  - Return list of input_ids (one per sample)
  - Append sample_index and final_mask to internal lists
  - Return (input_ids, meta_info) where meta_info has `input_pad_to` key
- **`update(gen_output)`**: Process LLM output. Can execute tools, update state, return modified output.
- **`done()`**: Return True when generation should stop (checked before each turn).
- **`end()`**: Cleanup. Concatenate and return `(final_mask, sample_index)` tensors.

### `AsyncRAgent` (Abstract Base Class)

Async interface for concurrent per-sample generation.

**Note**: Currently disabled. See `AsyncLLMGenerationManager` for details.

```python
from recurrent.interface import AsyncRAgent, AsyncOutput
import asyncio

class MyAsyncAgent(AsyncRAgent):
    async def rollout(self, gen_item: DataProtoItem) -> AsyncOutput:
        """Rollout a single sample asynchronously."""
        # Build prompt
        prompt = build_prompt(gen_item)

        # Call LLM asynchronously
        completion = await self.proxy.get_chat_completions(
            messages=[{"role": "user", "content": prompt}]
        )

        # Build conversation
        conversations = [[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion.choices[0].message.content}
        ]]

        # Return async output
        return AsyncOutput(
            conversations=conversations,
            sample_index=gen_item.meta_info["idx"],
            final_mask=True,
            timing_raw=self.timing_raw,
            metrics={"tokens": len(completion.choices[0].message.content)}
        )
```

### `AsyncOutput`

Container for async LLM generation results.

```python
class AsyncOutput:
    def __init__(
        self,
        conversations: list[list[dict]],  # Multi-turn conversation
        sample_index: int,                 # Index in batch
        final_mask: bool,                  # Is this the final turn?
        timing_raw: dict[str, float],      # Timing information
        metrics: dict[str, Any] | None = None
    ):
        """Initialize async output with results."""
```

### `RDataset`

Dataset interface for recurrent RL training.

```python
from recurrent.interface import RDataset, RConfig
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

class MyDataset(RDataset):
    def __init__(self, config: RConfig, data_files: str, tokenizer: PreTrainedTokenizer, data_config: DictConfig):
        super().__init__(config, data_files, tokenizer, data_config)
        # Custom initialization

    def __getitem__(self, idx) -> dict:
        # Get base sample (includes sample_uuid)
        sample = super().__getitem__(idx)
        # Add custom fields
        sample["custom_field"] = "value"
        return sample

    def get_batch_keys(self) -> tuple[list[str], list[str]]:
        # Return (tensor_keys, non_tensor_keys)
        return ["input_ids", "attention_mask", "position_ids"], ["sample_uuid"]
```

### `RRegister`

Registry for loading custom agent implementations.

```python
from recurrent.interface import RRegister, RConfig, RAgent, RDataset

# In your custom_agent.py file:
register = RRegister(
    config_cls=MyAgentConfig,
    dataset_cls=MyDataset,
    agent_cls=MyAgent
)

# In your training code:
register = RRegister.from_filename("custom_agent.py", "register")
agent = register.agent_cls(tokenizer, register.config_cls())
dataset = register.dataset_cls(register.config_cls(), "data.parquet", tokenizer, config)
```

## Creating a Custom Agent

### Step 1: Define Configuration

```python
from dataclasses import dataclass
from recurrent.interface import RConfig

@dataclass
class SearchAgentConfig(RConfig):
    max_turns: int = 5
    max_retries: int = 3
    search_backend: str = "google"
```

### Step 2: Implement Agent Class

```python
from recurrent.interface import RAgent
from transformers import PreTrainedTokenizer
from verl import DataProto
import torch
from typing import Any

class SearchAgent(RAgent):
    def __init__(self, tokenizer: PreTrainedTokenizer, config: SearchAgentConfig):
        self.tokenizer = tokenizer
        self.config = config
        # Tools
        self.search_tool = GoogleSearchTool()

    def start(self, gen_batch: DataProto, timing_raw: dict[str, float]):
        self.gen_batch = gen_batch
        self.timing_raw = timing_raw
        self.step = 0
        self.state = {
            i: {"search_queries": [], "search_results": []}
            for i in range(len(gen_batch))
        }
        self.final_mask_list = []
        self.sample_index_list = []

    def action(self) -> tuple[list[torch.Tensor], dict[str, Any]]:
        # Build search prompts
        prompts = []
        for idx in range(len(self.gen_batch)):
            query_template = "Based on the question '{q}', what search queries would help? List one per line."
            question = self.gen_batch.batch["questions"][idx]  # Example field
            prompt = query_template.format(q=question)
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")[0]
            prompts.append(input_ids)

        sample_index = torch.arange(len(self.gen_batch))
        self.sample_index_list.append(sample_index)
        self.final_mask_list.append(torch.zeros(len(self.gen_batch), dtype=torch.bool))

        return prompts, {"input_pad_to": 256, "temperature": 0.1}

    def update(self, gen_output: DataProto) -> DataProto:
        # Execute searches
        responses = gen_output.batch["responses"]
        for idx, response in enumerate(responses):
            response_text = self.tokenizer.decode(response, skip_special_tokens=True)
            queries = response_text.split("\n")
            self.state[idx]["search_queries"] = queries

            # Actually search
            results = []
            for query in queries:
                result = self.search_tool.search(query)
                results.append(result)
            self.state[idx]["search_results"] = results

        self.step += 1
        return gen_output

    def done(self) -> bool:
        return self.step >= self.config.max_turns

    def end(self) -> tuple[torch.Tensor, torch.Tensor]:
        final_mask = torch.cat(self.final_mask_list)
        sample_index = torch.cat(self.sample_index_list)
        del self.gen_batch, self.state
        return final_mask, sample_index
```

### Step 3: Create Register and Use

```python
register = RRegister(
    config_cls=SearchAgentConfig,
    dataset_cls=MyDataset,
    agent_cls=SearchAgent
)

# In training code:
from recurrent.generation_manager import LLMGenerationManager

manager = LLMGenerationManager(
    tokenizer=tokenizer,
    actor_rollout_wg=ray_actors,  # Ray actor group
    config=register.config_cls(),
    agent_cls=register.agent_cls
)

output_batch, final_mask, sample_index = manager.run_llm_loop(gen_batch, timing_raw)
```

## Best Practices

1. **State Management**: Keep agent state lightweight. Delete large objects in `end()`.
2. **Masking**: Always track `final_mask` to distinguish actual outputs from padding.
3. **Indexing**: Track `sample_index` to align outputs with original batch.
4. **Timing**: Use timing_raw dict to profile each turn (used for monitoring).
5. **Determinism**: Use `do_sample=False` for reproducible results during validation.

## See Also

- [Synchronous Generation Manager](./recurrent-generation.md) - Orchestrates the multi-turn loop
- [Utility Functions](./recurrent-utils.md) - Helper functions for tokens and templates
- [Vision Processing](./vision-processing.md) - VL model integration
