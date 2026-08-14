# API Reference

Complete API reference for MemOCR recurrent agent framework.

## Core Interfaces (`recurrent.interface`)

### Classes

#### `RConfig`

Base configuration dataclass for agents.

- **Subclass this** for your agent configuration
- Use `@dataclass` decorator
- All config flows through this class

```python
from recurrent.interface import RConfig
from dataclasses import dataclass

@dataclass
class MyConfig(RConfig):
    max_turns: int = 5
```

#### `RAgent` (Abstract)

Abstract base class for synchronous agents.

- **Methods to implement**:
  - `__init__(tokenizer, config)`: Initialize agent
  - `start(gen_batch, timing_raw)`: Initialize for batch
  - `action() -> (list[torch.Tensor], dict)`: Generate prompts
  - `update(gen_output) -> DataProto`: Process LLM output
  - `done() -> bool`: Check if generation should stop
  - `end() -> (torch.Tensor, torch.Tensor)`: Cleanup and return results

#### `AsyncRAgent` (Abstract)

Abstract base class for asynchronous per-sample agents.

- **Methods to implement**:
  - `__init__(proxy, tokenizer, config, rollout_config)`: Initialize
  - `rollout(gen_item) -> AsyncOutput`: Async rollout for single sample
  - `start(gen_batch, timing_raw)`: Initialize (optional)
  - `end()`: Cleanup (optional)

#### `RDataset`

Dataset interface for recurrent RL training.

- **Extends**: `RLHFDataset`
- **Methods to override**:
  - `__init__(recurrent_config, data_files, tokenizer, data_config, processor=None)`
  - `__getitem__(item) -> dict`: Override to customize per-sample processing
  - `get_batch_keys() -> (list[str], list[str])`: Return (tensor_keys, non_tensor_keys)
  - `get_collate_fn()`: Return collation function

#### `RRegister`

Registry for loading custom agent implementations.

```python
register = RRegister(
    config_cls=MyConfig,
    dataset_cls=MyDataset,
    agent_cls=MyAgent
)

# Load from file
register = RRegister.from_filename("agent.py", "register")
```

**Class Methods**:

- `from_filename(file_path, obj_name) -> RRegister`: Load from Python file

#### `AsyncOutput`

Container for async generation results.

```python
AsyncOutput(
    conversations=list[list[dict]],  # Multi-turn conversation
    sample_index=int,                 # Index in batch
    final_mask=bool,                  # Is final turn?
    timing_raw=dict[str, float],      # Timing info
    metrics=dict[str, Any]            # Metrics
)
```

---

## Generation Managers

### `LLMGenerationManager` (`recurrent.generation_manager`)

Orchestrates multi-turn generation.

**Initialization**:

```python
LLMGenerationManager(
    tokenizer: PreTrainedTokenizer,
    actor_rollout_wg: Any,
    config: RConfig,
    agent_cls: type[RAgent],
    processor: ProcessorMixin | None = None
)
```

**Methods**:

- `run_llm_loop(gen_batch, timing_raw) -> (DataProto, torch.Tensor, torch.Tensor)`: Run text-only generation
- `run_llm_loop_vl(gen_batch, timing_raw) -> (DataProto, torch.Tensor, torch.Tensor)`: Run VL generation
- `run_llm_loop_vl_triple_turn(gen_batch, timing_raw) -> (DataProto, tuple, torch.Tensor, list)`: Run triple-objective VL

**Helper Methods**:

- `get_paddings(shape) -> (torch.Tensor, torch.Tensor, torch.Tensor)`: Get padding tensors
- `get_paddings_vl(model_inputs) -> (torch.Tensor, torch.Tensor, torch.Tensor)`: Get VL padding
- `generate_with_graceful_padding(input_ids, attention_masks, position_ids, meta_info) -> DataProto`: Generate with padding
- `process_messages_vl(prompts, images, meta_info_gen) -> (dict, dict)`: Process VL messages

### `AsyncLLMGenerationManager` (`recurrent.async_generation_manager`)

Concurrent per-sample generation (currently disabled).

**Status**: Raises `NotImplementedError`

**Methods** (for future use):

- `run_llm_loop(gen_batch, timing_raw) -> (DataProto, torch.Tensor, torch.Tensor)`
- `concat_output(batch_list) -> DataProto`
- `tokenize_output(gen_output) -> dict[str, np.ndarray]`

---

## Utilities

### Tensor Operations (`recurrent.utils`)

#### `TokenTemplate`

Template formatting with token IDs.

```python
template = TokenTemplate(template_str, tokenizer)
output = template.format(keyword=token_ids)

# Properties
length = template.length  # Total tokens
```

**Methods**:

- `__init__(template, tokenizer=None)`
- `init(tokenizer)`
- `format(**kwargs) -> torch.Tensor`

#### Functions

- `chat_template(tokenizer, system=False) -> str`: Get chat template
- `pad_tensor_list_to_length(response, pad_token_id, max_length=None, left_pad=True, return_mask=False) -> torch.Tensor | tuple`: Pad tensor list
- `unpad(tokenizer, tensor, remove_eos=False) -> np.ndarray`: Remove padding
- `create_attention_mask(input_ids, pad_token_id) -> torch.Tensor`: Create attention mask
- `create_position_ids(attention_mask) -> torch.Tensor`: Create position IDs
- `graceful_padding(bsz, group_nums) -> (torch.Tensor, torch.Tensor)`: Calculate graceful padding
- `indexing_proto(proto, indices) -> DataProto`: Index DataProto by mask/indices
- `now() -> str`: Get timestamp

### Async Utilities (`recurrent.async_utils`)

#### `ChatCompletionProxy`

Async HTTP client for LLM API calls.

```python
proxy = ChatCompletionProxy(config, model_path, server_addresses)
completion, error = await proxy.get_chat_completions(
    model="qwen-vl",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=256
)
```

**Methods**:

- `__init__(config, model_path, server_addresses, max_cache_size=10000)`
- `get_client(address) -> AsyncClient`
- `get_chat_completions(model=None, **kwargs) -> (ChatCompletion | None, Exception | None)`
- `_chat_completions_openai(address, **kwargs) -> ChatCompletion` (internal)
- `_chat_completions_aiohttp(address, **kwargs) -> ChatCompletion` (internal)

#### Functions

- `run_coroutine_in_chat_scheduler_loop(async_server, coro) -> Any`: Run coroutine in scheduler thread

### Vision Processing (`recurrent.qwen_vl_utils`, `recurrent.vision_process_utils`)

#### Functions

- `process_rlhf_inputs(messages, images_data, tokenizer, processor, max_prompt_length, truncation) -> dict`: Process VL inputs
- `batch_subsample_images(images, ratio=0.5, max_workers=10) -> list[Image]`: Subsample images in parallel

---

## Helper Functions

### Batch Processing (`recurrent.generation_manager`)

- `batch_subsample_images(images, ratio=0.5, max_workers=10) -> list[Image]`: Parallel image downsampling
- `collate_fn(data_list) -> (dict, dict)`: Collate mixed tensor/non-tensor data

---

## Data Structures

### `DataProto` (from `verl`)

Main data container for batches.

```python
batch = DataProto.from_dict(
    tensors={"input_ids": torch.Tensor, ...},
    non_tensors={"text": np.array([...]), ...},
    meta_info={"key": "value", ...}
)

# Access
ids = batch.batch["input_ids"]
texts = batch.non_tensor_batch["text"]
meta = batch.meta_info
```

### `AsyncOutput`

Output container for async generation.

```python
AsyncOutput(
    conversations: list[list[dict]],
    sample_index: int,
    final_mask: bool,
    timing_raw: dict[str, float],
    metrics: dict[str, Any] | None = None
)
```

---

## Configuration Examples

### Text-Only Agent Config

```python
from dataclasses import dataclass
from recurrent.interface import RConfig

@dataclass
class TextOnlyConfig(RConfig):
    max_turns: int = 5
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
```

### Vision-Language Agent Config

```python
@dataclass
class VLAgentConfig(RConfig):
    max_turns: int = 3
    min_image_pixels: int = 224 * 224
    max_image_pixels: int = 1024 * 1024
    temperature: float = 0.5
```

### Rollout Config (OmegaConf)

```yaml
generation:
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  max_tokens: 256
  do_sample: true

validation:
  temperature: 0.0
  top_p: 1.0
  do_sample: false
```

---

## Import Quick Reference

```python
# Core interfaces
from recurrent.interface import RConfig, RAgent, AsyncRAgent, RDataset, RRegister, AsyncOutput

# Generation managers
from recurrent.generation_manager import LLMGenerationManager, batch_subsample_images, collate_fn
from recurrent.async_generation_manager import AsyncLLMGenerationManager

# Utilities
from recurrent.utils import (
    TokenTemplate, chat_template, pad_tensor_list_to_length, create_attention_mask,
    create_position_ids, graceful_padding, indexing_proto
)

# Async utilities
from recurrent.async_utils import ChatCompletionProxy, run_coroutine_in_chat_scheduler_loop

# Vision processing
from recurrent.qwen_vl_utils import process_rlhf_inputs
from recurrent.vision_process_utils import batch_subsample_images

# Markdown to image
from md2img.markdown_api_server import MarkdownRenderer
from md2img.html_api_server import HTMLRenderer
```

---

## See Also

- [Module Documentation](./modules/) - In-depth guides for each module
- [Recurrent Interface Guide](./modules/recurrent-interface.md) - Complete agent implementation tutorial
- [Generation Manager Guide](./modules/recurrent-generation.md) - Generation loop details
