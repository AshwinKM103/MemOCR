# Synchronous Generation Manager (`recurrent.generation_manager`)

## Overview

Implements the main multi-turn generation loop for recurrent agents. The `LLMGenerationManager` orchestrates generation by iteratively calling `agent.action()`, dispatching batches to Ray rollout workers for LLM inference, and calling `agent.update()` to process outputs.

## Key Class: `LLMGenerationManager`

### Initialization

```python
from recurrent.generation_manager import LLMGenerationManager
from recurrent.interface import RConfig, RAgent
from transformers import PreTrainedTokenizer

manager = LLMGenerationManager(
    tokenizer=tokenizer,  # HuggingFace tokenizer
    actor_rollout_wg=ray_actors,  # Ray actor group for generation
    config=agent_config,  # RConfig instance
    agent_cls=MyAgent,  # RAgent subclass
    processor=processor  # Optional: for VL models
)
```

**Parameters**:

- `tokenizer` (PreTrainedTokenizer): Tokenizer for encoding/decoding
- `actor_rollout_wg`: Ray actor group managing distributed generation
- `config` (RConfig): Agent configuration
- `agent_cls` (type[RAgent]): Agent class to instantiate
- `processor` (ProcessorMixin | None): Multimodal processor for VL models

### Main Methods

#### `run_llm_loop(gen_batch, timing_raw)`

Runs the main multi-turn generation loop.

```python
output_batch, final_mask, sample_index = manager.run_llm_loop(gen_batch, timing_raw)
```

**Args**:

- `gen_batch` (DataProto): Batch with context (typically contains "context_ids", "context_length", "prompt_ids")
- `timing_raw` (dict[str, float]): Timing dictionary to accumulate profiling data

**Returns**:

- `output_batch` (DataProto): Concatenated LLM outputs from all turns
- `final_mask` (torch.Tensor): Boolean mask indicating which samples are final
- `sample_index` (torch.Tensor): Integer indices for tracking samples

**Flow**:

```
1. agent.start(gen_batch, timing_raw)
2. For each turn:
   a. agent.action() → get prompts
   b. generate_with_graceful_padding() → LLM inference
   c. agent.update(output) → process output
   d. Check agent.done()
3. agent.end() → cleanup and return masks
```

**Example**:

```python
# Prepare batch
gen_batch = DataProto.from_dict(
    tensors={
        "context_ids": context_ids,
        "prompt_ids": prompt_ids,
    },
    meta_info={"do_sample": True}
)

timing_raw = {}
output_batch, final_mask, sample_index = manager.run_llm_loop(gen_batch, timing_raw)

# Access results
responses = output_batch.batch["responses"]
print(f"Generated {len(responses)} responses")
print(f"Timing: {timing_raw}")
```

#### `run_llm_loop_vl(gen_batch, timing_raw)`

Vision-language version of the generation loop.

```python
output_batch, final_mask, sample_index = manager.run_llm_loop_vl(gen_batch, timing_raw)
```

Works like `run_llm_loop()` but:

- Agent's `action()` returns `(text_messages, images, meta_info)`
- Processes images through the VL processor
- Handles image/text alignment

**Requirements**: `processor` must be provided during initialization

#### `run_llm_loop_vl_triple_turn(gen_batch, timing_raw)`

Advanced VL mode with three training objectives.

```python
output_batch, masks, sample_index, answers = manager.run_llm_loop_vl_triple_turn(gen_batch, timing_raw)
final_mask, vanilla_qa_mask, subsampled_qa_mask, gap_fill_mask = masks
```

Generates three objectives:

1. **Vanilla QA**: Question-answering with full-resolution images
2. **Subsampled QA**: QA with images downsampled to 0.25x resolution
3. **Gap-fill QA**: Fill-in-the-blank questions with full-resolution images

Returns:

- `output_batch`: Concatenated outputs from all three objectives
- `masks`: Tuple of (final_mask, vanilla_qa_mask, subsampled_qa_mask, gap_fill_mask)
- `sample_index`: Sample indices
- `answers`: Gap-fill answer texts

## Helper Functions

### `batch_subsample_images(images, ratio=0.5, max_workers=10)`

Downsample images in parallel using thread pool.

```python
from recurrent.generation_manager import batch_subsample_images

images = [Image.open(f) for f in image_paths]
downsampled = batch_subsample_images(images, ratio=0.25, max_workers=8)
```

**Args**:

- `images` (list[PIL.Image]): Images to subsample
- `ratio` (float): Downsampling ratio (0 to 1)
- `max_workers` (int): Number of worker threads

**Returns**:

- Subsampled images in the same order as input

### `collate_fn(data_list)`

Collate mixed tensor/non-tensor data into batches.

```python
tensor_batch, non_tensor_batch = collate_fn(data_list)
```

**Args**:

- `data_list` (list[dict]): List of sample dictionaries

**Returns**:

- `tensor_batch` (dict): Stacked tensors
- `non_tensor_batch` (dict): Object arrays of non-tensors

## Important Details

### Graceful Padding

The manager pads batches to be divisible by `world_size` using "Hello" token padding. This ensures efficient distributed generation. The padding is automatically removed from outputs.

### Timing Dictionary

The `timing_raw` dict tracks performance:

- `mt_prepare`: Time to prepare prompts
- `mt_gen`: Time for LLM generation
- `mt_update`: Time for agent.update()
- `mt_engine`: Engine startup/shutdown time

**Example**:

```python
timing_raw = {}
output, final_mask, sample_index = manager.run_llm_loop(batch, timing_raw)
print(f"Generation latency: {timing_raw['mt_gen']:.2f}s")
print(f"Agent update latency: {timing_raw['mt_update']:.2f}s")
```

### Sample Tracking

The manager tracks samples across turns using:

- `sample_index`: Original sample ID in batch
- `final_mask`: Boolean indicating if this turn's output is final

This allows finding the final output for each original sample:

```python
final_outputs = output_batch[final_mask]
for idx, out in zip(sample_index[final_mask], final_outputs):
    print(f"Sample {idx}: {tokenizer.decode(out)}")
```

## Vision-Language Support

When using VL models, the agent's `action()` method must return text messages and images:

```python
class MyVLAgent(RAgent):
    def action(self):
        # Return text prompts and images
        prompts = []
        images = []
        for idx in range(len(self.gen_batch)):
            prompts.append(f"Describe this image: {self.context[idx]}")
            images.append(self.image_data[idx])  # PIL Image or None

        return prompts, images, {"input_pad_to": 4096}
```

The manager handles:

- Converting prompts and images to model inputs
- Applying VL processor (tokenization + image encoding)
- Tracking image token counts
- Padding/unpadding for distributed inference

## See Also

- [Recurrent Agent Interface](./recurrent-interface.md) - Agent implementation guide
- [Utility Functions](./recurrent-utils.md) - Helper functions for tensors
- [Vision Processing](./vision-processing.md) - VL model integration details
