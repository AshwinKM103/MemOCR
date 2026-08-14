# Vision Processing (`recurrent.vision_process_utils`, `recurrent.qwen_vl_utils`)

## Overview

Provides utilities for vision-language model integration, including image preprocessing, Qwen VL-specific message processing, and image token management.

## Core Functions

### Qwen VL Integration

#### `process_rlhf_inputs()`

Process text and image inputs for Qwen VL models.

```python
from recurrent.qwen_vl_utils import process_rlhf_inputs

output = process_rlhf_inputs(
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "image": image},  # PIL Image
            {"type": "text", "text": "Describe this image."}
        ]
    }],
    images_data=[image],
    tokenizer=tokenizer,
    processor=processor,
    max_prompt_length=2048,
    truncation="error"
)

# output contains:
# - input_ids: Tokenized input with image tokens
# - attention_mask: Attention mask
# - position_ids: Position IDs
# - multi_modal_data: Processed image data
```

**Parameters**:

- `messages`: List of message dicts with role/content
- `images_data`: List of PIL Images (or None)
- `tokenizer`: HuggingFace tokenizer
- `processor`: Vision-language processor (e.g., AutoProcessor)
- `max_prompt_length`: Maximum sequence length
- `truncation`: How to handle overflow ("error", "truncate", or None)

**Returns**: Dict with tokenized and processed inputs

### Image Processing

#### Batch Image Processing

```python
from recurrent.generation_manager import batch_subsample_images
from PIL import Image

images = [Image.open(f) for f in image_files]

# Subsample to 0.25x resolution in parallel
small_images = batch_subsample_images(
    images,
    ratio=0.25,
    max_workers=8
)

# Or just check image dimensions
for img, small_img in zip(images, small_images):
    print(f"Original: {img.size}, Subsampled: {small_img.size}")
```

## Vision-Language Modes

### Mode 1: Text-Only with Context

Standard mode where images are embedded in message context.

```python
class TextOnlyAgent(RAgent):
    def action(self):
        # Generate text prompts (no images)
        prompts = [...]  # Just text
        return prompts, {"input_pad_to": 512}
```

Used with `manager.run_llm_loop()` (text-only mode).

### Mode 2: Vision-Language (Single Objective)

Standard VL mode with image-question pairs.

```python
class VLAgent(RAgent):
    def action(self):
        # Generate text prompts and images
        prompts = ["Describe this image."]
        images = [Image.open("image.jpg")]
        return prompts, images, {"input_pad_to": 2048}
```

Used with `manager.run_llm_loop_vl()`.

### Mode 3: Triple-Objective VL

Advanced mode with three training objectives:

1. **Vanilla QA**: Full-resolution image QA
2. **Subsampled QA**: 0.25x resolution image QA
3. **Gap-fill QA**: Fill-in-the-blank with full-resolution images

```python
class TripleObjectiveAgent(RAgent):
    def action(self):
        # Return text and images
        prompts = [...]
        images = [...]
        return prompts, images, {"input_pad_to": 2048}

    def end(self, triple_answer_turn=True):
        # Return multiple masks for each objective
        if triple_answer_turn:
            return (
                (final_mask, vanilla_qa_mask, subsampled_qa_mask, gap_fill_mask),
                sample_index,
                (gap_fill_questions, gap_fill_answers)
            )
        else:
            return final_mask, sample_index
```

Used with `manager.run_llm_loop_vl_triple_turn()`.

## Message Format

Messages follow OpenAI format with image support:

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": PIL_Image,
                "min_pixels": 224*224,  # Minimum resolution
                "max_pixels": 1024*1024  # Maximum resolution
            },
            {
                "type": "text",
                "text": "What do you see?"
            }
        ]
    }
]
```

## Image Token Management

The generation manager tracks image token counts:

```python
# In LLMGenerationManager.process_messages_vl()
image_token_id = 151655  # Qwen VL image token ID
image_token_counts = (model_inputs["input_ids"] == image_token_id).sum(-1)

stats = {
    "avg": image_token_counts.float().mean().item(),
    "max": image_token_counts.max().item(),
    "min": image_token_counts.min().item(),
    "median": image_token_counts.median().item(),
}
print(f"Image Tokens - Avg: {stats['avg']:.2f}, Max: {stats['max']}")
```

## Configuration

```python
# In your agent config
@dataclass
class VLAgentConfig(RConfig):
    min_image_pixels: int = 224 * 224      # Minimum image resolution
    max_image_pixels: int = 1024 * 1024    # Maximum image resolution
    image_token_id: int = 151655           # Qwen VL image token ID
    subsampling_ratio: float = 0.25        # For triple-objective mode
```

## Best Practices

1. **Image Size**: Keep images in min/max pixel range to control token counts
2. **Batching**: Process images in batches using `batch_subsample_images()`
3. **Memory**: Resize images before processing to avoid OOM
4. **Consistency**: Use same processor/tokenizer across batches
5. **Error Handling**: Handle None images gracefully (treat as text-only)

## Example: Complete VL Agent

```python
from recurrent.interface import RAgent, RConfig
from recurrent.vision_process_utils import process_rlhf_inputs
from dataclasses import dataclass
import torch

@dataclass
class MyVLConfig(RConfig):
    max_turns: int = 3
    min_image_pixels: int = 224 * 224
    max_image_pixels: int = 1024 * 1024

class MyVLAgent(RAgent):
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config

    def start(self, gen_batch, timing_raw):
        self.gen_batch = gen_batch
        self.images = gen_batch.non_tensor_batch["images"]
        self.step = 0
        self.final_mask_list = []
        self.sample_index_list = []

    def action(self):
        # Build prompts with image context
        prompts = []
        images = []
        for idx in range(len(self.gen_batch)):
            if self.step == 0:
                prompts.append("Describe this image in detail.")
            else:
                prompts.append("What else can you see?")
            images.append(self.images[idx])

        sample_index = torch.arange(len(self.gen_batch))
        self.sample_index_list.append(sample_index)

        final_mask = (self.step == self.config.max_turns - 1)
        self.final_mask_list.append(torch.full((len(self.gen_batch),), final_mask))

        return prompts, images, {
            "input_pad_to": 2048,
            "temperature": 0.7
        }

    def update(self, gen_output):
        self.step += 1
        return gen_output

    def done(self):
        return self.step >= self.config.max_turns

    def end(self):
        return torch.cat(self.final_mask_list), torch.cat(self.sample_index_list)
```

## See Also

- [Synchronous Generation Manager](./recurrent-generation.md) - VL mode integration
- [Recurrent Agent Interface](./recurrent-interface.md) - Custom agent implementation
- [Markdown to Image](./md2img.md) - Converting documents to images
