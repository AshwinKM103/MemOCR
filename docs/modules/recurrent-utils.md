# Utility Functions (`recurrent.utils`)

## Overview

Provides essential utilities for tensor operations, template formatting, and batch processing in recurrent generation pipelines.

## Token Template

### `TokenTemplate` Class

Template formatter that works with token IDs instead of text strings.

```python
from recurrent.utils import TokenTemplate
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("qwen/qwen-72b")

# Define template with placeholders
template_str = "Question: {question}\nContext: {context}\nAnswer: "
template = TokenTemplate(template_str, tokenizer)

# Format with token IDs
question_ids = tokenizer.encode("What is the capital of France?")
context_ids = tokenizer.encode("France is a country in Western Europe.")

output_ids = template.format(
    question=question_ids,
    context=context_ids
)

# Decode to verify
output_text = tokenizer.decode(output_ids)
print(output_text)
# Output: Question: What is the capital of France?
#         Context: France is a country in Western Europe.
#         Answer:
```

**Benefits**:

- Works directly with token sequences, no re-encoding
- Useful for building prompts where parts have known token lengths
- Faster than text-based templating for large-scale generation

**Methods**:

#### `__init__(template, tokenizer=None)`

```python
# Initialize with lazy initialization
template = TokenTemplate("Hello {name}, you have {count} messages.")

# Initialize with immediate tokenization
template = TokenTemplate("Hello {name}", tokenizer)
```

#### `init(tokenizer)`

```python
# Initialize template structure by tokenizing sections
template.init(tokenizer)
```

#### `format(**kwargs) -> torch.Tensor`

```python
output = template.format(
    question=[1, 2, 3],  # list of ints
    context=torch.tensor([4, 5, 6]),  # tensor
    answer=np.array([7, 8, 9])  # numpy array
)
# Returns concatenated tensor
```

#### `length` (property)

```python
total_tokens = template.length  # Total tokens in template
```

## Tensor Operations

### Chat Template Extraction

```python
from recurrent.utils import chat_template

# Get user-only template
template_str = chat_template(tokenizer, system=False)
# Returns: "<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"

# Get system + user template
template_str = chat_template(tokenizer, system=True)
# Returns: "<|im_start|>system\n{system}<|im_end|>\n..."
```

### Padding

#### `pad_tensor_list_to_length()`

Pad list of 1D tensors to same length.

```python
from recurrent.utils import pad_tensor_list_to_length
import torch

token_lists = [
    torch.tensor([1, 2, 3]),
    torch.tensor([4, 5, 6, 7, 8]),
    torch.tensor([9])
]

# Left-pad to max length
padded = pad_tensor_list_to_length(
    token_lists,
    pad_token_id=0,
    max_length=8,
    left_pad=True
)
# Result: [[0, 0, 0, 0, 0, 1, 2, 3],
#          [0, 0, 0, 4, 5, 6, 7, 8],
#          [0, 0, 0, 0, 0, 0, 0, 9]]

# Get attention mask too
padded, mask = pad_tensor_list_to_length(
    token_lists,
    pad_token_id=0,
    return_mask=True,
    left_pad=True
)
# mask: [[False, False, False, False, False, True, True, True],
#        [False, False, False, True, True, True, True, True],
#        [False, False, False, False, False, False, False, True]]
```

**Parameters**:

- `response`: List of 1D tensors to pad
- `pad_token_id`: Token ID for padding (typically 0)
- `max_length`: Target length (uses longest if None)
- `left_pad`: Pad on left (True) or right (False)
- `return_mask`: Return attention mask alongside padded tensor

**Performance**: 20x faster than `verl.utils.torch_functional.pad_2d_list_to_length`

#### `unpad()`

Remove padding tokens from tensor.

```python
from recurrent.utils import unpad

padded_tensor = torch.tensor([
    [0, 0, 1, 2, 3],
    [0, 4, 5, 0, 0],
    [6, 7, 8, 9, 10]
])

unpadded = unpad(
    tokenizer,
    padded_tensor,
    remove_eos=False
)
# Result: object array of unpadded sequences
# unpadded[0] = tensor([1, 2, 3])
# unpadded[1] = tensor([4, 5])
# unpadded[2] = tensor([6, 7, 8, 9, 10])
```

#### `graceful_padding()`

Calculate padding indices for batch divisibility by world_size.

```python
from recurrent.utils import graceful_padding

bsz = 7
world_size = 3

padding_index, no_padding_mask = graceful_padding(bsz, world_size)
# padding_index: indices for selecting samples with padding
# no_padding_mask: boolean mask for real samples
```

Used internally by LLMGenerationManager for distributed generation.

### Attention and Position

#### `create_attention_mask()`

```python
from recurrent.utils import create_attention_mask

input_ids = torch.tensor([
    [101, 2054, 2003, 0, 0],
    [101, 2054, 102, 0, 0]
])

mask = create_attention_mask(input_ids, pad_token_id=0)
# Result: [[1, 1, 1, 0, 0],
#          [1, 1, 1, 0, 0]]
```

**Returns**: 1 for valid tokens, 0 for padding

#### `create_position_ids()`

```python
from recurrent.utils import create_position_ids

attention_mask = torch.tensor([
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0]
])

position_ids = create_position_ids(attention_mask)
# Result: [[0, 1, 2, 0, 0],
#          [0, 1, 2, 0, 0]]
```

**Note**: Positions are 0-indexed and reset to 0 after padding

## DataProto Operations

### `indexing_proto()`

Index a DataProto batch by selecting samples.

```python
from recurrent.utils import indexing_proto
from verl import DataProto

original_batch = DataProto.from_dict(
    tensors={"ids": torch.arange(10)},
    non_tensors={"texts": np.array([f"text_{i}" for i in range(10)])},
    meta_info={"batch_size": 10}
)

# Select first 5 samples
subset = indexing_proto(original_batch, torch.tensor([0, 1, 2, 3, 4]))

# Select by boolean mask
mask = torch.tensor([True, False, True, False, True, False, False, False, True, False])
subset = indexing_proto(original_batch, mask)
```

**Supports**:

- torch.Tensor indices
- List of ints
- NumPy array indices
- Boolean masks

## Utility Functions

### `now()`

Get current timestamp.

```python
from recurrent.utils import now

timestamp = now()
# Returns: "2025-08-14 13:30:45"
```

## Common Patterns

### Building Multi-Turn Prompts

```python
from recurrent.utils import TokenTemplate, chat_template

template_str = chat_template(tokenizer, system=False)
# template_str = "<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"

template = TokenTemplate(template_str, tokenizer)

# Build prompts for each turn
for turn in range(num_turns):
    turn_messages = [...]  # List of strings
    turn_ids = [tokenizer.encode(msg) for msg in turn_messages]

    prompts = [template.format(message=ids) for ids in turn_ids]
    # Now dispatch prompts to LLM
```

### Handling Variable-Length Sequences

```python
from recurrent.utils import pad_tensor_list_to_length, unpad

# Collect responses from LLM
responses = [...]  # List of variable-length tensors

# Pad for batch processing
padded, mask = pad_tensor_list_to_length(
    responses,
    pad_token_id=tokenizer.pad_token_id,
    return_mask=True,
    left_pad=False
)

# Process in batches
processed = model(padded, attention_mask=mask)

# Unpad for storing individual results
unpadded = unpad(tokenizer, padded)
```

### Creating Input/Output Pairs

```python
from recurrent.utils import create_attention_mask, create_position_ids

# Build input and output sequences
input_ids = torch.cat([prompt_ids, response_ids], dim=1)

# Create supporting tensors
attention_mask = create_attention_mask(input_ids, pad_token_id=0)
position_ids = create_position_ids(attention_mask)

# Ready for training
batch = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "position_ids": position_ids,
    "labels": input_ids.clone()  # Set some positions to -100 for loss masking
}
```

## Performance Notes

- `TokenTemplate.format()` avoids re-tokenization, making it faster than string templating
- `pad_tensor_list_to_length()` is optimized for speed (uses direct concatenation)
- `graceful_padding()` is called during generation but cached between batches

## See Also

- [Recurrent Agent Interface](./recurrent-interface.md) - How these utilities are used in agents
- [Synchronous Generation Manager](./recurrent-generation.md) - How padding/masking are used
- [Vision Processing](./vision-processing.md) - Image-specific utilities
