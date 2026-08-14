# Async Generation Manager (`recurrent.async_generation_manager`)

## Overview

Implements concurrent per-sample generation for high-throughput scenarios. Unlike `LLMGenerationManager` which processes batches synchronously, this manager runs one `agent.rollout()` coroutine per sample for concurrent execution.

**Status**: Currently disabled (raises `NotImplementedError` during initialization). The interface is maintained for future async support.

## Key Class: `AsyncLLMGenerationManager`

### Initialization (Currently Disabled)

```python
from recurrent.async_generation_manager import AsyncLLMGenerationManager
from recurrent.interface import AsyncRAgent, RConfig

manager = AsyncLLMGenerationManager(
    tokenizer=tokenizer,
    async_server=async_server,
    config=agent_config,
    rollout_config=rollout_config,
    agent_cls=MyAsyncAgent
)
# Raises: NotImplementedError("Async utils is not supported for now")
```

## Methods (For Future Use)

### `run_llm_loop(gen_batch, timing_raw)`

Runs async generation with concurrent sample rollout.

```python
output_batch, final_mask, sample_index = manager.run_llm_loop(gen_batch, timing_raw)
```

**Flow**:

1. `agent.start(gen_batch, timing_raw)`
2. Launch `agent.rollout()` coroutine for each sample
3. Wait for all coroutines to complete
4. Gather and concatenate results
5. `agent.end()`

## Helper Methods

### `concat_output(batch_list)`

Concatenate outputs from multiple samples.

```python
output_batch = manager.concat_output([sample1_output, sample2_output, ...])
```

Handles:

- Padding prompts and responses to uniform length
- Creating attention masks
- Computing position IDs
- Building DataProto output

### `tokenize_output(gen_output)`

Tokenize async output conversations using chat template.

```python
tokenized = manager.tokenize_output(async_output)
```

Returns dict with:

- `prompts`: Tokenized prompts
- `responses`: Tokenized responses
- `response_mask`: Boolean mask for assistant tokens

## Differences from Synchronous Manager

| Aspect      | Sync                | Async                     |
| ----------- | ------------------- | ------------------------- |
| Processing  | Batch at a time     | Per-sample coroutines     |
| Concurrency | None                | Full async concurrency    |
| Use Case    | Standard generation | High-throughput scenarios |
| Status      | Active              | Disabled                  |

## Architecture

```
run_llm_loop(batch)
├─→ agent.start()
├─→ For each sample, launch rollout() coroutine
├─→ Gather all coroutines
├─→ tokenize_output() each result
├─→ concat_output() all results
└─→ agent.end()
```

## Future Re-enablement

To re-enable async generation:

1. Remove `NotImplementedError` guard in `__init__`
2. Ensure `async_server.chat_scheduler` is initialized
3. Verify `ChatCompletionProxy` is properly configured
4. Test concurrent request handling

## Related Classes

- `AsyncRAgent`: Implement this for async agents
- `AsyncOutput`: Container for async generation results
- `ChatCompletionProxy`: Async HTTP client for remote LLMs

## See Also

- [Async Utilities](./recurrent-async-utils.md) - HTTP client and concurrency utilities
- [Recurrent Agent Interface](./recurrent-interface.md) - AsyncRAgent implementation
- [Synchronous Generation Manager](./recurrent-generation.md) - Current active generation manager
