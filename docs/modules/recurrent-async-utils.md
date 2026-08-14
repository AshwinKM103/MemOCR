# Async Utilities (`recurrent.async_utils`)

## Overview

Provides async HTTP client and event loop utilities for concurrent LLM generation against remote OpenAI-compatible servers.

## Key Classes

### `ChatCompletionProxy`

Async proxy for submitting chat completion requests with load balancing.

```python
from recurrent.async_utils import ChatCompletionProxy
from omegaconf import OmegaConf

# Configure proxy
config = OmegaConf.create({
    "api_key": "token-abc123",
    "model_name": "qwen-vl-max"
})

proxy = ChatCompletionProxy(
    config=config,
    model_path="qwen-vl-max",
    server_addresses=["localhost:8000", "localhost:8001", "localhost:8002"]
)

# Use in async context
import asyncio

async def generate():
    completion, error = await proxy.get_chat_completions(
        model="qwen-vl-max",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
        top_p=0.9,
        max_tokens=256
    )

    if error:
        print(f"Error: {error}")
    else:
        response_text = completion.choices[0].message.content
        print(f"Response: {response_text}")

asyncio.run(generate())
```

**Features**:

- Load balancing across multiple servers
- Request tracking via x-request-id header
- Configurable HTTP client with connection pooling
- Fallback between OpenAI client and aiohttp backends

**Attributes**:

- `addr_client_map`: Cache of HTTP clients per server address
- `session`: Shared aiohttp session for requests
- `model_name`: Default model name

**Methods**:

#### `get_chat_completions(model=None, **chat_complete_request)`

Submit a chat completion request with load balancing.

```python
completion, error = await proxy.get_chat_completions(
    model="qwen-vl-max",
    messages=[
        {"role": "user", "content": "What is in this image?"}
    ],
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
    extra_headers={"x-request-id": "my-request-123"}
)

if error:
    # Handle error
    print(f"Request failed: {error}")
else:
    # Use completion
    choice = completion.choices[0]
    print(choice.message.content)
```

**Parameters**:

- `model` (str | None): Model name (uses default if None)
- `**chat_complete_request`: OpenAI-compatible chat parameters
  - `messages`: List of message dicts with role and content
  - `temperature`: Sampling temperature (0 to 2)
  - `top_p`: Nucleus sampling parameter
  - `max_tokens`: Maximum response length
  - `extra_headers` (optional): Custom HTTP headers (must include x-request-id for tracking)

**Returns**:

- `(completion, error)`: ChatCompletion object or None, Exception or None
  - If successful: `(completion, None)`
  - If failed: `(None, exception)`

**Load Balancing**:
Routes requests to server with least pending requests using a min-heap.

#### `get_client(address)`

Get or create HTTP client for a server address.

```python
client = proxy.get_client("localhost:8000")
```

**Returns**: AsyncClient with timeout and connection limits configured

## Helper Functions

### `run_coroutine_in_chat_scheduler_loop(async_server, coro)`

Run a coroutine in the async server's event loop (in separate thread).

```python
from recurrent.async_utils import run_coroutine_in_chat_scheduler_loop

async def my_async_operation():
    # Async operations here
    return "result"

result = run_coroutine_in_chat_scheduler_loop(async_server, my_async_operation())
```

**Purpose**: Bridges sync and async code by submitting coroutine to scheduler thread's event loop

**Returns**: Result of the coroutine

## Usage Example: Async Agent

```python
from recurrent.interface import AsyncRAgent, AsyncOutput
from recurrent.async_utils import ChatCompletionProxy
import asyncio

class MyAsyncAgent(AsyncRAgent):
    async def rollout(self, gen_item):
        # Build prompt from item
        prompt = gen_item.meta_info["prompt"]

        # Call LLM asynchronously
        completion, error = await self.proxy.get_chat_completions(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.rollout_config.temperature,
            max_tokens=self.rollout_config.max_tokens
        )

        if error:
            # Handle error gracefully
            response_text = "[Error generating response]"
        else:
            response_text = completion.choices[0].message.content

        # Build conversation for output
        conversations = [[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response_text}
        ]]

        return AsyncOutput(
            conversations=conversations,
            sample_index=gen_item.meta_info["idx"],
            final_mask=True,
            timing_raw=self.timing_raw,
            metrics={"tokens": len(response_text.split())}
        )
```

## Configuration

```python
# In your config file (YAML)
async_generation:
  chat_completion:
    api_key: "token-abc123"
    model_name: "qwen-vl-max"
    server_addresses:
      - "localhost:8000"
      - "localhost:8001"
      - "localhost:8002"
    timeout:
      connect: 60
      read: null  # No timeout on reads
```

## Performance Considerations

1. **Connection Pooling**: Uses HTTP connection pooling for efficiency
2. **Load Balancing**: Automatically distributes requests across servers
3. **Concurrent Requests**: All requests are async (non-blocking)
4. **Keepalive**: HTTP keep-alive connections reused (600s timeout)

## Error Handling

```python
async def safe_generate(proxy, prompt):
    try:
        completion, error = await proxy.get_chat_completions(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        if error:
            print(f"API error: {error}")
            return None

        return completion.choices[0].message.content

    except asyncio.TimeoutError:
        print("Request timed out")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
```

## See Also

- [Async Generation Manager](./recurrent-async-generation.md) - Uses these utilities
- [Recurrent Agent Interface](./recurrent-interface.md) - AsyncRAgent implementation
