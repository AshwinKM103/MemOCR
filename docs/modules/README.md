# MemOCR Module Documentation

This directory contains comprehensive documentation for all public modules in the MemOCR project.

## Core Modules

### [Recurrent Agent Interface](./recurrent-interface.md)

- **Module**: `recurrent.interface`
- **Purpose**: Defines the core interfaces for memory-augmented agents
- **Key Classes**: RConfig, RAgent, AsyncRAgent, RDataset, RRegister
- **Use When**: Implementing custom multi-turn agents or extending existing ones

### [Synchronous Generation Manager](./recurrent-generation.md)

- **Module**: `recurrent.generation_manager`
- **Purpose**: Orchestrates multi-turn LLM generation loops
- **Key Classes**: LLMGenerationManager
- **Use When**: Running text-only or vision-language generation at scale

### [Async Generation Manager](./recurrent-async-generation.md)

- **Module**: `recurrent.async_generation_manager`
- **Purpose**: Concurrent generation for high-throughput scenarios
- **Key Classes**: AsyncLLMGenerationManager
- **Use When**: Implementing async/concurrent agent rollout (currently disabled)

### [Utility Functions](./recurrent-utils.md)

- **Module**: `recurrent.utils`
- **Purpose**: Helper functions for tensor operations and template formatting
- **Key Classes**: TokenTemplate
- **Key Functions**: chat_template, pad_tensor_list_to_length, create_attention_mask, etc.
- **Use When**: Building custom generation loops or processing batches

### [Async Utilities](./recurrent-async-utils.md)

- **Module**: `recurrent.async_utils`
- **Purpose**: Async HTTP client and scheduler utilities for concurrent generation
- **Key Classes**: ChatCompletionProxy
- **Key Functions**: run_coroutine_in_chat_scheduler_loop
- **Use When**: Running async LLM requests against remote servers

### [Vision Processing](./vision-processing.md)

- **Module**: `recurrent.vision_process_utils`, `recurrent.qwen_vl_utils`
- **Purpose**: Image preprocessing and vision-language model integration
- **Use When**: Building VL agents or processing images for LLMs

### [Markdown to Image Rendering](./md2img.md)

- **Module**: `md2img` (subpackage)
- **Purpose**: Convert markdown/HTML to rendered images
- **Key Modules**: markdown_api_server, html_api_server
- **Use When**: Converting documents to visual format for vision models

## Getting Started

### For Beginners

1. Start with [Recurrent Agent Interface](./recurrent-interface.md) to understand the core abstractions
2. Read [Utility Functions](./recurrent-utils.md) for common patterns
3. Implement a simple agent using RAgent

### For Advanced Users

1. Explore [Synchronous Generation Manager](./recurrent-generation.md) to understand the generation loop
2. Study [Vision Processing](./vision-processing.md) for VL model integration
3. Dive into specific utility functions in [Async Utilities](./recurrent-async-utils.md) if needed

## Architecture Overview

```
MemOCR Recurrent Agent Loop
=============================

1. Initialize Agent (RAgent.__init__)
   ↓
2. Start Generation (RAgent.start)
   ↓
3. Multi-turn Loop:
   ├─→ action() → generate prompts/input_ids
   ├─→ dispatch to LLM (actor_rollout_wg)
   ├─→ update() → process LLM output
   ├─→ done() → check termination
   └─→ repeat if not done
   ↓
4. End Generation (RAgent.end)
   ↓
5. Return final masks and sample indices
```

The LLMGenerationManager handles steps 3-4, calling your agent's methods at each stage.

## Common Patterns

### Creating a Custom Agent

See [Recurrent Agent Interface](./recurrent-interface.md#creating-a-custom-agent)

### Building Memory-Augmented Prompts

See [Utility Functions](./recurrent-utils.md#token-template) for TokenTemplate examples

### Processing Vision-Language Inputs

See [Vision Processing](./vision-processing.md)

## Dependencies

- **Core**: torch, transformers, tensordict, omegaconf
- **Vision**: PIL, qwen-vl (Qwen VL integration)
- **Async**: aiohttp, httpx, openai
- **Rendering**: md2img (custom module for markdown/HTML rendering)

## Related Files

- **Configuration**: See `../adr/` for architecture decision records
- **Examples**: Check project root README for quick start examples
- **Tests**: Look for `tests/` directory for usage examples
