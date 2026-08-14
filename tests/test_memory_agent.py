"""Tests for the MemoryAgent lifecycle (start -> action -> update -> done -> end).

Uses a fake tokenizer (no network/model download) so this suite runs in CI
without pulling a real Qwen checkpoint. The fake tokenizer implements only
the surface MemoryAgent touches: `apply_chat_template`, `encode`,
`pad_token_id`, `eos_token_id`.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from recurrent.impls.memory import MemoryAgent, MemoryConfig


class FakeTokenizer:
    """Deterministic char-level tokenizer sufficient for MemoryAgent's needs."""

    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        # MemoryAgent only ever calls this with a single {"content": "{message}"}
        # user turn (see recurrent.utils.chat_template), so echo it back as a
        # minimal, deterministic wrapper string.
        content = messages[0]["content"]
        return f"<user>{content}</user>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Map each character to its ordinal, offset to avoid colliding with
        # pad_token_id=0 / eos_token_id=1.
        return [ord(ch) + 2 for ch in text]


@pytest.fixture
def memory_config() -> MemoryConfig:
    return MemoryConfig(
        context_key="context",
        max_prompt_length=32,
        chunk_size=4,
        max_memorization_length=16,
        max_chunks=3,
        max_final_response_length=16,
    )


@pytest.fixture
def memory_agent(memory_config: MemoryConfig) -> MemoryAgent:
    return MemoryAgent(tokenizer=FakeTokenizer(), config=memory_config)


def make_gen_batch(bsz: int, context_length: list[int], chunk_size: int, num_chunks: int):
    """Build a minimal duck-typed stand-in for `verl.DataProto`.

    MemoryAgent only accesses `.batch['context_length']`, `.batch['context_ids']`,
    and `.non_tensor_batch['prompt_ids']`, so a SimpleNamespace with plain
    dicts avoids pulling in tensordict/Ray for a unit test.
    """
    context_ids = torch.zeros((bsz, chunk_size * num_chunks), dtype=torch.long)
    for i in range(bsz):
        context_ids[i, : context_length[i]] = torch.arange(2, 2 + context_length[i])

    prompt_ids = np.empty(bsz, dtype=object)
    for i in range(bsz):
        prompt_ids[i] = torch.tensor([10 + i], dtype=torch.long)

    return SimpleNamespace(
        batch={
            "context_length": torch.tensor(context_length, dtype=torch.long),
            "context_ids": context_ids,
        },
        non_tensor_batch={"prompt_ids": prompt_ids},
    )


class TestMemoryAgentLifecycle:
    def test_start_initializes_state(self, memory_agent: MemoryAgent) -> None:
        gen_batch = make_gen_batch(bsz=2, context_length=[8, 4], chunk_size=4, num_chunks=3)
        memory_agent.start(gen_batch, timing_raw={})
        assert memory_agent.bsz == 2
        assert memory_agent.step == 0
        assert memory_agent.is_final is False
        assert len(memory_agent.memory) == 2
        assert all(m is None for m in memory_agent.memory)

    def test_action_returns_active_samples_only(self, memory_agent: MemoryAgent) -> None:
        # sample 0 has 2 chunks worth of context, sample 1 has 1 chunk.
        gen_batch = make_gen_batch(bsz=2, context_length=[8, 4], chunk_size=4, num_chunks=3)
        memory_agent.start(gen_batch, timing_raw={})
        messages, meta_info = memory_agent.action()
        assert len(messages) == 2  # both samples still active on step 0
        assert "generation_kwargs" in meta_info
        assert meta_info["generation_kwargs"]["max_tokens"] == memory_agent.config.gen_max_tokens_memorization

    def test_update_stores_memory_for_active_samples(self, memory_agent: MemoryAgent) -> None:
        gen_batch = make_gen_batch(bsz=2, context_length=[8, 4], chunk_size=4, num_chunks=3)
        memory_agent.start(gen_batch, timing_raw={})
        memory_agent.action()

        fake_responses = torch.tensor([[5, 6, 0, 0], [7, 0, 0, 0]], dtype=torch.long)
        gen_output = SimpleNamespace(batch={"responses": fake_responses})
        memory_agent.update(gen_output)

        assert memory_agent.step == 1
        assert memory_agent.memory[0] is not None
        assert memory_agent.memory[1] is not None

    def test_done_is_false_until_final_turn(self, memory_agent: MemoryAgent) -> None:
        gen_batch = make_gen_batch(bsz=1, context_length=[4], chunk_size=4, num_chunks=1)
        memory_agent.start(gen_batch, timing_raw={})
        assert memory_agent.done() is False

    def test_full_loop_terminates_and_end_returns_masks(self, memory_agent: MemoryAgent) -> None:
        gen_batch = make_gen_batch(bsz=1, context_length=[4], chunk_size=4, num_chunks=1)
        memory_agent.start(gen_batch, timing_raw={})

        max_iterations = 5
        iterations = 0
        while not memory_agent.done() and iterations < max_iterations:
            messages, _ = memory_agent.action()
            fake_responses = torch.zeros((len(messages), 4), dtype=torch.long)
            memory_agent.update(SimpleNamespace(batch={"responses": fake_responses}))
            iterations += 1

        assert memory_agent.done() is True
        final_mask, sample_index = memory_agent.end()
        assert final_mask.dtype == torch.bool
        assert sample_index.dtype in (torch.long, torch.int)
        assert int(final_mask.sum()) == 1  # exactly one final turn for bsz=1

    def test_agent_state_cleared_after_end(self, memory_agent: MemoryAgent) -> None:
        gen_batch = make_gen_batch(bsz=1, context_length=[4], chunk_size=4, num_chunks=1)
        memory_agent.start(gen_batch, timing_raw={})
        while not memory_agent.done():
            messages, _ = memory_agent.action()
            fake_responses = torch.zeros((len(messages), 4), dtype=torch.long)
            memory_agent.update(SimpleNamespace(batch={"responses": fake_responses}))
        memory_agent.end()
        assert not hasattr(memory_agent, "gen_batch")
        assert not hasattr(memory_agent, "memory")
