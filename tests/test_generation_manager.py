"""Tests for the generation managers and src/generation.py's GenerationConfig.

`LLMGenerationManager` and `AsyncLLMGenerationManager` depend on live Ray
rollout workers / an async chat-completion scheduler, so full end-to-end
generation is out of scope for a unit test. These tests cover:
  - `GenerationConfig` validation (the pydantic contract new callers rely on)
  - the pure helper functions in `recurrent/generation_manager.py`
    (`collate_fn`, `_timer`, `batch_subsample_images`)
  - error handling in the async loop's `NotImplementedError` guard
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image
from pydantic import ValidationError

from recurrent.generation_manager import _timer, batch_subsample_images, collate_fn
from src.generation import GenerationConfig


class TestGenerationConfig:
    def test_defaults_are_valid(self) -> None:
        config = GenerationConfig()
        assert config.temperature == 0.7
        assert config.top_p == 0.9

    def test_rejects_top_p_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            GenerationConfig(top_p=1.5)

    def test_rejects_negative_temperature(self) -> None:
        with pytest.raises(ValidationError):
            GenerationConfig(temperature=-0.1)

    def test_rejects_max_tokens_over_context_window(self) -> None:
        with pytest.raises(ValidationError):
            GenerationConfig(max_tokens=100_000)

    def test_to_sampling_kwargs_greedy_when_not_sampling(self) -> None:
        config = GenerationConfig(temperature=0.7, top_p=0.9, do_sample=False)
        kwargs = config.to_sampling_kwargs()
        assert kwargs["temperature"] == 0.0
        assert kwargs["top_p"] == 1.0

    def test_to_sampling_kwargs_preserves_values_when_sampling(self) -> None:
        config = GenerationConfig(temperature=0.5, top_p=0.8, do_sample=True)
        kwargs = config.to_sampling_kwargs()
        assert kwargs["temperature"] == 0.5
        assert kwargs["top_p"] == 0.8

    def test_for_validation_is_deterministic(self) -> None:
        config = GenerationConfig.for_validation()
        assert config.temperature == 0.0
        assert config.do_sample is False
        assert config.n == 1


class TestCollateFn:
    def test_splits_tensor_and_non_tensor_keys(self) -> None:
        data_list = [
            {"input_ids": torch.tensor([1, 2, 3]), "sample_id": "a"},
            {"input_ids": torch.tensor([4, 5, 6]), "sample_id": "b"},
        ]
        tensors, non_tensors = collate_fn(data_list)
        assert torch.equal(tensors["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 6]]))
        assert list(non_tensors["sample_id"]) == ["a", "b"]

    def test_empty_input_returns_empty_dicts(self) -> None:
        tensors, non_tensors = collate_fn([])
        assert tensors == {}
        assert non_tensors == {}


class TestTimerContextManager:
    def test_accumulates_across_multiple_calls(self) -> None:
        timing_raw: dict[str, float] = {}
        with _timer("step_a", timing_raw):
            pass
        with _timer("step_a", timing_raw):
            pass
        assert "step_a" in timing_raw
        assert timing_raw["step_a"] >= 0.0

    def test_separate_names_do_not_interfere(self) -> None:
        timing_raw: dict[str, float] = {}
        with _timer("step_a", timing_raw):
            pass
        with _timer("step_b", timing_raw):
            pass
        assert set(timing_raw.keys()) == {"step_a", "step_b"}

    def test_propagates_exceptions_from_the_timed_block(self) -> None:
        timing_raw: dict[str, float] = {}
        with pytest.raises(ValueError):
            with _timer("step_fail", timing_raw):
                raise ValueError("boom")


class TestBatchSubsampleImages:
    def test_preserves_order_and_none_entries(self) -> None:
        img = Image.new("RGB", (10, 10), color=(255, 0, 0))
        images = [img, None, img]
        result = batch_subsample_images(images, ratio=0.5)
        assert len(result) == 3
        assert result[1] is None
        assert result[0].size == (10, 10)  # resized down then back up to original size

    def test_ratio_one_returns_images_unchanged(self) -> None:
        img = Image.new("RGB", (8, 8))
        result = batch_subsample_images([img], ratio=1.0)
        assert result[0] is img


class TestAsyncGenerationManagerGuard:
    """The async manager currently raises NotImplementedError at construction
    (see recurrent/async_generation_manager.py); this test locks in that
    documented, intentional behavior so it fails loudly if silently removed
    without updating the docstring."""

    def test_construction_raises_not_implemented(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock

        import recurrent.async_generation_manager as agm_module

        fake_tokenizer = MagicMock()
        fake_tokenizer.chat_template = "{% generation %}"
        # `set_chat_template` requires a registered tokenizer class (see
        # recurrent/chat_template/utils.py); stub it so this test exercises
        # only the NotImplementedError guard under test, not tokenizer
        # registration.
        monkeypatch.setattr(agm_module, "set_chat_template", lambda tok: tok)

        with pytest.raises(NotImplementedError):
            agm_module.AsyncLLMGenerationManager(
                tokenizer=fake_tokenizer,
                async_server=MagicMock(),
                config=MagicMock(),
                rollout_config=MagicMock(),
                agent_cls=MagicMock(__name__="FakeAgent"),
            )
