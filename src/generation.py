"""Pydantic-validated generation config shared by the sync and async
generation managers (`recurrent/generation_manager.py`,
`recurrent/async_generation_manager.py`) and the eval/train scripts.

Kept separate from `recurrent/interface.py::RConfig` (a plain dataclass with
no validation, extended freely by agent subclasses) because generation
sampling params are orthogonal to agent-specific memory config and benefit
from strict validation at config-load time rather than at first use.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class GenerationConfig(BaseModel):
    """Sampling parameters for a single generation call.

    Mirrors `RAgent.sampling_params` / `AsyncRAgent.sampling_params` in
    `recurrent/interface.py`, but validated eagerly so a bad config value
    (e.g. `top_p=1.5`) fails at Hydra config-load time, not mid-rollout.
    """

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    max_tokens: int = Field(default=512, gt=0)
    seed: Optional[int] = None
    do_sample: bool = True
    n: int = Field(default=1, ge=1)

    @field_validator("max_tokens")
    @classmethod
    def _max_tokens_reasonable(cls, value: int) -> int:
        if value > 32768:
            raise ValueError(f"max_tokens={value} exceeds the supported context window (32768).")
        return value

    def to_sampling_kwargs(self) -> dict[str, float | int | bool | None]:
        """Render as the kwargs shape expected by `RAgent.sampling_params`."""
        return {
            "n": self.n,
            "temperature": self.temperature if self.do_sample else 0.0,
            "top_p": self.top_p if self.do_sample else 1.0,
        }

    @classmethod
    def for_validation(cls) -> GenerationConfig:
        """Deterministic, greedy config for eval/validation runs."""
        return cls(temperature=0.0, top_p=1.0, do_sample=False, n=1)
