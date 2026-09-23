"""Concurrency-safe heuristic token reservation for one public A2A execution."""

from __future__ import annotations

import asyncio
import math
from typing import Any

from config import Config
from llm import LLMClient


class TokenBudgetExceeded(RuntimeError):
    """Raised before a provider call that cannot fit in the heuristic execution budget."""


def _text_tokens(value: str) -> int:
    return max(1, math.ceil(len(value) / 4))


def _message_tokens(value: Any) -> int:
    """Conservatively estimate prompt tokens without counting base64 bytes as text."""
    if value is None:
        return 0
    if isinstance(value, str):
        if value.startswith("data:image/"):
            return 2_048
        return _text_tokens(value)
    if isinstance(value, dict):
        return sum(_text_tokens(str(key)) + _message_tokens(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return sum(_message_tokens(item) for item in value)
    return _text_tokens(str(value))


class BudgetedLLMClient(LLMClient):
    """Prevent concurrent calls from oversubscribing one heuristic execution budget.

    Prompt reservation is a heuristic, not exact tokenizer accounting or a provider
    token boundary. Provider-reported usage remains the authoritative observation.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.cost_tracker.max_tokens = config.max_tokens_per_task
        self._budget_limit = config.max_tokens_per_task
        self._budget_spent = 0
        self._budget_reserved = 0
        self._budget_lock = asyncio.Lock()

    async def _reserve(self, prompt_tokens: int, max_tokens: int) -> tuple[int, int]:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        async with self._budget_lock:
            remaining = self._budget_limit - self._budget_spent - self._budget_reserved
            completion_tokens = min(max_tokens, remaining - prompt_tokens)
            if completion_tokens <= 0:
                raise TokenBudgetExceeded("Task token budget is exhausted")
            reservation = prompt_tokens + completion_tokens
            self._budget_reserved += reservation
            return completion_tokens, reservation

    async def _commit(self, reservation: int) -> None:
        async with self._budget_lock:
            self._budget_reserved -= reservation
            self._budget_spent += reservation

    @property
    def budget_remaining(self) -> int:
        return max(0, self._budget_limit - self._budget_spent - self._budget_reserved)

    async def generate(
        self,
        prompt: str,
        *,
        model_tier: str = "standard",
        system_prompt: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        prompt_tokens = _message_tokens([system_prompt, prompt])
        allowed, reservation = await self._reserve(prompt_tokens, max_tokens)
        try:
            return await super().generate(
                prompt,
                model_tier=model_tier,
                system_prompt=system_prompt,
                json_mode=json_mode,
                temperature=temperature,
                max_tokens=allowed,
            )
        finally:
            await self._commit(reservation)

    async def vision_analyze(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        model_tier: str = "vision",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        image_tokens = max(1_024, min(8_192, math.ceil(len(image_bytes) / 1_024)))
        prompt_tokens = _text_tokens(prompt) + image_tokens
        allowed, reservation = await self._reserve(prompt_tokens, max_tokens)
        try:
            return await super().vision_analyze(
                image_bytes,
                prompt,
                model_tier=model_tier,
                temperature=temperature,
                max_tokens=allowed,
            )
        finally:
            await self._commit(reservation)

    async def generate_with_messages(
        self,
        messages: list[dict],
        *,
        model_tier: str = "standard",
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        prompt_tokens = _message_tokens(messages)
        allowed, reservation = await self._reserve(prompt_tokens, max_tokens)
        try:
            return await super().generate_with_messages(
                messages,
                model_tier=model_tier,
                json_mode=json_mode,
                temperature=temperature,
                max_tokens=allowed,
            )
        finally:
            await self._commit(reservation)
