"""Tests for the public Agent's estimated token reservation budget."""

import asyncio
from pathlib import Path

import pytest

from budgeted_llm import BudgetedLLMClient, TokenBudgetExceeded, _message_tokens
from config import Config
from llm import LLMClient


@pytest.mark.asyncio
async def test_completion_limit_is_clamped_to_remaining_budget(monkeypatch):
    captured = {}

    async def fake_generate(self, prompt, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(LLMClient, "generate", fake_generate)
    client = BudgetedLLMClient(Config(max_tokens_per_task=20))

    assert await client.generate("12345678", max_tokens=100) == "ok"
    assert captured["max_tokens"] == 18
    assert client.budget_remaining == 0


@pytest.mark.asyncio
async def test_exhausted_budget_makes_no_provider_call(monkeypatch):
    called = False

    async def fake_generate(self, prompt, **kwargs):
        nonlocal called
        called = True
        return "unexpected"

    monkeypatch.setattr(LLMClient, "generate", fake_generate)
    client = BudgetedLLMClient(Config(max_tokens_per_task=10))
    client._budget_spent = 10

    with pytest.raises(TokenBudgetExceeded, match="exhausted"):
        await client.generate("task", max_tokens=1)
    assert called is False


@pytest.mark.asyncio
async def test_concurrent_calls_cannot_oversubscribe_budget(monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_generate(self, prompt, **kwargs):
        entered.set()
        await release.wait()
        return "ok"

    monkeypatch.setattr(LLMClient, "generate", blocked_generate)
    client = BudgetedLLMClient(Config(max_tokens_per_task=100))
    first = asyncio.create_task(client.generate("x", max_tokens=99))
    await entered.wait()

    with pytest.raises(TokenBudgetExceeded):
        await client.generate("second", max_tokens=1)

    release.set()
    assert await first == "ok"
    assert client.budget_remaining == 0


@pytest.mark.asyncio
async def test_failed_provider_attempt_still_consumes_reserved_budget(monkeypatch):
    async def failed_generate(self, prompt, **kwargs):
        raise RuntimeError("provider failed after request dispatch")

    monkeypatch.setattr(LLMClient, "generate", failed_generate)
    client = BudgetedLLMClient(Config(max_tokens_per_task=20))

    with pytest.raises(RuntimeError, match="provider failed"):
        await client.generate("1234", max_tokens=9)
    assert client.budget_remaining == 10


def test_image_data_url_is_counted_as_bounded_visual_input():
    short = _message_tokens({"image_url": {"url": "data:image/jpeg;base64,abc"}})
    long = _message_tokens({"image_url": {"url": "data:image/jpeg;base64," + ("a" * 1_000_000)}})

    assert short == long
    assert short >= 2_048


def test_public_agent_uses_budgeted_client():
    source = Path(__file__).parents[1].joinpath("src/agent.py").read_text(encoding="utf-8")

    assert "BudgetedLLMClient(self.config)" in source
