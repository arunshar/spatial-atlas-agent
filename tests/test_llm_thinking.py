from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import llm as llm_module
from config import Config
from llm import LLMClient

pytestmark = pytest.mark.unit


def _response(content: str = "7") -> SimpleNamespace:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    return SimpleNamespace(choices=[choice], usage=usage)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "vision_analyze", "generate_with_messages"])
async def test_llm_forwards_explicit_no_thinking(monkeypatch, method):
    completion = AsyncMock(return_value=_response())
    monkeypatch.setattr(llm_module.litellm, "acompletion", completion)
    client = LLMClient(Config(llm_enable_thinking=False))

    if method == "generate":
        await client.generate("question")
    elif method == "vision_analyze":
        await client.vision_analyze(b"image", "question")
    else:
        await client.generate_with_messages([{"role": "user", "content": "question"}])

    assert completion.call_args.kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


@pytest.mark.asyncio
async def test_llm_omits_chat_template_options_by_default(monkeypatch):
    completion = AsyncMock(return_value=_response())
    monkeypatch.setattr(llm_module.litellm, "acompletion", completion)

    await LLMClient(Config(llm_enable_thinking=None)).generate("question")

    assert "extra_body" not in completion.call_args.kwargs
