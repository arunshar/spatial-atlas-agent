"""Sanitized per-call telemetry tests for the unified LLM client."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import llm as llm_module
from config import Config
from cost.tracker import CostTracker
from llm import LLMClient

pytestmark = pytest.mark.unit


def _response(
    *,
    content: str | None = "visible response text",
    reasoning_content: str | None = "hidden reasoning text",
) -> SimpleNamespace:
    message = SimpleNamespace(content=content, reasoning_content=reasoning_content)
    choice = SimpleNamespace(message=message, finish_reason="length")
    usage = SimpleNamespace(
        prompt_tokens=11,
        completion_tokens=6,
        total_tokens=17,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=4),
    )
    return SimpleNamespace(
        id="provider-response-id-sensitive",
        choices=[choice],
        usage=usage,
        _hidden_params={"response_cost": 0.25},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "call_kind"),
    [
        ("generate", "generate"),
        ("vision_analyze", "vision_analyze"),
        ("generate_with_messages", "generate_with_messages"),
    ],
)
async def test_llm_records_sanitized_call_telemetry(monkeypatch, method, call_kind):
    completion = AsyncMock(return_value=_response())
    monkeypatch.setattr(llm_module.litellm, "acompletion", completion)
    client = LLMClient(Config(llm_enable_thinking=False))
    cursor = client.cost_tracker.telemetry_cursor()

    if method == "generate":
        await client.generate(
            "secret prompt text",
            model_tier="fast",
            system_prompt="secret system text",
            max_tokens=123,
        )
    elif method == "vision_analyze":
        await client.vision_analyze(
            b"secret image bytes",
            "secret vision prompt",
            model_tier="fast",
            max_tokens=123,
        )
    else:
        await client.generate_with_messages(
            [{"role": "user", "content": "secret message text"}],
            model_tier="fast",
            max_tokens=123,
        )

    records = client.cost_tracker.telemetry_since(cursor)
    assert records == [
        {
            "schema_version": 1,
            "role": "fast",
            "model_tier": "fast",
            "call_kind": call_kind,
            "configured_max_tokens": 123,
            "provider_finish_reason": "length",
            "token_usage": {
                "prompt_tokens": 11,
                "completion_tokens": 6,
                "total_tokens": 17,
                "reasoning_tokens": 4,
            },
            "response_id_sha256": hashlib.sha256(
                b"spatial-atlas-llm-response-id-v1\0provider-response-id-sensitive"
            ).hexdigest(),
            "message_content_empty": False,
            "reasoning_content_empty": False,
        }
    ]
    serialized = json.dumps(records, sort_keys=True)
    for forbidden in (
        "provider-response-id-sensitive",
        "secret prompt text",
        "secret system text",
        "secret vision prompt",
        "secret message text",
        "visible response text",
        "hidden reasoning text",
    ):
        assert forbidden not in serialized
    assert client.cost_tracker.stats.num_calls == 1
    assert client.cost_tracker.stats.total_tokens == 17
    assert client.cost_tracker.stats.estimated_cost_usd == 0.25


@pytest.mark.parametrize(
    "unsafe_finish_reason",
    [
        "",
        "length with provider text",
        "x" * 65,
        "léngth",
        7,
    ],
)
def test_tracker_rejects_unsafe_finish_reason(unsafe_finish_reason):
    response = _response()
    response.choices[0].finish_reason = unsafe_finish_reason
    tracker = CostTracker()

    with pytest.raises(ValueError, match="safe bounded token"):
        tracker.track(
            response,
            role="main",
            model_tier="main",
            call_kind="generate",
            configured_max_tokens=8192,
        )

    assert tracker.stats.num_calls == 0
    assert tracker.telemetry_since(0) == []


def test_tracker_records_empty_content_and_missing_usage_without_raw_payloads():
    response = SimpleNamespace(
        id=None,
        choices=[
            SimpleNamespace(
                finish_reason=None,
                message=SimpleNamespace(content="  ", reasoning_content=None),
            )
        ],
        usage=None,
    )
    tracker = CostTracker()

    tracker.track(
        response,
        role="main",
        model_tier="main",
        call_kind="generate_with_messages",
        configured_max_tokens=8192,
    )

    record = tracker.telemetry_since(0)[0]
    assert record["provider_finish_reason"] is None
    assert record["response_id_sha256"] is None
    assert record["message_content_empty"] is True
    assert record["reasoning_content_empty"] is True
    assert record["token_usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
    }
    assert tracker.stats.num_calls == 1


def test_telemetry_slices_are_detached_from_tracker_state():
    tracker = CostTracker()
    tracker.track(
        _response(),
        role="vlm",
        model_tier="vlm",
        call_kind="vision_analyze",
        configured_max_tokens=8192,
    )

    first = tracker.telemetry_since(0)
    first[0]["token_usage"]["completion_tokens"] = 999

    assert tracker.telemetry_since(0)[0]["token_usage"]["completion_tokens"] == 6
