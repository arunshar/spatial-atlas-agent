"""Hermetic tests for the smoke client's success rule."""

from __future__ import annotations

import pytest

import eval_smoke

pytestmark = pytest.mark.unit


def _task(state: str, text: str) -> dict:
    return {
        "kind": "task",
        "status": {"state": state},
        "artifacts": [{"name": "Analysis", "parts": [{"kind": "text", "text": text}]}],
    }


def test_completed_task_returns_its_answer():
    assert eval_smoke._final_answer(_task("completed", "3")) == "3"


@pytest.mark.parametrize("state", ["failed", "canceled", "rejected", "working"])
def test_task_that_did_not_complete_raises_even_with_an_error_artifact(state):
    error_text = "Error: Spatial Atlas could not complete this task. Reference: abc123"

    with pytest.raises(RuntimeError, match=state):
        eval_smoke._final_answer(_task(state, error_text))


def test_completed_task_without_text_raises():
    with pytest.raises(RuntimeError, match="no text"):
        eval_smoke._final_answer({"status": {"state": "completed"}, "artifacts": []})


def test_bare_message_without_state_returns_its_text():
    message = {"kind": "message", "parts": [{"kind": "text", "text": "ok"}]}

    assert eval_smoke._final_answer(message) == "ok"


@pytest.mark.parametrize(
    ("url", "internal"),
    [
        ("http://0.0.0.0:9019/", True),
        ("http://127.0.0.1:9019/", True),
        ("http://localhost:9019/", True),
        ("https://agent.example.org/", False),
    ],
)
def test_internal_url_detection(url, internal):
    assert eval_smoke._is_internal_url(url) is internal
