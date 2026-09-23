"""Tests that task failures cannot be reported as successful completions."""

from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import AsyncMock

import pytest

import executor as executor_module
from agent import Agent, AgentTaskError
from executor import Executor
from fieldwork.vision import VisionPipeline


class RecordingUpdater:
    instances: ClassVar[list] = []

    def __init__(self, _queue, _task_id, _context_id):
        self._terminal_state_reached = False
        self.completed = False
        self.failed_called = False
        self.artifacts = []
        self.__class__.instances.append(self)

    async def start_work(self):
        return None

    async def complete(self):
        self.completed = True
        self._terminal_state_reached = True

    async def failed(self, _message):
        self.failed_called = True
        self._terminal_state_reached = True


class RecordingEventQueue:
    async def enqueue_event(self, _event):
        return None


def _context():
    return SimpleNamespace(message=object(), current_task=None)


@pytest.mark.asyncio
async def test_agent_error_artifact_is_sanitized_and_exception_is_raised(caplog):
    agent = Agent.__new__(Agent)
    agent.cost_tracker = SimpleNamespace(summary=lambda: "unused")
    agent._parse_message = lambda _message: (["task"], [])
    agent._classify_domain = lambda _text, _files: "fieldwork"
    agent._handle_fieldwork = AsyncMock(side_effect=ValueError("provider-secret-detail"))
    updater = SimpleNamespace(
        update_status=AsyncMock(),
        add_artifact=AsyncMock(),
    )

    with pytest.raises(AgentTaskError, match="could not complete"):
        await agent.run(object(), updater)

    artifact = updater.add_artifact.await_args.kwargs
    text = artifact["parts"][0].root.text
    assert "provider-secret-detail" not in text
    assert "could not complete" in text
    assert "provider-secret-detail" not in caplog.text


@pytest.mark.asyncio
async def test_executor_marks_agent_error_failed_and_never_completed(monkeypatch, caplog):
    RecordingUpdater.instances.clear()
    monkeypatch.setattr(executor_module, "TaskUpdater", RecordingUpdater)
    monkeypatch.setattr(
        executor_module,
        "new_task",
        lambda _message: SimpleNamespace(id="task", context_id="context"),
    )

    class FailingAgent:
        async def run(self, _message, _updater):
            raise AgentTaskError("a1b2c3d4e5f6")

    await Executor(agent_factory=FailingAgent).execute(_context(), RecordingEventQueue())

    updater = RecordingUpdater.instances[-1]
    assert updater.failed_called is True
    assert updater.completed is False
    assert "provider-secret-detail" not in caplog.text


@pytest.mark.asyncio
async def test_executor_sanitizes_unexpected_agent_exception(monkeypatch, caplog):
    RecordingUpdater.instances.clear()
    monkeypatch.setattr(executor_module, "TaskUpdater", RecordingUpdater)
    monkeypatch.setattr(
        executor_module,
        "new_task",
        lambda _message: SimpleNamespace(id="task", context_id="context"),
    )

    class FailingAgent:
        async def run(self, _message, _updater):
            raise ValueError("provider-secret-detail")

    await Executor(agent_factory=FailingAgent).execute(_context(), RecordingEventQueue())

    assert RecordingUpdater.instances[-1].failed_called is True
    assert "provider-secret-detail" not in caplog.text


@pytest.mark.asyncio
async def test_executor_uses_fresh_agent_for_each_task(monkeypatch):
    RecordingUpdater.instances.clear()
    monkeypatch.setattr(executor_module, "TaskUpdater", RecordingUpdater)
    monkeypatch.setattr(
        executor_module,
        "new_task",
        lambda _message: SimpleNamespace(id="task", context_id="same-context"),
    )
    created = []

    class SuccessfulAgent:
        def __init__(self):
            created.append(self)

        async def run(self, _message, _updater):
            return None

    executor = Executor(agent_factory=SuccessfulAgent)
    await executor.execute(_context(), RecordingEventQueue())
    await executor.execute(_context(), RecordingEventQueue())

    assert len(created) == 2
    assert created[0] is not created[1]
    assert not hasattr(executor, "agents")
    assert all(updater.completed for updater in RecordingUpdater.instances)


def test_pdf_error_marker_cannot_reinject_exception_text(monkeypatch, caplog):
    def fail_reader(_stream):
        raise ValueError("provider-secret-detail")

    monkeypatch.setattr("fieldwork.vision.PdfReader", fail_reader)
    result = VisionPipeline(llm=None)._process_pdf("broken.pdf", b"broken")

    assert result == "[PDF: broken.pdf] Error: extraction failed"
    assert "provider-secret-detail" not in caplog.text


@pytest.mark.asyncio
async def test_video_error_marker_cannot_reinject_exception_text(monkeypatch, caplog):
    pipeline = VisionPipeline(llm=None)

    def fail_frames(_data):
        raise ValueError("provider-secret-detail")

    monkeypatch.setattr(pipeline, "_extract_video_frames", fail_frames)
    result = await pipeline._process_video("broken.mp4", b"broken")

    assert result == "[Video: broken.mp4] Error: processing failed"
    assert "provider-secret-detail" not in caplog.text
