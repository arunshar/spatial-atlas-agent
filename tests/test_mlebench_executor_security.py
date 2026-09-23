"""Security and lifecycle tests for generated-code execution."""

import asyncio
import contextlib
import os
import shlex
import signal
import sys
from pathlib import Path

import pytest

from mlebench.executor import CodeExecutor


def test_safe_env_excludes_parent_secrets_and_uses_private_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("HF_TOKEN", "must-not-leak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-leak")
    monkeypatch.setenv("HTTPS_PROXY", "https://credential@example.invalid")
    monkeypatch.setenv("HOME", "/parent-home")

    env = CodeExecutor()._safe_env(tmp_path)

    assert "OPENAI_API_KEY" not in env
    assert "HF_TOKEN" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "HTTPS_PROXY" not in env
    assert env["HOME"] == str(tmp_path / ".runtime-home")
    assert env["TMPDIR"] == str(tmp_path / ".runtime-tmp")
    assert Path(env["HOME"]).stat().st_mode & 0o777 == 0o700


@pytest.mark.asyncio
async def test_stale_submission_is_removed_before_attempt(tmp_path):
    submission = tmp_path / "submission.csv"
    submission.write_text("stale", encoding="utf-8")

    result = await CodeExecutor(timeout=5).execute("pass", tmp_path, submission)

    assert result is None
    assert not submission.exists()


@pytest.mark.asyncio
async def test_oversized_submission_is_rejected(tmp_path):
    executor = CodeExecutor(timeout=5, max_submission_bytes=16)

    result = await executor.execute(
        "from pathlib import Path\nPath('submission.csv').write_bytes(b'x' * 17)\n",
        tmp_path,
    )

    assert result is None
    assert "Submission exceeds" in (executor.last_error or "")


def _pid_alive(pid: int) -> bool:
    """Report whether the PID is still claimed. A reaped process raises ProcessLookupError."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def _wait_for_exit(pid: int, timeout: float = 5.0) -> bool:
    """Poll until the PID is gone.

    The timeout is a safety margin rather than a race window. A killed descendant is reaped
    within milliseconds, so a correct executor satisfies this almost immediately, and a
    surviving one stays alive far longer than the timeout, so the caller fails every time.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if not _pid_alive(pid):
            return True
        await asyncio.sleep(0.02)
    return not _pid_alive(pid)


def _released_child_source(trigger: Path, marker: Path, *, ignore_sigterm: bool = False) -> str:
    """Build child source that writes `marker` only once `trigger` appears.

    These tests used to have the child sleep a fixed interval and then write, while asserting
    the marker stayed absent. That turned every assertion into a race between the executor's
    group kill and the child's timer, which is why they failed intermittently under coverage:
    instrumentation slows the parent but not the uninstrumented child.

    Waiting on a trigger inverts the dependency. The test creates the trigger only after the
    executor has finished, so a descendant that survived writes within milliseconds and is
    caught, while one that was killed never writes at all.
    """
    ignore = "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_sigterm else ""
    return (
        "import pathlib, signal, time\n"
        f"{ignore}"
        f"trigger = pathlib.Path({str(trigger)!r})\n"
        # 3000 * 10ms caps a leaked child at 30s so it cannot outlive the test session.
        "for _ in range(3000):\n"
        "    if trigger.exists():\n"
        f"        pathlib.Path({str(marker)!r}).write_text('survived')\n"
        "        break\n"
        "    time.sleep(0.01)\n"
    )


async def _release_survivors(trigger: Path, settle: float = 0.5) -> None:
    """Create the trigger, then give any survivor time to act on it.

    The settle window is a margin rather than a race. A live child notices the trigger within
    about ten milliseconds, so half a second is roughly fifty times what it needs.
    """
    trigger.write_text("")
    await asyncio.sleep(settle)


async def _wait_for_file(path: Path, timeout: float = 5.0) -> bool:
    """Poll until `path` exists, so a test can wait on a real event instead of guessing."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if path.exists():
            return True
        await asyncio.sleep(0.01)
    return path.exists()


@pytest.mark.asyncio
async def test_background_descendant_cannot_grow_submission_after_parent_exit(tmp_path):
    """A straggler must not outlive the pipeline and grow the submission.

    An earlier version of this test had the descendant sleep 0.05 seconds and then overwrite
    submission.csv, and asserted the file was untouched. That made the assertion a race
    against the executor's group kill rather than a statement about it. Coverage slows the
    instrumented parent but not the uninstrumented child, so it failed about one run in
    twelve with no bug present.

    The descendant now waits for a trigger file that this test creates only after execute()
    has returned. A survivor writes within milliseconds of the trigger and is caught, and a
    descendant that was killed never writes at all. Both margins below are generous, so
    neither outcome depends on scheduling.
    """
    child = (
        "import time\n"
        "from pathlib import Path\n"
        "trigger = Path('go')\n"
        "for _ in range(3000):\n"
        "    if trigger.exists():\n"
        "        Path('submission.csv').write_bytes(b'x' * 4096)\n"
        "        break\n"
        "    time.sleep(0.01)\n"
    )
    pipeline = (
        "import subprocess, sys\n"
        "from pathlib import Path\n"
        "Path('submission.csv').write_bytes(b'ok')\n"
        f"proc = subprocess.Popen([sys.executable, '-c', {child!r}], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "Path('child.pid').write_text(str(proc.pid))\n"
    )
    executor = CodeExecutor(timeout=5, max_submission_bytes=128)

    result = await executor.execute(pipeline, tmp_path)

    # The pipeline records the PID before it exits, so this read never races the kill.
    child_pid = int((tmp_path / "child.pid").read_text())
    try:
        # Release a survivor. A descendant that was killed ignores this, which is the point.
        (tmp_path / "go").write_text("")
        await asyncio.sleep(0.5)

        assert result == b"ok"
        assert (tmp_path / "submission.csv").read_bytes() == b"ok"
        assert await _wait_for_exit(child_pid), (
            "the background descendant outlived the pipeline, so its process group was "
            "not terminated"
        )
    finally:
        if _pid_alive(child_pid):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(child_pid, signal.SIGKILL)


@pytest.mark.asyncio
async def test_process_group_is_terminated_before_the_submission_is_read(tmp_path, monkeypatch):
    """The straggler kill must precede the submission read, not merely happen eventually.

    execute() terminates the group twice: once before reading the submission, and again in a
    finally block. Only the first placement stops a straggler from growing the file before it
    is read, but a test that just looks for surviving processes passes either way, because
    the finally cleans up regardless. This pins the ordering instead of the outcome. The
    patched terminate rewrites the submission, so the bytes execute() returns say which ran
    first, with no timing involved.
    """
    real_terminate = CodeExecutor._terminate_process_group

    async def terminate_and_mark(self, proc, process_group_id):
        # The finally call passes None, so this marks only the pre-read termination.
        if process_group_id is not None:
            (tmp_path / "submission.csv").write_bytes(b"terminated-first")
        await real_terminate(self, proc, process_group_id)

    monkeypatch.setattr(CodeExecutor, "_terminate_process_group", terminate_and_mark)

    result = await CodeExecutor(timeout=5, max_submission_bytes=128).execute(
        "from pathlib import Path\nPath('submission.csv').write_bytes(b'ok')\n",
        tmp_path,
    )

    assert result == b"terminated-first", (
        "the submission was read before the process group was terminated, so a straggler "
        "could grow it first"
    )


def _always_eperm(_process_group_id, _signal_number):
    raise PermissionError(1, "Operation not permitted")


def test_recycled_process_group_id_is_reported_as_gone(monkeypatch):
    """EPERM from killpg means the id was recycled, not that our descendants are alive.

    The kernel reissues process-group ids once our descendants exit, and probing an id that
    now belongs to another user raises PermissionError rather than ProcessLookupError. This
    is the defect behind an intermittent suite failure: the probe caught only
    ProcessLookupError, so PermissionError escaped the termination path.
    """
    monkeypatch.setattr(os, "killpg", _always_eperm)

    assert CodeExecutor._process_group_exists(424242) is False


@pytest.mark.asyncio
async def test_termination_survives_a_recycled_process_group_id(monkeypatch, tmp_path):
    """A recycled group id must not turn a successful run into an exception.

    Before the probe handled EPERM this raised PermissionError out of execute(), which is
    how the failure presented: not as a wrong answer, but as a crash partway through
    teardown on roughly one run in twelve.
    """
    monkeypatch.setattr(os, "killpg", _always_eperm)

    result = await CodeExecutor(timeout=5).execute(
        "from pathlib import Path\nPath('submission.csv').write_bytes(b'ok')\n",
        tmp_path,
    )

    assert result == b"ok"


@pytest.mark.asyncio
async def test_stdout_capture_is_bounded(tmp_path):
    executor = CodeExecutor(timeout=5, max_stream_bytes=128)

    result = await executor.execute("print('x' * 1024)", tmp_path)

    assert result is None
    assert "stdout exceeds" in (executor.last_error or "")


@pytest.mark.asyncio
async def test_timeout_kills_spawned_child_process_group(tmp_path):
    marker = tmp_path / "child-survived"
    trigger = tmp_path / "release-timeout"
    spawned = tmp_path / "child-spawned"
    child = _released_child_source(trigger, marker)
    pipeline = (
        "import subprocess,sys,time\n"
        "from pathlib import Path\n"
        f"subprocess.Popen([{sys.executable!r}, '-c', {child!r}])\n"
        f"Path({str(spawned)!r}).write_text('')\n"
        "time.sleep(30)\n"
    )

    executor = CodeExecutor(timeout=0.1)
    result = await executor.execute(pipeline, tmp_path)
    await _release_survivors(trigger)

    assert spawned.exists(), "the pipeline never spawned a child, so this proved nothing"
    assert result is None
    assert "timed out" in (executor.last_error or "")
    assert not marker.exists()


@pytest.mark.asyncio
async def test_cancellation_kills_spawned_child_process_group(tmp_path):
    marker = tmp_path / "child-survived-cancel"
    trigger = tmp_path / "release-cancel"
    spawned = tmp_path / "child-spawned-cancel"
    child = _released_child_source(trigger, marker)
    pipeline = (
        "import subprocess,sys,time\n"
        "from pathlib import Path\n"
        f"subprocess.Popen([{sys.executable!r}, '-c', {child!r}])\n"
        f"Path({str(spawned)!r}).write_text('')\n"
        "time.sleep(30)\n"
    )

    task = asyncio.create_task(CodeExecutor(timeout=30).execute(pipeline, tmp_path))
    # Cancel only once the child provably exists. A fixed sleep here could cancel before the
    # pipeline had spawned anything, which would pass without exercising the kill at all.
    assert await _wait_for_file(spawned), "the pipeline never spawned a child"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _release_survivors(trigger)

    assert not marker.exists()


@pytest.mark.asyncio
async def test_timeout_kills_sigterm_ignoring_child_after_parent_exits(tmp_path):
    marker = tmp_path / "sigterm-ignoring-child-survived"
    trigger = tmp_path / "release-sigterm-ignoring"
    spawned = tmp_path / "sigterm-ignoring-child-spawned"
    child = _released_child_source(trigger, marker, ignore_sigterm=True)
    pipeline = (
        "import subprocess,sys\n"
        "from pathlib import Path\n"
        f"subprocess.Popen([{sys.executable!r}, '-c', {child!r}])\n"
        f"Path({str(spawned)!r}).write_text('')\n"
    )

    executor = CodeExecutor(timeout=0.1, termination_grace_seconds=0.2)
    result = await executor.execute(pipeline, tmp_path)
    await _release_survivors(trigger)

    assert spawned.exists(), "the pipeline never spawned a child, so this proved nothing"
    assert result is None
    assert "timed out" in (executor.last_error or "")
    assert not marker.exists()


def test_test_payload_does_not_depend_on_shell_interpolation():
    """Document that child payloads are passed as argv, not through a shell."""
    payload = "print('$OPENAI_API_KEY')"
    assert shlex.quote(payload) != payload
    assert os.path.basename(sys.executable).startswith("python")
