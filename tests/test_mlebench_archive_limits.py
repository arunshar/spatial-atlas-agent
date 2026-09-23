"""Fail-closed tests for MLE-Bench archive handling."""

import asyncio
import io
import tarfile
import types
from pathlib import Path

import pytest

import mlebench.handler as handler_module
from mlebench.handler import MLEBenchHandler


def _handler() -> MLEBenchHandler:
    handler = MLEBenchHandler.__new__(MLEBenchHandler)
    handler._active_work_dirs = set()
    return handler


def _tar_bytes(entries: list[tuple[str, bytes, bytes | None]]) -> bytes:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for name, data, member_type in entries:
            info = tarfile.TarInfo(name)
            if member_type is not None:
                info.type = member_type
                if member_type in {tarfile.SYMTYPE, tarfile.LNKTYPE}:
                    info.linkname = "target"
                archive.addfile(info)
            else:
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    return payload.getvalue()


@pytest.mark.asyncio
async def test_code_execution_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ATLAS_ENABLE_MLEBENCH_CODE_EXECUTION", raising=False)
    monkeypatch.delenv("ATLAS_TRUSTED_ISOLATED_WORKER", raising=False)

    with pytest.raises(PermissionError, match="disabled"):
        await _handler().handle("task", [], object())


@pytest.mark.asyncio
async def test_enable_flag_without_isolated_worker_attestation_is_rejected(monkeypatch):
    monkeypatch.setenv("ATLAS_ENABLE_MLEBENCH_CODE_EXECUTION", "true")
    monkeypatch.delenv("ATLAS_TRUSTED_ISOLATED_WORKER", raising=False)

    with pytest.raises(PermissionError, match="both"):
        await _handler().handle("task", [], object())


def test_invalid_base64_is_rejected():
    with pytest.raises(ValueError, match="valid base64"):
        _handler()._extract_competition(
            [("competition.tar.gz", "application/gzip", "not base64 **")]
        )


def test_decoded_archive_limit_is_enforced(monkeypatch):
    monkeypatch.setattr(handler_module, "MAX_ARCHIVE_BYTES", 4)

    with pytest.raises(ValueError, match="upload limit"):
        _handler()._extract_competition([("competition.tar.gz", "application/gzip", b"12345")])


def test_member_count_limit_is_enforced(monkeypatch):
    monkeypatch.setattr(handler_module, "MAX_ARCHIVE_MEMBERS", 1)
    payload = _tar_bytes([("a.txt", b"a", None), ("b.txt", b"b", None)])

    with pytest.raises(ValueError, match="too many members"):
        _handler()._extract_competition([("competition.tar.gz", "application/gzip", payload)])


def test_member_and_total_size_limits_are_enforced(monkeypatch):
    payload = _tar_bytes([("a.txt", b"1234", None), ("b.txt", b"5678", None)])
    monkeypatch.setattr(handler_module, "MAX_ARCHIVE_MEMBER_BYTES", 3)
    with pytest.raises(ValueError, match="member is too large"):
        _handler()._extract_competition([("competition.tar.gz", "application/gzip", payload)])

    handler = _handler()
    monkeypatch.setattr(handler_module, "MAX_ARCHIVE_MEMBER_BYTES", 10)
    monkeypatch.setattr(handler_module, "MAX_ARCHIVE_EXPANDED_BYTES", 7)
    with pytest.raises(ValueError, match="expands beyond"):
        handler._extract_competition([("competition.tar.gz", "application/gzip", payload)])


@pytest.mark.parametrize(
    ("name", "member_type"),
    [
        ("../escape", None),
        ("/absolute", None),
        ("link", tarfile.SYMTYPE),
        ("hardlink", tarfile.LNKTYPE),
        ("device", tarfile.CHRTYPE),
        ("fifo", tarfile.FIFOTYPE),
    ],
)
def test_unsafe_archive_members_are_rejected(name, member_type):
    payload = _tar_bytes([(name, b"x", member_type)])

    with pytest.raises(ValueError, match="Unsafe|Unsupported"):
        _handler()._extract_competition([("competition.tar.gz", "application/gzip", payload)])


def test_duplicate_archive_destinations_are_rejected():
    payload = _tar_bytes([("same.txt", b"a", None), ("same.txt", b"b", None)])

    with pytest.raises(ValueError, match="Duplicate"):
        _handler()._extract_competition([("competition.tar.gz", "application/gzip", payload)])


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
async def test_work_directory_is_removed_for_every_terminal_path(tmp_path, monkeypatch, outcome):
    monkeypatch.setenv("ATLAS_ENABLE_MLEBENCH_CODE_EXECUTION", "true")
    monkeypatch.setenv("ATLAS_TRUSTED_ISOLATED_WORKER", "true")
    payload = _tar_bytes([("data/train.csv", b"x,y\n1,2\n", None)])
    handler = _handler()
    extracted: list[Path] = []

    async def fake_inner(self, _text, _file_parts, _updater):
        work_dir = self._extract_competition([("competition.tar.gz", "application/gzip", payload)])
        extracted.append(work_dir)
        assert work_dir.exists()
        if outcome == "failure":
            raise RuntimeError("expected failure")
        if outcome == "cancel":
            await asyncio.sleep(30)
        return b"csv", "summary"

    handler._handle = types.MethodType(fake_inner, handler)
    if outcome == "success":
        assert await handler.handle("task", [], object()) == (b"csv", "summary")
    elif outcome == "failure":
        with pytest.raises(RuntimeError, match="expected failure"):
            await handler.handle("task", [], object())
    else:
        task = asyncio.create_task(handler.handle("task", [], object()))
        while not extracted:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert extracted
    assert not extracted[0].exists()
