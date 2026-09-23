"""Tests for the public HTTP safety middleware."""

import asyncio
from pathlib import Path

import pytest

from server import (
    BearerAuthMiddleware,
    ConcurrencyLimitMiddleware,
    RequestSizeLimitMiddleware,
    _build_protected_app,
    _validate_execution_security,
    _validate_public_binding,
)


class DrainingApp:
    def __init__(self):
        self.calls = 0
        self.body = b""

    async def __call__(self, _scope, receive, send):
        self.calls += 1
        while True:
            message = await receive()
            self.body += message.get("body", b"")
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


async def _invoke(body_parts, *, limit, content_length=None):
    app = DrainingApp()
    middleware = RequestSizeLimitMiddleware(app, limit)
    headers = []
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode("ascii")))
    scope = {"type": "http", "method": "POST", "path": "/", "headers": headers}
    messages = [
        {
            "type": "http.request",
            "body": body,
            "more_body": index < len(body_parts) - 1,
        }
        for index, body in enumerate(body_parts)
    ]
    sent = []

    async def receive():
        return messages.pop(0)

    async def send(message):
        sent.append(message)

    await middleware(scope, receive, send)
    status = next(message["status"] for message in sent if message["type"] == "http.response.start")
    return app, status


@pytest.mark.asyncio
async def test_declared_request_over_limit_is_rejected_before_downstream():
    app, status = await _invoke([b"ignored"], limit=4, content_length=5)

    assert status == 413
    assert app.calls == 0


@pytest.mark.asyncio
async def test_streamed_request_crossing_limit_is_rejected():
    app, status = await _invoke([b"123", b"45"], limit=4)

    assert status == 413
    assert app.calls == 0


@pytest.mark.asyncio
async def test_streaming_responder_never_starts_before_body_is_validated():
    downstream_started = False

    class EarlyResponder:
        async def __call__(self, _scope, _receive, send):
            nonlocal downstream_started
            downstream_started = True
            await send({"type": "http.response.start", "status": 200, "headers": []})

    messages = [
        {"type": "http.request", "body": b"123", "more_body": True},
        {"type": "http.request", "body": b"45", "more_body": False},
    ]
    sent = []

    async def receive():
        return messages.pop(0)

    async def send(message):
        sent.append(message)

    middleware = RequestSizeLimitMiddleware(EarlyResponder(), 4)
    await middleware({"type": "http", "method": "POST", "headers": []}, receive, send)

    assert downstream_started is False
    assert [m["status"] for m in sent if m["type"] == "http.response.start"] == [413]


@pytest.mark.asyncio
async def test_request_at_limit_reaches_application_unchanged():
    app, status = await _invoke([b"12", b"34"], limit=4, content_length=4)

    assert status == 200
    assert app.calls == 1
    assert app.body == b"1234"


@pytest.mark.asyncio
async def test_replayed_body_delegates_disconnect_to_server_receive():
    seen = []

    class StreamingApp:
        async def __call__(self, _scope, receive, send):
            seen.append(await receive())
            seen.append(await receive())
            await send({"type": "http.response.start", "status": 200, "headers": []})

    messages = [
        {"type": "http.request", "body": b"body", "more_body": False},
        {"type": "http.disconnect"},
    ]

    async def receive():
        return messages.pop(0)

    async def send(_message):
        return None

    await RequestSizeLimitMiddleware(StreamingApp(), 8)(
        {"type": "http", "method": "POST", "headers": []}, receive, send
    )

    assert seen == [
        {"type": "http.request", "body": b"body", "more_body": False},
        {"type": "http.disconnect"},
    ]


def test_public_landing_page_contains_no_unexecuted_result_table():
    source = Path(__file__).parents[1].joinpath("src/server.py").read_text(encoding="utf-8")

    for unsupported in (
        "+21-24 pts",
        "+7-8 pts",
        "82%",
        "35-40%",
        "Deterministic spatial scene graphs replace VLM hallucination",
    ):
        assert unsupported not in source
    assert "no result is reported" in source


def test_public_middleware_order_limits_before_auth_and_body_buffering(monkeypatch):
    monkeypatch.setenv("ATLAS_MAX_REQUEST_BYTES", "1024")
    monkeypatch.setenv("ATLAS_MAX_CONCURRENT_REQUESTS", "2")

    protected = _build_protected_app(DrainingApp(), "t" * 32)

    assert isinstance(protected, ConcurrencyLimitMiddleware)
    assert isinstance(protected.app, BearerAuthMiddleware)
    assert isinstance(protected.app.app, RequestSizeLimitMiddleware)


async def _simple_call(app, *, method="POST", authorization=None):
    headers = []
    if authorization is not None:
        raw = authorization if isinstance(authorization, bytes) else authorization.encode("utf-8")
        headers.append((b"authorization", raw))
    scope = {"type": "http", "method": method, "path": "/", "headers": headers}
    received = False
    sent = []

    async def receive():
        nonlocal received
        if received:
            return {"type": "http.disconnect"}
        received = True
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    return next(message["status"] for message in sent if message["type"] == "http.response.start")


@pytest.mark.asyncio
async def test_configured_bearer_token_protects_write_requests():
    token = "t" * 32
    app = BearerAuthMiddleware(DrainingApp(), token)

    assert await _simple_call(app) == 401
    assert await _simple_call(app, authorization=f"Bearer {token}") == 200
    assert await _simple_call(app, method="GET") == 200
    # A non-ASCII or non-UTF-8 header must be rejected with 401, never crash with a 500.
    assert await _simple_call(app, authorization="Bearer \u00e9".encode("utf-8")) == 401
    assert await _simple_call(app, authorization=b"Bearer \xff" + b"t" * 31) == 401


@pytest.mark.asyncio
async def test_non_ascii_configured_token_still_authenticates():
    token = "é" * 32
    app = BearerAuthMiddleware(DrainingApp(), token)

    assert await _simple_call(app, authorization=f"Bearer {token}") == 200
    assert await _simple_call(app, authorization="Bearer " + "t" * 32) == 401


@pytest.mark.asyncio
async def test_concurrency_limit_rejects_excess_request():
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingApp:
        async def __call__(self, _scope, _receive, send):
            entered.set()
            await release.wait()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

    middleware = ConcurrencyLimitMiddleware(BlockingApp(), maximum=1)
    first = asyncio.create_task(_simple_call(middleware))
    await entered.wait()
    assert await _simple_call(middleware) == 503
    release.set()
    assert await first == 200


def test_mle_execution_requires_isolated_worker_and_auth(monkeypatch):
    monkeypatch.setenv("ATLAS_ENABLE_MLEBENCH_CODE_EXECUTION", "true")
    monkeypatch.delenv("ATLAS_TRUSTED_ISOLATED_WORKER", raising=False)
    with pytest.raises(RuntimeError, match="isolated-worker"):
        _validate_execution_security("t" * 32)

    monkeypatch.setenv("ATLAS_TRUSTED_ISOLATED_WORKER", "true")
    with pytest.raises(RuntimeError, match="ATLAS_BEARER_TOKEN"):
        _validate_execution_security("")

    _validate_execution_security("t" * 32)


def test_non_loopback_binding_fails_closed_without_auth(monkeypatch):
    monkeypatch.delenv("ATLAS_ALLOW_UNAUTHENTICATED_PUBLIC", raising=False)
    with pytest.raises(RuntimeError, match="non-loopback"):
        _validate_public_binding("0.0.0.0", "")

    _validate_public_binding("127.0.0.1", "")
    _validate_public_binding("0.0.0.0", "t" * 32)
    monkeypatch.setenv("ATLAS_ALLOW_UNAUTHENTICATED_PUBLIC", "true")
    _validate_public_binding("0.0.0.0", "")
