"""Regression tests for two hazards that would each have broken a live poster demo.

Both were found by reading the code rather than by any failing test, which is why these exist:
neither hazard is reachable from the existing suite, and both fail in ways that look like
something else (a 503 looks like load, a four-hour hang looks like a slow model).
"""

from __future__ import annotations

import os

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from src.server import ConcurrencyLimitMiddleware


def _app_with_limit(maximum: int) -> Starlette:
    async def ok(request):
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/", ok, methods=["GET", "POST"])])
    return ConcurrencyLimitMiddleware(app, maximum=maximum)


def test_read_only_requests_never_consume_a_concurrency_slot() -> None:
    """A QR code produces a burst of GETs; none of them may be rejected.

    The slot is pre-saturated deliberately. TestClient issues requests sequentially, so
    `_active` returns to zero between them and a plain loop would pass even with the
    exemption removed, which a mutation check confirmed. Holding the slot open is what
    actually reproduces the condition a crowd at a poster creates.
    """
    middleware = _app_with_limit(1)
    middleware._active = middleware.maximum  # every slot busy with real work
    client = TestClient(middleware)

    for _ in range(10):
        assert client.get("/").status_code == 200
    for method in ("head", "options"):
        assert getattr(client, method)("/").status_code in (200, 405)


def test_the_limit_still_applies_to_work_bearing_methods() -> None:
    """The exemption must not disarm the protection it was added to preserve.

    POST does real model work, so it must still be bounded. This is the half of the
    behaviour that had to survive the fix.
    """
    middleware = _app_with_limit(1)
    # Occupy the single slot exactly as a live request would.
    middleware._active = middleware.maximum
    client = TestClient(middleware)
    assert client.post("/").status_code == 503
    # And a GET still passes while POST is saturated.
    assert client.get("/").status_code == 200


def test_gpu_server_url_is_defaulted_so_discovery_cannot_hang(monkeypatch) -> None:
    """Never leave SPATIALCLAW_GPU_SERVER_URL unset when entering tool discovery.

    With an empty registry, SpatialClaw's upstream `_get_server_url` enters a
    1440-iteration, ten-second-interval wait, which is about four hours. The fast
    branch that this variable selects comes from a local modification of
    SpatialClaw, and NVIDIA's public release does not read the variable.
    """
    from src.fieldwork import perception

    monkeypatch.delenv("SPATIALCLAW_GPU_SERVER_URL", raising=False)
    backend = perception.MetricPerceptionBackend.__new__(
        perception.MetricPerceptionBackend
    )
    backend._recon = None
    backend._sam3 = None
    backend.config = type("C", (), {"reconstruct_max_frames": 8})()

    # spatial_agent is not installed in the test env, so the lazy import raises. The
    # env var must already be set by the time it does.
    # Both failure paths are in scope on purpose. Without spatial_agent installed the
    # lazy import raises ImportError; with it installed but no reachable tool server,
    # _ensure_tools raises RuntimeError. The assertion under test is that the env var
    # is already set by the time either one fires, so narrowing to one would make the
    # test pass or fail for reasons unrelated to what it checks.
    with pytest.raises((ImportError, RuntimeError)):
        backend._ensure_tools()

    assert os.environ.get("SPATIALCLAW_GPU_SERVER_URL"), "must not be left unset"


def test_an_operator_supplied_gpu_url_is_never_overwritten(monkeypatch) -> None:
    """GPU benchmark harnesses export an operator endpoint, so the guard must defer.

    `setdefault` is load-bearing here. Overwriting would silently point a real
    benchmark run at a closed port, which is far worse than the hang it prevents.
    """
    from src.fieldwork import perception

    operator_url = "http://192.0.2.10:18072"
    monkeypatch.setenv("SPATIALCLAW_GPU_SERVER_URL", operator_url)
    backend = perception.MetricPerceptionBackend.__new__(
        perception.MetricPerceptionBackend
    )
    backend._recon = None
    backend._sam3 = None
    backend.config = type("C", (), {"reconstruct_max_frames": 8})()

    # Both failure paths are in scope on purpose. Without spatial_agent installed the
    # lazy import raises ImportError; with it installed but no reachable tool server,
    # _ensure_tools raises RuntimeError. The assertion under test is that the env var
    # is already set by the time either one fires, so narrowing to one would make the
    # test pass or fail for reasons unrelated to what it checks.
    with pytest.raises((ImportError, RuntimeError)):
        backend._ensure_tools()

    assert os.environ["SPATIALCLAW_GPU_SERVER_URL"] == operator_url
