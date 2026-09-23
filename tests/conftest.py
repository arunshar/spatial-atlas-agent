"""
Spatial Atlas: Test Configuration

Shared fixtures for all test modules.
"""

import io
import socket

import pytest

from config import Config
from llm import LLMClient


def pytest_addoption(parser):
    parser.addoption(
        "--agent-url",
        default="http://localhost:9019",
        help="Base URL of the running agent container",
    )


def pytest_configure(config):
    for marker in ("unit", "integration", "smoke", "e2e", "gpu", "network", "slow"):
        config.addinivalue_line("markers", marker)


@pytest.fixture(autouse=True)
def _no_network_in_unit(request, monkeypatch):
    """Hermeticity guard: @pytest.mark.unit tests may not reach the network, so an
    accidental real OpenAI/HTTP call fails fast (Google 'no network in small tests').

    We block name resolution + outbound connect (the paths httpx/openai use) rather
    than the socket constructor, so asyncio's event-loop self-pipe still works."""
    if request.node.get_closest_marker("unit") is None:
        return

    def _blocked(*_a, **_k):
        raise RuntimeError("network blocked in @pytest.mark.unit test")

    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


@pytest.fixture
def png_bytes():
    """Tiny valid PNG for image-decoding paths."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (120, 120, 120)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def agent_url(request):
    return request.config.getoption("--agent-url")


@pytest.fixture
def config():
    """Provide a default config for testing."""
    return Config()


@pytest.fixture
def llm(config):
    """Provide an LLM client (requires API keys in env)."""
    return LLMClient(config)
