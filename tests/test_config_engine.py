"""Config tests for the ATLAS_FIELDWORK_ENGINE flag."""

import importlib

import pytest

pytestmark = pytest.mark.unit


def test_default_engine_is_scenegraph(monkeypatch):
    monkeypatch.delenv("ATLAS_FIELDWORK_ENGINE", raising=False)
    import config as c

    importlib.reload(c)
    cfg = c.Config()
    assert cfg.fieldwork_engine == "scenegraph"
    assert cfg.fieldwork_metric_strict is False
    assert cfg.reconstruct_max_frames == 32


def test_env_override_metric(monkeypatch):
    monkeypatch.setenv("ATLAS_FIELDWORK_ENGINE", "metric")
    import config as c

    importlib.reload(c)
    assert c.Config().fieldwork_engine == "metric"


def test_blank_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("ATLAS_FIELDWORK_ENGINE", "")
    import config as c

    importlib.reload(c)
    assert c.Config().fieldwork_engine == "scenegraph"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("true", True), ("1", True), ("OFF", False), ("0", False)],
)
def test_config_parses_optional_thinking(monkeypatch, raw, expected):
    monkeypatch.setenv("ATLAS_ENABLE_THINKING", raw)
    import config as c

    importlib.reload(c)
    assert c.Config().llm_enable_thinking is expected


def test_config_rejects_invalid_optional_thinking(monkeypatch):
    monkeypatch.setenv("ATLAS_ENABLE_THINKING", "sometimes")
    import config as c

    importlib.reload(c)
    with pytest.raises(ValueError, match="ATLAS_ENABLE_THINKING"):
        c.Config()


def test_model_tiers_and_startup_validation(capsys):
    import config as c

    cfg = c.Config(
        fast_model="provider/fast",
        standard_model="provider/standard",
        strong_model="provider/strong",
        vision_model="provider/vision",
    )

    assert cfg.model_tiers == {
        "fast": "provider/fast",
        "standard": "provider/standard",
        "strong": "provider/strong",
        "vision": "provider/vision",
    }
    cfg.log_resolved_tiers()
    assert "provider/vision" in capsys.readouterr().out


def test_startup_validation_rejects_empty_model():
    import config as c

    with pytest.raises(RuntimeError, match="Model tier 'fast' is empty"):
        c.Config(fast_model="").log_resolved_tiers()


def test_startup_validation_rejects_providerless_model():
    import config as c

    with pytest.raises(RuntimeError, match="has no provider prefix"):
        c.Config(fast_model="model-without-provider").log_resolved_tiers()
