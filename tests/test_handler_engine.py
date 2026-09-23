"""Integration tests for the FieldWork handler's engine selection + fallback.

All sub-components are mocked; we assert which scene path runs under each
ATLAS_FIELDWORK_ENGINE setting and that the metric path degrades gracefully."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from fieldwork.spatial import SpatialEntity, SpatialRelation, SpatialScene

pytestmark = pytest.mark.integration


def _handler(engine):
    from config import Config
    from fieldwork.handler import FieldWorkHandler

    cfg = Config()
    cfg.fieldwork_engine = engine

    h = FieldWorkHandler.__new__(FieldWorkHandler)
    h.config = cfg
    h.llm = MagicMock()
    h.parser = MagicMock()
    h.parser.parse.return_value = MagicMock(query="distance?", output_format="text")
    h.vision = MagicMock()
    h.vision.process_file = AsyncMock(return_value="ctx")
    h.vision._decode_data = lambda d: b"imgbytes"
    h.spatial = MagicMock()
    h.spatial.build_scene = AsyncMock(
        return_value=MagicMock(entity_count=1, relations=[], violation_count=0)
    )
    h.reasoner = MagicMock()
    h.reasoner.reason = AsyncMock(return_value="2.0 m")
    h.formatter = MagicMock()
    h.formatter.format_answer = lambda a, f: a
    return h


def _updater():
    up = MagicMock()
    up.update_status = AsyncMock()
    return up


def _metric_scene(entity_count=2):
    scene = SpatialScene()
    if entity_count >= 1:
        scene.add_entity(SpatialEntity("a", "Region 0", (0.0, 0.0)))
    if entity_count >= 2:
        scene.add_entity(SpatialEntity("b", "Region 1", (3.0, 4.0)))
        scene.add_relation(SpatialRelation("a", "distance_to", "b", 5.0))
    return scene


def _qspatial_scene_and_evidence():
    from fieldwork import perception

    scene = SpatialScene()
    provenance = "sam3+da3-horizontal-surface-gap"
    scene.add_entity(SpatialEntity("a", "object a", attributes={"source": provenance}))
    scene.add_entity(SpatialEntity("b", "object b", attributes={"source": provenance}))
    scene.add_relation(SpatialRelation("a", "horizontal_surface_gap", "b", 0.05))
    evidence = {
        "fallback_used": False,
        "geometry_valid": True,
        "geometry_status": "ok",
        "parser": {"contract_sha256": perception.QSPATIAL_PARSER_PROTOCOL_SHA256},
        "geometry": {
            "contract_sha256": perception.QSPATIAL_GEOMETRY_PROTOCOL_SHA256,
            "gap_m": 0.05,
            "directed_q05_m": [0.05, 0.06],
            "voxel_keys_sha256": ["1" * 64, "2" * 64],
        },
    }
    return scene, evidence


def test_constructor_initializes_engine_state():
    from config import Config
    from fieldwork.handler import FieldWorkHandler

    handler = FieldWorkHandler(Config(), MagicMock())

    assert handler.last_fieldwork_engine is None
    assert handler.last_metric_grounding is None
    assert handler.last_metric_evidence is None
    assert handler.vision.max_video_frames == handler.config.max_video_frames


@pytest.mark.asyncio
async def test_scenegraph_path_skips_metric():
    h = _handler("scenegraph")
    out = await h.handle("t", [("a.jpg", "image/jpeg", b"x")], _updater())
    assert out == "2.0 m"
    h.spatial.build_scene.assert_awaited_once()
    assert h.last_fieldwork_engine == "scenegraph"


@pytest.mark.asyncio
async def test_metric_success_uses_metric_scene(monkeypatch):
    h = _handler("metric")
    from fieldwork import perception

    fake_scene = _metric_scene()
    monkeypatch.setattr(perception, "build_metric_scene", AsyncMock(return_value=fake_scene))

    await h.handle("t", [("a.jpg", "image/jpeg", b"x")], _updater())

    h.spatial.build_scene.assert_not_awaited()  # metric path won
    _, kwargs = h.reasoner.reason.call_args
    assert kwargs["scene"] is fake_scene
    assert h.last_fieldwork_engine == "metric"
    assert h.last_metric_grounding == "sam3+da3"


@pytest.mark.asyncio
async def test_metric_forwards_raw_image_and_exact_regions(monkeypatch):
    h = _handler("metric")
    from fieldwork import perception

    build = AsyncMock(return_value=_metric_scene())
    monkeypatch.setattr(perception, "build_metric_scene", build)
    regions = [{"size": [2, 2], "counts": "encoded"}]

    await h.handle(
        "t",
        [("overlay.png", "image/png", b"overlay")],
        _updater(),
        metric_image_bytes=b"raw",
        metric_regions=regions,
    )

    assert h.vision.process_file.await_args.args[2] == b"overlay"
    assert build.await_args.args[:4] == (
        "distance?",
        [b"raw"],
        h.llm,
        h.config,
    )
    assert build.await_args.kwargs == {"regions": regions}
    assert h.last_metric_grounding == "rle+da3"


@pytest.mark.asyncio
async def test_metric_failure_falls_back_to_scenegraph(monkeypatch):
    h = _handler("metric")
    from fieldwork import perception

    monkeypatch.setattr(
        perception,
        "build_metric_scene",
        AsyncMock(side_effect=RuntimeError("no gpu server")),
    )

    await h.handle("t", [("a.jpg", "image/jpeg", b"x")], _updater())

    h.spatial.build_scene.assert_awaited_once()  # fell back cleanly
    assert h.last_fieldwork_engine == "scenegraph"


@pytest.mark.asyncio
async def test_exact_region_failure_preserves_production_fallback(monkeypatch):
    h = _handler("metric")
    from fieldwork import perception

    build = AsyncMock(side_effect=RuntimeError("bad RLE"))
    monkeypatch.setattr(perception, "build_metric_scene", build)

    await h.handle(
        "t",
        [("overlay.png", "image/png", b"overlay")],
        _updater(),
        metric_image_bytes=b"raw",
        metric_regions=[{"counts": "bad"}],
    )

    h.spatial.build_scene.assert_awaited_once()
    assert h.vision.process_file.await_args.args[2] == b"overlay"
    assert build.await_args.args[1] == [b"raw"]
    assert h.last_fieldwork_engine == "scenegraph"
    assert h.last_metric_grounding == "rle+da3"


@pytest.mark.asyncio
async def test_metric_strict_failure_is_not_mislabeled_as_scenegraph(monkeypatch):
    h = _handler("metric")
    h.config.fieldwork_metric_strict = True
    from fieldwork import perception

    monkeypatch.setattr(
        perception,
        "build_metric_scene",
        AsyncMock(side_effect=RuntimeError("no gpu server")),
    )

    with pytest.raises(RuntimeError, match="no gpu server"):
        await h.handle("t", [("a.jpg", "image/jpeg", b"x")], _updater())

    h.spatial.build_scene.assert_not_awaited()
    assert h.last_fieldwork_engine == "metric_failed"


@pytest.mark.parametrize("entity_count", [0, 1])
@pytest.mark.asyncio
async def test_metric_strict_rejects_unusable_geometry(monkeypatch, entity_count):
    h = _handler("metric")
    h.config.fieldwork_metric_strict = True
    from fieldwork import perception

    monkeypatch.setattr(
        perception,
        "build_metric_scene",
        AsyncMock(return_value=_metric_scene(entity_count)),
    )

    with pytest.raises(RuntimeError, match="at least two finite grounded entities"):
        await h.handle("t", [("a.jpg", "image/jpeg", b"x")], _updater())

    h.spatial.build_scene.assert_not_awaited()
    assert h.last_fieldwork_engine == "metric_failed"
    assert h.last_metric_grounding == "sam3+da3"


@pytest.mark.asyncio
async def test_qspatial_metric_uses_frozen_gap_path(monkeypatch):
    h = _handler("metric")
    from fieldwork import perception

    scene, evidence = _qspatial_scene_and_evidence()
    build = AsyncMock(return_value=(scene, evidence))
    monkeypatch.setattr(perception, "build_qspatial_gap_scene", build)
    metadata = {"row_id": 0}

    result = await h.handle(
        "t",
        [("a.jpg", "image/jpeg", b"x")],
        _updater(),
        metric_protocol="qspatial-horizontal-gap-v1",
        metric_sample_metadata=metadata,
        metric_question="distance?",
    )

    assert result == "2.0 m"
    assert build.await_args.args[:3] == ("distance?", [b"x"], h.config)
    assert build.await_args.kwargs == {"sample_metadata": metadata}
    h.spatial.build_scene.assert_not_awaited()
    assert h.last_fieldwork_engine == "metric"
    assert h.last_metric_grounding == "sam3+da3-horizontal-surface-gap"
    assert h.last_metric_evidence is evidence


@pytest.mark.asyncio
async def test_qspatial_expected_invalid_returns_sentinel_without_fallback(monkeypatch):
    h = _handler("metric")
    from fieldwork import perception

    evidence = {
        "geometry_valid": False,
        "geometry_status": "invalid",
        "fallback_used": False,
        "geometry": {"invalid_reason": "sam_empty_a"},
    }
    monkeypatch.setattr(
        perception,
        "build_qspatial_gap_scene",
        AsyncMock(return_value=(None, evidence)),
    )

    result = await h.handle(
        "t",
        [("a.jpg", "image/jpeg", b"x")],
        _updater(),
        metric_protocol="qspatial-horizontal-gap-v1",
        metric_sample_metadata={"row_id": 0},
        metric_question="distance?",
    )

    assert result == "measurement unavailable"
    h.spatial.build_scene.assert_not_awaited()
    h.reasoner.reason.assert_not_awaited()
    assert h.last_fieldwork_engine == "metric"
    assert h.last_metric_evidence is evidence


@pytest.mark.asyncio
async def test_qspatial_service_error_never_falls_back(monkeypatch):
    h = _handler("metric")
    from fieldwork import perception

    monkeypatch.setattr(
        perception,
        "build_qspatial_gap_scene",
        AsyncMock(side_effect=RuntimeError("service_error: unavailable")),
    )

    with pytest.raises(RuntimeError, match="service_error"):
        await h.handle(
            "t",
            [("a.jpg", "image/jpeg", b"x")],
            _updater(),
            metric_protocol="qspatial-horizontal-gap-v1",
            metric_sample_metadata={"row_id": 0},
            metric_question="distance?",
        )

    h.spatial.build_scene.assert_not_awaited()
    assert h.last_fieldwork_engine == "metric_failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_question", [None, "", "   "])
async def test_qspatial_metric_requires_the_raw_question(monkeypatch, missing_question):
    """Regression: the composed goal must never reach the parser."""
    h = _handler("metric")
    from fieldwork import perception

    build = AsyncMock()
    monkeypatch.setattr(perception, "build_qspatial_gap_scene", build)

    with pytest.raises(RuntimeError, match="raw benchmark question"):
        await h.handle(
            "t",
            [("a.jpg", "image/jpeg", b"x")],
            _updater(),
            metric_protocol="qspatial-horizontal-gap-v1",
            metric_sample_metadata={"row_id": 0},
            metric_question=missing_question,
        )

    build.assert_not_awaited()
    assert h.last_fieldwork_engine == "metric_failed"


@pytest.mark.asyncio
async def test_qspatial_metric_parses_the_raw_question_not_the_goal_query(monkeypatch):
    """The frozen parser and evidence hashes consume metric_question, not task.query."""
    h = _handler("metric")
    from fieldwork import perception

    scene, evidence = _qspatial_scene_and_evidence()
    build = AsyncMock(return_value=(scene, evidence))
    monkeypatch.setattr(perception, "build_qspatial_gap_scene", build)
    raw = "what is the gap between the vase and the lamp?"

    result = await h.handle(
        "t",
        [("a.jpg", "image/jpeg", b"x")],
        _updater(),
        metric_protocol="qspatial-horizontal-gap-v1",
        metric_sample_metadata={"row_id": 0},
        metric_question=raw,
    )

    assert result == "2.0 m"
    assert build.await_args.args[0] == raw
    assert build.await_args.args[0] != "distance?"
