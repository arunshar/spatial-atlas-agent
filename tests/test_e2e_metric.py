"""End-to-end metric perception test (opt-in; requires a live SpatialClaw GPU tool server).

Run on a GPU host after the smoke gate, with the GPU tool server up and the spatial_agent
package importable:
    SPATIAL_ATLAS_E2E=1 pytest tests/test_e2e_metric.py -m e2e
"""
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]


@pytest.mark.skipif(
    os.environ.get("SPATIAL_ATLAS_E2E") != "1",
    reason="set SPATIAL_ATLAS_E2E=1 + a running SpatialClaw GPU tool server",
)
@pytest.mark.asyncio
async def test_metric_distance_on_warehouse_png():
    from config import Config
    from fieldwork.perception import build_metric_scene
    from llm import LLMClient

    img = (Path(__file__).parent / "test_warehouse.png").read_bytes()
    scene = await build_metric_scene(
        "What is the distance between the worker and the forklift?",
        [img], LLMClient(Config()), Config(),
    )
    dists = [r.distance for r in scene.relations if r.distance is not None]
    assert dists, "expected at least one measured distance"
    assert all(0.1 <= d <= 50.0 for d in dists), f"implausible metric distances: {dists}"
