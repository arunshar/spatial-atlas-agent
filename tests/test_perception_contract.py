"""Contract test: the metric backend's output must satisfy the SpatialScene shape
that the downstream FieldWorkReasoner + AnswerFormatter consume unchanged."""

import numpy as np
import pytest
from fakes import FakePerFrameMask, FakeReconstructTool, FakeSAM3

from config import Config
from fieldwork.perception import MetricPerceptionBackend

pytestmark = pytest.mark.unit


def test_scene_factsheet_carries_measured_distance(png_bytes):
    points = np.zeros((8, 8, 3), dtype=float)
    left = np.zeros((8, 8), dtype=bool)
    right = np.zeros((8, 8), dtype=bool)
    left[:, :4] = True
    right[:, 4:] = True
    points[left] = np.column_stack([np.zeros(32), np.linspace(1.2, 2.2, 32), np.zeros(32)])
    points[right] = np.column_stack([np.full(32, 3.0), np.linspace(0.0, 1.0, 32), np.full(32, 4.0)])
    b = MetricPerceptionBackend(Config())
    b._sam3 = FakeSAM3(
        {
            "worker": FakePerFrameMask(1, masks=left[np.newaxis, np.newaxis]),
            "forklift": FakePerFrameMask(1, masks=right[np.newaxis, np.newaxis]),
        }
    )
    b._recon = FakeReconstructTool(points)
    scene = b.ground(png_bytes, ["worker", "forklift"])

    # to_fact_sheet() is fed verbatim into the reasoner prompt; it must be
    # non-empty and contain the measured distance.
    sheet = scene.to_fact_sheet()
    assert sheet
    assert "5.0" in sheet
    # downstream-consumed properties exist
    assert scene.entity_count == 2
    assert hasattr(scene, "violation_count")
