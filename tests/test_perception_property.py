"""Property-based tests: distance symmetry + the ground-plane (x, z) invariant.

The (x, z) invariant is the regression guard for the core fix: positions are
measured on the floor plane, so the y (height) coordinate must never affect a
pairwise distance (the old code guessed full coordinates from text)."""

import io
import math

import numpy as np
import pytest
from fakes import FakePerFrameMask, FakeReconstructTool, FakeSAM3
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image

from config import Config
from fieldwork.perception import MetricPerceptionBackend

pytestmark = pytest.mark.unit

# Module-level PNG so @given does not depend on a function-scoped fixture.
_buf = io.BytesIO()
Image.new("RGB", (8, 8), (120, 120, 120)).save(_buf, "PNG")
_PNG = _buf.getvalue()

_coord = st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False)


@given(ax=_coord, ay=_coord, az=_coord, bx=_coord, by=_coord, bz=_coord)
def test_distance_symmetric_and_ground_plane(ax, ay, az, bx, by, bz):
    points = np.zeros((8, 8, 3), dtype=float)
    left = np.zeros((8, 8), dtype=bool)
    right = np.zeros((8, 8), dtype=bool)
    left[:, :4] = True
    right[:, 4:] = True
    points[left] = np.column_stack(
        [np.full(32, ax), np.linspace(ay - 0.5, ay + 0.5, 32), np.full(32, az)]
    )
    points[right] = np.column_stack(
        [np.full(32, bx), np.linspace(by - 0.5, by + 0.5, 32), np.full(32, bz)]
    )
    b = MetricPerceptionBackend(Config())
    b._sam3 = FakeSAM3(
        {
            "a": FakePerFrameMask(1, masks=left[np.newaxis, np.newaxis]),
            "b": FakePerFrameMask(1, masks=right[np.newaxis, np.newaxis]),
        }
    )
    b._recon = FakeReconstructTool(points)
    scene = b.ground(_PNG, ["a", "b"])
    if scene.entity_count < 2:
        return
    d_ab = scene.compute_distance("e0_i0", "e1_i0")
    d_ba = scene.compute_distance("e1_i0", "e0_i0")
    assert d_ab == pytest.approx(d_ba)  # symmetry
    # uses (x, z), not (x, y): y heights are irrelevant to the distance
    assert d_ab == pytest.approx(math.hypot(ax - bx, az - bz))
