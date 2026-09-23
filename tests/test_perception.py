"""Unit tests for the metric perception backend (hermetic; fakes injected)."""

# The QSpatial questions in this file are copied from Q-Spatial Bench (QSpatial++) by
# Yuan-Hong Liao, Rafid Mahmood, Sanja Fidler, and David Acuna. That text stays under the
# dataset's Apache-2.0 license, and the MIT License of this repository does not cover it.

import asyncio
import sys
import threading
import types
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fakes import FakePerFrameMask, FakeReconstructTool, FakeSAM3

from config import Config
from fieldwork import perception
from fieldwork.perception import (
    MetricPerceptionBackend,
    build_metric_scene,
    validate_metric_height_scene,
    validate_metric_scene,
)
from fieldwork.spatial import SpatialEntity, SpatialScene

pytestmark = pytest.mark.unit

# A stand-in slice digest. The real one belongs to a private control artifact and is supplied
# at run time through ATLAS_QSPATIAL_GAP_SLICE_SHA256.
STAND_IN_SLICE_SHA256 = "5" * 64


@pytest.fixture(autouse=True)
def _configured_gap_slice_digest(monkeypatch):
    monkeypatch.setattr(perception, "QSPATIAL_GAP_SLICE_SHA256", STAND_IN_SLICE_SHA256)


def test_configured_sha256_accepts_only_a_64_hex_digest(monkeypatch):
    name = "ATLAS_TEST_CONFIGURED_DIGEST"
    monkeypatch.delenv(name, raising=False)
    assert perception._configured_sha256(name) == ""
    monkeypatch.setenv(name, "  " + "AB" * 32 + "\n")
    assert perception._configured_sha256(name) == "ab" * 32
    monkeypatch.setenv(name, "ab" * 31)
    assert perception._configured_sha256(name) == ""
    monkeypatch.setenv(name, "zz" * 32)
    assert perception._configured_sha256(name) == ""


def test_qspatial_evidence_fails_closed_without_configured_slice_digest(monkeypatch):
    parser_record = perception.parse_qspatial_gap_question(
        "What is the minimum distance between the coffee grinder and the base of the "
        "gooseneck kettle in the image?",
        row_id=0,
    )
    metadata = _qspatial_metadata(0)
    monkeypatch.setattr(perception, "QSPATIAL_GAP_SLICE_SHA256", "")

    with pytest.raises(RuntimeError, match="slice hash is not configured"):
        perception._qspatial_evidence_base(b"question", b"image", parser_record, metadata)


def _backend(by_phrase):
    b = MetricPerceptionBackend(Config())
    b._sam3 = FakeSAM3(by_phrase)  # set private handles so _ensure_tools no-ops
    b._recon = FakeReconstructTool()
    return b


def _split_point_geometry():
    points = np.zeros((8, 8, 3), dtype=float)
    confidence = np.ones((8, 8), dtype=float)
    left = np.zeros((8, 8), dtype=bool)
    right = np.zeros((8, 8), dtype=bool)
    left[:, :4] = True
    right[:, 4:] = True
    points[left, 0] = 0.0
    points[left, 1] = np.linspace(1.2, 2.2, int(left.sum()))
    points[left, 2] = 0.0
    points[right, 0] = 3.0
    points[right, 1] = np.linspace(0.0, 1.0, int(right.sum()))
    points[right, 2] = 4.0
    return points, confidence, left, right


def _horizontal_gap_geometry():
    rows, columns = np.indices((12, 24))
    points = np.zeros((12, 24, 3), dtype=float)
    points[..., 0] = columns * 0.01
    points[..., 2] = rows * 0.01
    confidence = np.ones((12, 24), dtype=float)
    left = np.zeros((12, 24), dtype=bool)
    right = np.zeros((12, 24), dtype=bool)
    left[1:11, 1:11] = True
    right[1:11, 13:23] = True
    return points, confidence, left, right


def test_horizontal_surface_gap_follows_frozen_geometry_contract():
    points, confidence, left, right = _horizontal_gap_geometry()

    result = perception._measure_horizontal_surface_gap(
        left,
        right,
        points,
        confidence,
    )

    assert result["gap_m"] == pytest.approx(0.05)
    assert result["directed_gap_a_to_b_m"] == pytest.approx(0.05)
    assert result["directed_gap_b_to_a_m"] == pytest.approx(0.05)
    assert result["mask_overlap_coefficient"] == 0.0
    assert result["eroded_mask_pixel_counts"] == [64, 64]
    assert result["voxel_counts"] == [64, 64]
    assert len(result["voxel_keys_sha256"]) == 2
    assert all(len(value) == 64 for value in result["voxel_keys_sha256"])
    assert result["protocol"] == "qspatial-horizontal-gap-v1"
    assert result["confidence_threshold"] == 0.3
    assert result["voxel_size_m"] == 0.005
    assert result["nearest_percentile"] == 5.0


def test_horizontal_surface_gap_is_deterministic_and_order_invariant():
    points, confidence, left, right = _horizontal_gap_geometry()

    first = perception._measure_horizontal_surface_gap(left, right, points, confidence)
    second = perception._measure_horizontal_surface_gap(left, right, points, confidence)
    reversed_pair = perception._measure_horizontal_surface_gap(
        right,
        left,
        points,
        confidence,
    )

    assert first == second
    assert first["gap_m"] == reversed_pair["gap_m"]
    assert first["directed_gap_a_to_b_m"] == reversed_pair["directed_gap_b_to_a_m"]
    assert first["directed_gap_b_to_a_m"] == reversed_pair["directed_gap_a_to_b_m"]
    assert first["voxel_keys_sha256"] == list(reversed(reversed_pair["voxel_keys_sha256"]))


def test_horizontal_surface_gap_resizes_then_erodes_once():
    points, confidence, left, right = _horizontal_gap_geometry()

    result = perception._measure_horizontal_surface_gap(
        left[::2, ::2],
        right[::2, ::2],
        points,
        confidence,
    )

    assert result["point_map_shape"] == [12, 24]
    assert result["input_mask_shapes"] == [[6, 12], [6, 12]]
    assert result["erosion_iterations"] == 1
    assert all(count >= 32 for count in result["voxel_counts"])


def test_horizontal_surface_gap_rejects_overlapping_masks():
    points, confidence, left, _ = _horizontal_gap_geometry()

    with pytest.raises(RuntimeError, match="overlap"):
        perception._measure_horizontal_surface_gap(left, left.copy(), points, confidence)


def test_horizontal_surface_gap_overlap_boundary_is_inclusive():
    rows, columns = np.indices((30, 50))
    points = np.zeros((30, 50, 3), dtype=float)
    points[..., 0] = columns * 0.01
    points[..., 2] = rows * 0.01
    confidence = np.ones((30, 50), dtype=float)
    mask_a = np.zeros((30, 50), dtype=bool)
    mask_b = np.zeros((30, 50), dtype=bool)
    mask_a[2:14, 2:14] = True
    mask_b[16:28, 30:42] = True
    mask_b[5:8, 5:12] = True

    with pytest.raises(RuntimeError, match="mask_overlap_excess"):
        perception._measure_horizontal_surface_gap(mask_a, mask_b, points, confidence)


def test_horizontal_surface_gap_removes_subthreshold_shared_pixels():
    rows, columns = np.indices((30, 50))
    points = np.zeros((30, 50, 3), dtype=float)
    points[..., 0] = columns * 0.01
    points[..., 2] = rows * 0.01
    confidence = np.ones((30, 50), dtype=float)
    mask_a = np.zeros((30, 50), dtype=bool)
    mask_b = np.zeros((30, 50), dtype=bool)
    mask_a[2:14, 2:14] = True
    mask_b[16:28, 30:42] = True
    mask_b[5:8, 5:11] = True

    result = perception._measure_horizontal_surface_gap(
        mask_a,
        mask_b,
        points,
        confidence,
    )

    assert result["mask_overlap_coefficient"] == pytest.approx(0.04)
    assert result["mask_overlap_intersection_pixels"] == 4
    assert result["eroded_mask_pixel_counts"] == [96, 100]


def test_horizontal_surface_gap_confidence_threshold_is_strict():
    points, confidence, left, right = _horizontal_gap_geometry()
    confidence[left] = 0.3

    with pytest.raises(RuntimeError, match="minimum 32"):
        perception._measure_horizontal_surface_gap(left, right, points, confidence)


def test_horizontal_surface_gap_rejects_too_few_finite_voxels():
    points, confidence, left, right = _horizontal_gap_geometry()
    points[left] = np.nan

    with pytest.raises(RuntimeError, match="minimum 32"):
        perception._measure_horizontal_surface_gap(left, right, points, confidence)


@pytest.mark.parametrize("invalid_gap", [0.0, 21.0])
def test_horizontal_surface_gap_rejects_out_of_range_geometry(invalid_gap):
    points, confidence, left, right = _horizontal_gap_geometry()
    left_points = points[left].copy()
    points[right, 0] = left_points[:, 0] + invalid_gap
    points[right, 2] = left_points[:, 2]

    with pytest.raises(RuntimeError, match="outside frozen range"):
        perception._measure_horizontal_surface_gap(left, right, points, confidence)


def _qspatial_metadata(row_id):
    return {
        "measurement_family": "horizontal_surface_gap",
        "row_id": row_id,
        "cluster_id": f"qsg-{row_id:016d}",
        "slice_id": "qspatial_gap97_v1",
        "slice_sha256": perception.QSPATIAL_GAP_SLICE_SHA256,
        "output_format": "frozen test format",
    }


@pytest.mark.parametrize(
    ("row_id", "question", "mode"),
    [
        (
            0,
            ("What is the minimum distance between the coffee grinder and the base of "
            "the gooseneck kettle in the image?"),
            "distinct_pair",
        ),
        (
            23,
            "What is the minimum distance between the two orange chairs next to the window?",
            "repeated_instance",
        ),
        (
            35,
            ("What is the minimum distance between the wooden table and the bed frame near "
            "the camera?"),
            "distinct_pair",
        ),
        (
            46,
            "What is the minimum distance between the pillar and the left wall in the image?",
            "distinct_pair",
        ),
        (
            49,
            "What is the minimum distance between the bollards in the image?",
            "repeated_instance",
        ),
        (
            57,
            ("What is the minimum distance between the parking meter and parking sign post "
            "in the image?"),
            "distinct_pair",
        ),
        (
            65,
            "What is the gap between the two traffic barrels at the front in the image?",
            "repeated_instance",
        ),
        (
            85,
            "What is the horizontal gap between the front tires of the two bikes?",
            "repeated_instance",
        ),
    ],
)
def test_qspatial_parser_frozen_pilot_taxonomy(row_id, question, mode):
    first = perception.parse_qspatial_gap_question(question, row_id=row_id)
    second = perception.parse_qspatial_gap_question(question, row_id=row_id)

    assert first == second
    assert first["mode"] == mode
    assert first["contract_sha256"] == perception.QSPATIAL_PARSER_PROTOCOL_SHA256


def test_qspatial_parser_rejects_negative_space_without_tools(png_bytes):
    question = (
        "What is the horizontal gap in the remaining area of the bookshelf next to the last "
        "book on the left side?"
    )
    parsed = perception.parse_qspatial_gap_question(question, row_id=80)
    backend = MetricPerceptionBackend(Config())

    scene, evidence = backend.ground_horizontal_surface_gap(
        png_bytes,
        question,
        parsed,
        _qspatial_metadata(80),
    )

    assert scene is None
    assert evidence["geometry_valid"] is False
    assert evidence["geometry"]["invalid_reason"] == "unsupported_negative_space"
    assert evidence["fallback_used"] is False


@pytest.mark.parametrize(
    ("question", "row_id", "message"),
    [
        ("", None, "nonempty string"),
        (
            ("What is the horizontal gap in the remaining area of the bookshelf next to the last "
            "book on the left side?"),
            0,
            "unexpected row ID",
        ),
        ("How far apart are the two objects?", None, "outside the frozen QSpatial grammar"),
    ],
)
def test_qspatial_parser_rejects_invalid_or_out_of_contract_questions(question, row_id, message):
    with pytest.raises(RuntimeError, match=message):
        perception.parse_qspatial_gap_question(question, row_id=row_id)


def test_frozen_parser_rejects_the_composed_goal_query_but_accepts_the_raw_question():
    """Regression for an earlier metric pilot failure.

    eval_bench composes the question plus the benchmark output-format instruction
    inside the # Question section; GoalParser extracts the whole section as the
    query. The frozen grammar fullmatch can never accept that composed string, so
    the metric path must parse the raw benchmark question instead.
    """
    import types

    import eval_bench as driver
    from fieldwork.parser import GoalParser

    raw = "What is the gap between the vase and the lamp?"
    sample = types.SimpleNamespace(
        question=raw,
        choices=None,
        answer_type="float",
        question_type="distance",
    )
    goal = driver._build_goal(
        sample,
        "image.png",
        r"Return exactly: \scalar{<positive scalar>} \distance_unit{meter}. "
        "Use one scalar and one distance unit, with no explanation.",
    )
    composed_query = GoalParser().parse(goal).query
    assert composed_query != raw
    assert raw in composed_query
    with pytest.raises(RuntimeError, match="outside the frozen QSpatial grammar"):
        perception.parse_qspatial_gap_question(composed_query)
    record = perception.parse_qspatial_gap_question(raw)
    assert record["mode"] == "distinct_pair"
    assert record["phrase_a"] == "the vase"
    assert record["phrase_b"] == "the lamp"


@pytest.mark.parametrize(
    "mutation",
    ("missing_field", "measurement_family", "slice_hash", "row_id"),
)
def test_qspatial_evidence_rejects_metadata_drift(mutation):
    parser_record = perception.parse_qspatial_gap_question(
        "What is the minimum distance between the coffee grinder and the base of the "
        "gooseneck kettle in the image?",
        row_id=0,
    )
    metadata = _qspatial_metadata(0)
    if mutation == "missing_field":
        metadata.pop("cluster_id")
    elif mutation == "measurement_family":
        metadata["measurement_family"] = "wrong"
    elif mutation == "slice_hash":
        metadata["slice_sha256"] = "0" * 64
    else:
        metadata["row_id"] = 1

    with pytest.raises(RuntimeError, match="manifest_mismatch"):
        perception._qspatial_evidence_base(b"question", b"image", parser_record, metadata)


def test_qspatial_scene_validator_rejects_invalid_evidence_before_scene_access():
    evidence = {
        "fallback_used": False,
        "geometry_valid": True,
        "geometry_status": "ok",
        "geometry": {"contract_sha256": perception.QSPATIAL_GEOMETRY_PROTOCOL_SHA256},
        "parser": {"contract_sha256": perception.QSPATIAL_PARSER_PROTOCOL_SHA256},
    }
    scene = SpatialScene()

    for mutation, message in (
        (lambda: evidence.__setitem__("fallback_used", True), "forbids fallback"),
        (lambda: evidence.__setitem__("geometry_valid", False), "requires valid geometry"),
        (lambda: evidence.__setitem__("geometry", None), "has no geometry evidence"),
        (
            lambda: evidence.__setitem__("geometry", {"contract_sha256": "wrong"}),
            "geometry contract drifted",
        ),
        (
            lambda: evidence.__setitem__("parser", {"contract_sha256": "wrong"}),
            "parser contract drifted",
        ),
    ):
        evidence.update(
            {
                "fallback_used": False,
                "geometry_valid": True,
                "geometry_status": "ok",
                "geometry": {"contract_sha256": perception.QSPATIAL_GEOMETRY_PROTOCOL_SHA256},
                "parser": {"contract_sha256": perception.QSPATIAL_PARSER_PROTOCOL_SHA256},
            }
        )
        mutation()
        with pytest.raises(RuntimeError, match=message):
            perception.validate_qspatial_gap_scene(scene, evidence)


def test_centroid_finiteness_rejects_non_iterable_values():
    assert perception._centroid_is_finite(object()) is False


@pytest.mark.parametrize(
    ("points", "confidence", "message"),
    [
        (np.zeros((4, 4), dtype=float), np.ones((4, 4), dtype=float), "point-map shape"),
        (np.zeros((4, 4, 3), dtype=float), np.ones((4, 3), dtype=float), "confidence-map shape"),
    ],
)
def test_point_maps_reject_misaligned_da3_payloads(points, confidence, message):
    class PointStore:
        def __init__(self):
            self.points = [points]
            self.confidence = [confidence]

        @staticmethod
        def get_by_frame_index(_frame_index):
            return 0

    reconstruction = types.SimpleNamespace(points=PointStore())
    with pytest.raises(RuntimeError, match=message):
        perception._point_maps_for_frame(reconstruction, 0)


def test_qspatial_async_entrypoint_requires_an_image():
    with pytest.raises(RuntimeError, match="no image"):
        asyncio.run(
            perception.build_qspatial_gap_scene(
                "What is the minimum distance between the bollards in the image?",
                [],
                Config(),
            )
        )


def test_qspatial_backend_builds_frozen_surface_gap_scene(png_bytes):
    points, confidence, left, right = _horizontal_gap_geometry()
    question = (
        "What is the minimum distance between the coffee grinder and the base of the "
        "gooseneck kettle in the image?"
    )
    parsed = perception.parse_qspatial_gap_question(question, row_id=0)
    backend = _backend(
        {
            "the coffee grinder": FakePerFrameMask(1, masks=left[np.newaxis, np.newaxis]),
            "the base of the gooseneck kettle": FakePerFrameMask(
                1, masks=right[np.newaxis, np.newaxis]
            ),
        }
    )
    backend._recon = FakeReconstructTool(points, confidence)

    scene, evidence = backend.ground_horizontal_surface_gap(
        png_bytes,
        question,
        parsed,
        _qspatial_metadata(0),
    )

    assert scene is not None
    perception.validate_qspatial_gap_scene(scene, evidence)
    assert backend._recon.calls == 1
    assert backend._sam3.calls == [
        ("the coffee grinder", {"confidence_threshold": 0.3}),
        ("the base of the gooseneck kettle", {"confidence_threshold": 0.3}),
    ]
    assert evidence["geometry_valid"] is True
    assert evidence["geometry"]["gap_m"] == pytest.approx(0.05)
    assert evidence["geometry"]["rendered_prediction"] == (
        r"\scalar{0.050000} \distance_unit{meter}"
    )
    assert "distance: 0.050000m" in scene.to_fact_sheet()


def test_qspatial_backend_fails_closed_without_scenegraph_fallback(png_bytes):
    points, confidence, left, _ = _horizontal_gap_geometry()
    question = "What is the minimum distance between the pillar and the left wall in the image?"
    parsed = perception.parse_qspatial_gap_question(question, row_id=46)
    backend = _backend(
        {
            "the pillar": FakePerFrameMask(1, masks=left[np.newaxis, np.newaxis]),
            "the left wall": FakePerFrameMask(1, masks=left[np.newaxis, np.newaxis]),
        }
    )
    backend._recon = FakeReconstructTool(points, confidence)

    scene, evidence = backend.ground_horizontal_surface_gap(
        png_bytes,
        question,
        parsed,
        _qspatial_metadata(46),
    )

    assert scene is None
    assert evidence["geometry_valid"] is False
    assert evidence["geometry"]["invalid_reason"] == "no_valid_candidate_pair"
    assert evidence["geometry"]["candidate_pairs"][0]["invalid_reason"] == ("mask_overlap_excess")
    assert evidence["fallback_used"] is False


def test_qspatial_repeated_instances_select_minimum_gap_then_indices(png_bytes):
    rows, columns = np.indices((12, 40))
    points = np.zeros((12, 40, 3), dtype=float)
    points[..., 0] = columns * 0.01
    points[..., 2] = rows * 0.01
    confidence = np.ones((12, 40), dtype=float)
    masks = np.zeros((1, 3, 12, 40), dtype=bool)
    masks[0, 0, 1:11, 1:11] = True
    masks[0, 1, 1:11, 13:23] = True
    masks[0, 2, 1:11, 30:40] = True
    question = "What is the minimum distance between the bollards in the image?"
    parsed = perception.parse_qspatial_gap_question(question, row_id=49)
    backend = _backend({"bollard": FakePerFrameMask(3, masks=masks)})
    backend._recon = FakeReconstructTool(points, confidence)

    scene, evidence = backend.ground_horizontal_surface_gap(
        png_bytes,
        question,
        parsed,
        _qspatial_metadata(49),
    )

    assert scene is not None
    assert len(evidence["geometry"]["candidate_pairs"]) == 3
    assert evidence["geometry"]["selected_mask_indices"] == [0, 1]
    assert evidence["geometry"]["gap_m"] == pytest.approx(0.05)
    assert backend._sam3.calls == [("bollard", {"confidence_threshold": 0.3})]


def test_qspatial_repeated_instance_uses_grounding_query_override(png_bytes):
    """SAM3 gets the shorter override phrase; scoring keeps the frozen group_phrase.

    Regression guard for the "sam_fewer_than_two" zero-detection failure on
    some repeated-instance rows: SAM3's grounder returned zero
    candidates for "front traffic barrel" but found instances for
    "traffic barrel" alone. The override must change only the SAM3 query,
    never the group_phrase used for entity labels/evidence identity.
    """
    points, confidence, left, right = _horizontal_gap_geometry()
    question = "What is the gap between the two traffic barrels at the front in the image?"
    parsed = perception.parse_qspatial_gap_question(question, row_id=65)
    assert parsed["group_phrase"] == "front traffic barrel"
    assert "front traffic barrel" in perception.QSPATIAL_GROUNDING_QUERY_OVERRIDES
    masks = np.stack([left, right])[np.newaxis]
    backend = _backend({"traffic barrel": FakePerFrameMask(2, masks=masks)})
    backend._recon = FakeReconstructTool(points, confidence)

    scene, evidence = backend.ground_horizontal_surface_gap(
        png_bytes,
        question,
        parsed,
        _qspatial_metadata(65),
    )

    assert scene is not None
    assert backend._sam3.calls == [("traffic barrel", {"confidence_threshold": 0.3})]
    assert evidence["segmentation"]["queries"] == ["traffic barrel"]
    assert scene.entities["qspatial_a"].label == "front traffic barrel"
    assert scene.entities["qspatial_b"].label == "front traffic barrel"


def test_qspatial_repeated_instance_without_override_uses_group_phrase_directly(png_bytes):
    """Rows with no override entry (e.g. "bollard") are unaffected."""
    assert "bollard" not in perception.QSPATIAL_GROUNDING_QUERY_OVERRIDES
    rows, columns = np.indices((12, 40))
    points = np.zeros((12, 40, 3), dtype=float)
    points[..., 0] = columns * 0.01
    points[..., 2] = rows * 0.01
    confidence = np.ones((12, 40), dtype=float)
    masks = np.zeros((1, 2, 12, 40), dtype=bool)
    masks[0, 0, 1:11, 1:11] = True
    masks[0, 1, 1:11, 13:23] = True
    question = "What is the minimum distance between the bollards in the image?"
    parsed = perception.parse_qspatial_gap_question(question, row_id=49)
    backend = _backend({"bollard": FakePerFrameMask(2, masks=masks)})
    backend._recon = FakeReconstructTool(points, confidence)

    scene, _evidence = backend.ground_horizontal_surface_gap(
        png_bytes,
        question,
        parsed,
        _qspatial_metadata(49),
    )

    assert scene is not None
    assert backend._sam3.calls == [("bollard", {"confidence_threshold": 0.3})]


def test_qspatial_repeated_instance_discards_spurious_small_fragment(png_bytes):
    """A tiny fragment mask near a real instance must not win "minimum gap".

    Regression guard: SAM3 can return a small third candidate next to a real
    instance. Unfiltered minimum-gap selection then picks the fragment pair
    over the real pair. The two real instances here are the same left/right
    pair as test_horizontal_surface_gap_follows_frozen_geometry_contract
    (100px each); the fragment is a 2x2px patch hugging the right instance,
    which is well under QSPATIAL_MIN_INSTANCE_AREA_RATIO (0.30) of 100px.
    """
    points, confidence, left, right = _horizontal_gap_geometry()
    fragment = np.zeros_like(left)
    fragment[1:3, 21:23] = True  # 2x2 = 4px, touches the right instance's edge
    assert fragment.sum() < perception.QSPATIAL_MIN_INSTANCE_AREA_RATIO * left.sum()
    masks = np.stack([left, right, fragment])[np.newaxis]
    question = "What is the minimum distance between the bollards in the image?"
    parsed = perception.parse_qspatial_gap_question(question, row_id=49)
    backend = _backend({"bollard": FakePerFrameMask(3, masks=masks)})
    backend._recon = FakeReconstructTool(points, confidence)

    scene, evidence = backend.ground_horizontal_surface_gap(
        png_bytes,
        question,
        parsed,
        _qspatial_metadata(49),
    )

    assert scene is not None
    assert evidence["segmentation"]["instance_areas"] == {
        0: int(left.sum()),
        1: int(right.sum()),
        2: int(fragment.sum()),
    }
    # Only one pair (0, 1) survives the plausibility filter -- index 2 (the
    # fragment) never enters candidate_pairs at all.
    assert len(evidence["geometry"]["candidate_pairs"]) == 1
    assert evidence["geometry"]["selected_mask_indices"] == [0, 1]
    assert evidence["geometry"]["gap_m"] == pytest.approx(0.05)


def test_qspatial_repeated_instance_fails_closed_when_fewer_than_two_plausible(png_bytes):
    """Two tiny fragments plus one real instance is still "fewer than two"."""
    points, confidence, left, _ = _horizontal_gap_geometry()
    fragment_a = np.zeros_like(left)
    fragment_a[1:3, 1:3] = True
    fragment_b = np.zeros_like(left)
    fragment_b[1:3, 21:23] = True
    masks = np.stack([left, fragment_a, fragment_b])[np.newaxis]
    question = "What is the minimum distance between the bollards in the image?"
    parsed = perception.parse_qspatial_gap_question(question, row_id=49)
    backend = _backend({"bollard": FakePerFrameMask(3, masks=masks)})
    backend._recon = FakeReconstructTool(points, confidence)

    scene, evidence = backend.ground_horizontal_surface_gap(
        png_bytes,
        question,
        parsed,
        _qspatial_metadata(49),
    )

    assert scene is None
    assert evidence["geometry_valid"] is False
    assert evidence["geometry"]["invalid_reason"] == "sam_fewer_than_two"


def test_ground_measures_metric_distance(png_bytes):
    points, confidence, left, right = _split_point_geometry()
    b = _backend(
        {
            "worker": FakePerFrameMask(1, masks=left[np.newaxis, np.newaxis]),
            "black forklift": FakePerFrameMask(1, masks=right[np.newaxis, np.newaxis]),
        }
    )
    b._recon = FakeReconstructTool(points, confidence)
    scene = b.ground(png_bytes, ["worker", "black forklift"])
    assert scene.entity_count == 2
    # ground-plane uses (x, z): (0,0) and (3,4) -> 5.0 m. The differing y heights
    # (1.7 vs 0.5) must NOT change the distance (regression guard vs guessed coords).
    assert scene.relations[0].distance == 5.0
    assert scene.entities["e0_i0"].attributes["source"] == "sam3+da3-height"
    assert scene.entities["e0_i0"].attributes["centroid_xyz_m"] == [0.0, 1.7, 0.0]


def test_empty_mask_is_skipped(png_bytes):
    b = _backend({"ghost": FakePerFrameMask(num_objects=0)})
    assert b.ground(png_bytes, ["ghost"]).entity_count == 0


def test_none_centroid_is_skipped(png_bytes):
    points = np.full((8, 8, 3), np.nan)
    b = _backend({"thing": FakePerFrameMask(1)})
    b._recon = FakeReconstructTool(points, np.ones((8, 8)))
    assert b.ground(png_bytes, ["thing"]).entity_count == 0


def test_height_uses_gravity_axis_frozen_percentiles_and_rejects_outlier(png_bytes):
    points = np.zeros((8, 8, 3), dtype=float)
    points[..., 0] = 42.0
    points[..., 1] = np.linspace(0.5, 2.0, 64).reshape(8, 8)
    points[..., 2] = -7.0
    points[0, 0, 1] = 1000.0
    confidence = np.ones((8, 8), dtype=float)
    backend = _backend({"cabinet": FakePerFrameMask(1)})
    backend._recon = FakeReconstructTool(points, confidence)

    scene = backend.ground(png_bytes, ["cabinet"])

    entity = scene.entities["e0_i0"]
    expected_low, expected_high = np.percentile(points[..., 1], [2.5, 97.5])
    assert entity.attributes["height_m"] == pytest.approx(
        round(float(expected_high - expected_low), 4)
    )
    assert entity.attributes["height_percentile_low_m"] == pytest.approx(
        round(float(expected_low), 4)
    )
    assert entity.attributes["height_percentile_high_m"] == pytest.approx(
        round(float(expected_high), 4)
    )
    assert entity.attributes["height_percentiles"] == [2.5, 97.5]
    assert entity.attributes["gravity_axis"] == "y"
    assert entity.attributes["height_m"] < 100.0


def test_height_filters_confidence_and_records_frozen_threshold(png_bytes):
    points = np.zeros((8, 8, 3), dtype=float)
    points[..., 1] = np.concatenate(
        [np.linspace(1.0, 2.0, 32), np.linspace(100.0, 200.0, 32)]
    ).reshape(8, 8)
    confidence = np.concatenate([np.ones(32), np.full(32, 0.3)]).reshape(8, 8)
    backend = _backend({"chair": FakePerFrameMask(1)})
    backend._recon = FakeReconstructTool(points, confidence)

    entity = backend.ground(png_bytes, ["chair"]).entities["e0_i0"]

    expected_low, expected_high = np.percentile(points.reshape(-1, 3)[:32, 1], [2.5, 97.5])
    assert entity.attributes["finite_point_count"] == 64
    assert entity.attributes["confident_point_count"] == 32
    assert entity.attributes["confidence_threshold"] == 0.3
    assert entity.attributes["minimum_confident_points"] == 32
    assert entity.attributes["height_m"] == pytest.approx(
        round(float(expected_high - expected_low), 4)
    )


def test_ground_emits_every_sam3_instance_with_stable_ids(png_bytes):
    points, confidence, left, right = _split_point_geometry()
    masks = np.stack([left, right], axis=0)[np.newaxis]
    segmentation = FakePerFrameMask(
        2,
        masks=masks,
        labels=["crate_0", "crate_1"],
        object_ids=[11, 19],
    )
    backend = _backend({"crate": segmentation})
    backend._recon = FakeReconstructTool(points, confidence)

    first = backend.ground(png_bytes, ["crate"])
    second = backend.ground(png_bytes, ["crate"])

    assert list(first.entities) == ["e0_i0", "e0_i1"]
    assert list(second.entities) == ["e0_i0", "e0_i1"]
    assert [entity.attributes["mask_instance_index"] for entity in first.entities.values()] == [
        0,
        1,
    ]
    assert [entity.attributes["mask_object_id"] for entity in first.entities.values()] == [
        11,
        19,
    ]
    assert [entity.attributes["mask_label"] for entity in first.entities.values()] == [
        "crate_0",
        "crate_1",
    ]
    assert all(entity.attributes["query_phrase"] == "crate" for entity in first.entities.values())


@pytest.mark.parametrize("failure", ["empty", "nonfinite", "undersized"])
def test_generic_height_strict_rejects_degenerate_instance(png_bytes, failure):
    points = np.zeros((8, 8, 3), dtype=float)
    confidence = np.ones((8, 8), dtype=float)
    mask = np.ones((8, 8), dtype=bool)
    if failure == "empty":
        mask[:] = False
    elif failure == "nonfinite":
        points[:] = np.nan
    else:
        mask[:] = False
        mask.reshape(-1)[:31] = True
    segmentation = FakePerFrameMask(1, masks=mask[np.newaxis, np.newaxis])
    config = Config()
    config.fieldwork_metric_strict = True
    backend = MetricPerceptionBackend(config)
    backend._sam3 = FakeSAM3({"object": segmentation})
    backend._recon = FakeReconstructTool(points, confidence)

    with pytest.raises(RuntimeError, match=failure):
        backend.ground(png_bytes, ["object"])


def test_height_attributes_are_visible_in_fact_sheet(png_bytes):
    backend = _backend({"cabinet": FakePerFrameMask(1)})

    sheet = backend.ground(png_bytes, ["cabinet"]).to_fact_sheet()

    assert "height_m=" in sheet
    assert "height_percentiles=[2.5, 97.5]" in sheet
    assert "source=sam3+da3-height" in sheet
    assert "measurement_provenance=sam3+da3-height" in sheet
    assert "centroid_xyz_m=" in sheet


def test_explicit_height_validator_accepts_one_valid_entity(png_bytes):
    scene = _backend({"cabinet": FakePerFrameMask(1)}).ground(png_bytes, ["cabinet"])

    validate_metric_height_scene(scene)
    with pytest.raises(RuntimeError, match="at least two finite grounded entities"):
        validate_metric_scene(scene)


def test_explicit_height_validator_rejects_empty_scene():
    with pytest.raises(RuntimeError, match="at least one entity"):
        validate_metric_height_scene(SpatialScene())


@pytest.mark.parametrize(
    ("attribute", "value", "error"),
    [
        ("source", "sam3+da3", "provenance"),
        ("height_m", np.nan, "nonfinite"),
        ("height_m", 0.0, "degenerate"),
        ("confident_point_count", 31, "undersized"),
        ("confidence_threshold", 0.2, "frozen protocol"),
        ("centroid_xyz_m", [0.0, np.inf, 0.0], "nonfinite centroid"),
    ],
)
def test_explicit_height_validator_rejects_invalid_geometry(
    png_bytes,
    attribute,
    value,
    error,
):
    scene = _backend({"cabinet": FakePerFrameMask(1)}).ground(png_bytes, ["cabinet"])
    scene.entities["e0_i0"].attributes[attribute] = value

    with pytest.raises(RuntimeError, match=error):
        validate_metric_height_scene(scene)


@pytest.mark.parametrize(
    ("counts", "decoded", "expected_shape"),
    [
        ("compressed", np.array([[[0], [1]], [[1], [0]]]), (2, 2)),
        (b"compressed", np.array([[0, 1], [1, 0]]), (2, 2)),
    ],
)
def test_decode_coco_rle_is_lazy_and_normalizes_masks(monkeypatch, counts, decoded, expected_shape):
    calls = []
    mask_module = types.ModuleType("pycocotools.mask")
    mask_module.decode = lambda payload: calls.append(payload) or decoded
    package = types.ModuleType("pycocotools")
    package.mask = mask_module
    monkeypatch.setitem(sys.modules, "pycocotools", package)
    monkeypatch.setitem(sys.modules, "pycocotools.mask", mask_module)

    mask = perception._decode_coco_rle({"size": [2, 2], "counts": counts})

    assert mask.shape == expected_shape
    assert mask.dtype == bool
    assert calls[0]["counts"] == b"compressed"


def test_per_frame_mask_dependency_is_loaded_lazily(monkeypatch):
    expected = object()
    module = types.ModuleType("spatial_agent.kernel_types.per_frame_types")
    module.PerFrameMask = expected
    monkeypatch.setitem(sys.modules, "spatial_agent", types.ModuleType("spatial_agent"))
    monkeypatch.setitem(
        sys.modules,
        "spatial_agent.kernel_types",
        types.ModuleType("spatial_agent.kernel_types"),
    )
    monkeypatch.setitem(
        sys.modules,
        "spatial_agent.kernel_types.per_frame_types",
        module,
    )

    assert perception._per_frame_mask_class() is expected


def test_resize_mask_is_noop_at_reconstruction_resolution():
    mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    resized = perception._resize_mask_nearest(mask, (2, 2))

    assert resized.dtype == bool
    assert np.array_equal(resized, mask.astype(bool))


def test_pairwise_distance_skips_missing_and_nonfinite_geometry():
    scene = SpatialScene()
    scene.add_entity(SpatialEntity("finite", "finite", (0.0, 0.0)))
    scene.add_entity(SpatialEntity("missing", "missing", None))
    scene.add_entity(SpatialEntity("infinite", "infinite", (np.inf, 0.0)))

    perception._add_pairwise_distances(scene)

    assert scene.relations == []


def test_validate_metric_scene_requires_finite_distance_relation():
    scene = SpatialScene()
    scene.add_entity(SpatialEntity("a", "a", (0.0, 0.0)))
    scene.add_entity(SpatialEntity("b", "b", (1.0, 1.0)))

    with pytest.raises(RuntimeError, match="finite distance_to relation"):
        validate_metric_scene(scene)


class _RegionPoints:
    def __init__(self):
        rows, columns = np.indices((4, 4))
        point_map = np.stack(
            [columns.astype(float), np.ones((4, 4)), rows.astype(float)],
            axis=-1,
        )
        self.points = point_map[np.newaxis]
        self.confidence = np.ones((1, 4, 4), dtype=float)

    def __getitem__(self, frame_index):
        assert frame_index == 7
        return self.points[0]

    def get_by_frame_index(self, frame_index):
        assert frame_index == 7
        return 0


class _RegionRecon:
    frame_indices: ClassVar[list[int]] = [7]

    def __init__(self):
        self.points = _RegionPoints()


class _RegionReconTool:
    is_available = True

    def Reconstruct(self, _frames):
        return _RegionRecon()


class _RecordingMask:
    instances: ClassVar[list] = []

    def __init__(self, masks, labels, object_ids, frame_indices, frames):
        self.masks = masks
        self.labels = labels
        self.object_ids = object_ids
        self.frame_indices = frame_indices
        self.frames = frames
        self.__class__.instances.append(self)

    def get_centroid_3d(self, recon, frame, object):
        assert object == 0
        points = recon.points[frame][self.masks[0, 0]]
        return np.median(points, axis=0)


def _region_backend(strict=True):
    config = Config()
    config.fieldwork_metric_strict = strict
    backend = MetricPerceptionBackend(config)
    backend._sam3 = FakeSAM3({})
    backend._recon = _RegionReconTool()
    return backend


def test_exact_regions_decode_resize_and_preserve_identity(monkeypatch, png_bytes):
    masks = {
        "first": np.array([[1, 0], [0, 0]], dtype=bool),
        "second": np.array([[0, 0], [0, 1]], dtype=bool),
    }
    _RecordingMask.instances = []
    monkeypatch.setattr(perception, "_decode_coco_rle", lambda rle: masks[rle["id"]])
    monkeypatch.setattr(perception, "_per_frame_mask_class", lambda: _RecordingMask)

    scene = _region_backend().ground_regions(
        png_bytes,
        [{"id": "first"}, {"id": "second"}],
    )

    assert [entity.label for entity in scene.entities.values()] == ["Region 0", "Region 1"]
    assert scene.entities["region_0"].position == (0.5, 0.5)
    assert scene.entities["region_1"].position == (2.5, 2.5)
    assert scene.entities["region_0"].attributes["source"] == "rle+da3"
    assert "height_m" not in scene.entities["region_0"].attributes
    assert "centroid_xyz_m" not in scene.entities["region_0"].attributes
    assert scene.relations[0].distance == 2.83
    assert [instance.masks.shape for instance in _RecordingMask.instances] == [
        (1, 1, 4, 4),
        (1, 1, 4, 4),
    ]
    validate_metric_scene(scene)


def test_exact_region_strict_rejects_empty_mask(monkeypatch, png_bytes):
    monkeypatch.setattr(
        perception,
        "_decode_coco_rle",
        lambda _rle: np.zeros((4, 4), dtype=bool),
    )
    monkeypatch.setattr(perception, "_per_frame_mask_class", lambda: _RecordingMask)

    with pytest.raises(RuntimeError, match="Region 0 decoded to an empty mask"):
        _region_backend().ground_regions(png_bytes, [{"id": "empty"}])


def test_exact_region_strict_rejects_nonfinite_centroid(monkeypatch, png_bytes):
    class NonfiniteMask(_RecordingMask):
        def get_centroid_3d(self, recon, frame, object):
            return np.array([np.nan, 1.0, 2.0])

    monkeypatch.setattr(
        perception,
        "_decode_coco_rle",
        lambda _rle: np.ones((4, 4), dtype=bool),
    )
    monkeypatch.setattr(perception, "_per_frame_mask_class", lambda: NonfiniteMask)

    with pytest.raises(RuntimeError, match="Region 0 produced a nonfinite 3D centroid"):
        _region_backend().ground_regions(png_bytes, [{"id": "bad"}])


@pytest.mark.parametrize("failure", ["empty", "nonfinite"])
def test_exact_region_production_mode_skips_invalid_region(monkeypatch, png_bytes, failure):
    class ProductionMask(_RecordingMask):
        def get_centroid_3d(self, recon, frame, object):
            if failure == "nonfinite":
                return np.array([np.nan, 1.0, 2.0])
            return super().get_centroid_3d(recon, frame, object)

    decoded = np.zeros((4, 4), dtype=bool) if failure == "empty" else np.ones((4, 4), dtype=bool)
    monkeypatch.setattr(perception, "_decode_coco_rle", lambda _rle: decoded)
    monkeypatch.setattr(perception, "_per_frame_mask_class", lambda: ProductionMask)

    scene = _region_backend(strict=False).ground_regions(png_bytes, [{"id": failure}])

    assert scene.entity_count == 0


def test_exact_region_rejects_invalid_point_map(monkeypatch, png_bytes):
    class BadPoints:
        def __getitem__(self, _frame_index):
            return np.zeros((4, 4), dtype=float)

    class BadReconTool:
        is_available = True

        def Reconstruct(self, _frames):
            return type("BadRecon", (), {"frame_indices": [0], "points": BadPoints()})()

    backend = MetricPerceptionBackend(Config())
    backend._sam3 = FakeSAM3({})
    backend._recon = BadReconTool()

    with pytest.raises(RuntimeError, match="point map must have shape"):
        backend.ground_regions(png_bytes, [{"id": "region"}])


@pytest.mark.asyncio
async def test_extract_entities_parses_json():
    b = MetricPerceptionBackend(Config())
    llm = AsyncMock()
    llm.generate.return_value = '{"entities": ["worker", "forklift"]}'
    assert await b.extract_entities(llm, "q?") == ["worker", "forklift"]


@pytest.mark.asyncio
async def test_extract_entities_prompt_captures_height_without_requesting_answer():
    backend = MetricPerceptionBackend(Config())
    llm = AsyncMock()
    llm.generate.return_value = '{"entities": ["wooden cabinet"]}'

    await backend.extract_entities(llm, "What is the height of the wooden cabinet?")

    prompt = llm.generate.await_args.args[0]
    assert "height" in prompt.lower()
    assert "size" in prompt.lower()
    assert "do not answer" in prompt.lower()
    assert "do not estimate" in prompt.lower()


@pytest.mark.asyncio
async def test_extract_entities_caps_at_six():
    b = MetricPerceptionBackend(Config())
    llm = AsyncMock()
    llm.generate.return_value = '{"entities": ["a","b","c","d","e","f","g","h"]}'
    assert len(await b.extract_entities(llm, "q?")) == 6


@pytest.mark.asyncio
async def test_extract_entities_tolerates_nonjson():
    b = MetricPerceptionBackend(Config())
    llm = AsyncMock()
    llm.generate.return_value = "not json at all"
    assert await b.extract_entities(llm, "q?") == []


@pytest.mark.asyncio
async def test_build_metric_scene_requires_entities(png_bytes):
    llm = AsyncMock()
    llm.generate.return_value = '{"entities": []}'
    with pytest.raises(RuntimeError):
        await build_metric_scene("q?", [png_bytes], llm, Config())


@pytest.mark.asyncio
async def test_build_metric_scene_requires_image():
    llm = AsyncMock()
    llm.generate.return_value = '{"entities": ["worker"]}'
    with pytest.raises(RuntimeError):
        await build_metric_scene("q?", [], llm, Config())


@pytest.mark.asyncio
async def test_build_metric_scene_uses_exact_regions_without_entity_extraction(
    monkeypatch, png_bytes
):
    scene = SpatialScene()
    ground_regions = MagicMock(return_value=scene)
    monkeypatch.setattr(MetricPerceptionBackend, "ground_regions", ground_regions)
    llm = AsyncMock()
    regions = [{"size": [1, 1], "counts": "x"}]

    result = await build_metric_scene("q?", [png_bytes], llm, Config(), regions=regions)

    assert result is scene
    ground_regions.assert_called_once_with(png_bytes, regions)
    llm.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_build_metric_scene_rejects_empty_exact_regions(png_bytes):
    with pytest.raises(RuntimeError, match="no exact regions"):
        await build_metric_scene("q?", [png_bytes], AsyncMock(), Config(), regions=[])


@pytest.mark.asyncio
async def test_exact_region_grounding_overlaps_without_blocking_event_loop(monkeypatch, png_bytes):
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = 0
    max_active = 0

    def ground_regions(_self, _image_bytes, _regions):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            barrier.wait(timeout=2)
            return SpatialScene()
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(MetricPerceptionBackend, "ground_regions", ground_regions)
    regions = [{"size": [1, 1], "counts": "x"}]
    results = await asyncio.wait_for(
        asyncio.gather(
            build_metric_scene("q1", [png_bytes], AsyncMock(), Config(), regions=regions),
            build_metric_scene("q2", [png_bytes], AsyncMock(), Config(), regions=regions),
        ),
        timeout=3,
    )

    assert len(results) == 2
    assert max_active == 2


@pytest.mark.asyncio
async def test_build_metric_scene_generic_success_uses_sam3_path(monkeypatch, png_bytes):
    scene = SpatialScene()
    ground = MagicMock(return_value=scene)
    monkeypatch.setattr(MetricPerceptionBackend, "ground", ground)
    llm = AsyncMock()
    llm.generate.return_value = '{"entities": ["worker", "forklift"]}'

    result = await build_metric_scene("q?", [png_bytes], llm, Config())

    assert result is scene
    ground.assert_called_once_with(png_bytes, ["worker", "forklift"])


def _inject_spatial_agent(monkeypatch, sam3_available=True, recon_available=True):
    """Inject fake spatial_agent.tools modules so _ensure_tools can import them."""
    import sys
    import types

    sam3_mod = types.ModuleType("spatial_agent.tools.sam3_tool")
    recon_mod = types.ModuleType("spatial_agent.tools.reconstruct_tool")

    class _SAM3Tool:
        is_available = sam3_available

    class _ReconstructTool:
        def __init__(self, sam3, cfg):
            self.is_available = recon_available

    sam3_mod.SAM3Tool = _SAM3Tool
    recon_mod.ReconstructTool = _ReconstructTool
    for name, mod in [
        ("spatial_agent", types.ModuleType("spatial_agent")),
        ("spatial_agent.tools", types.ModuleType("spatial_agent.tools")),
        ("spatial_agent.tools.sam3_tool", sam3_mod),
        ("spatial_agent.tools.reconstruct_tool", recon_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)


def test_ensure_tools_constructs_when_available(monkeypatch):
    _inject_spatial_agent(monkeypatch)
    b = MetricPerceptionBackend(Config())
    b._ensure_tools()
    assert b._sam3 is not None and b._recon is not None


def test_ensure_tools_raises_when_server_unavailable(monkeypatch):
    _inject_spatial_agent(monkeypatch, sam3_available=False)
    b = MetricPerceptionBackend(Config())
    with pytest.raises(RuntimeError, match="GPU tool server"):
        b._ensure_tools()


def test_ensure_tools_raises_when_reconstruction_is_unavailable(monkeypatch):
    _inject_spatial_agent(monkeypatch, recon_available=False)
    b = MetricPerceptionBackend(Config())
    with pytest.raises(RuntimeError, match="GPU tool server"):
        b._ensure_tools()


def _valid_qspatial_scene_and_evidence(png_bytes):
    """Build a genuinely valid frozen surface-gap scene to perturb one field at a time."""
    points, confidence, left, right = _horizontal_gap_geometry()
    question = (
        "What is the minimum distance between the coffee grinder and the base of the "
        "gooseneck kettle in the image?"
    )
    parsed = perception.parse_qspatial_gap_question(question, row_id=0)
    backend = _backend(
        {
            "the coffee grinder": FakePerFrameMask(1, masks=left[np.newaxis, np.newaxis]),
            "the base of the gooseneck kettle": FakePerFrameMask(
                1, masks=right[np.newaxis, np.newaxis]
            ),
        }
    )
    backend._recon = FakeReconstructTool(points, confidence)
    scene, evidence = backend.ground_horizontal_surface_gap(
        png_bytes, question, parsed, _qspatial_metadata(0)
    )
    assert scene is not None
    perception.validate_qspatial_gap_scene(scene, evidence)
    return scene, evidence


def _drift_entity_provenance(scene, _evidence):
    entity = next(iter(scene.entities.values()))
    entity.attributes["source"] = "llm-guess"


def _make_gap_nonfinite(scene, _evidence):
    _horizontal_relation(scene).distance = float("nan")


def _push_gap_out_of_range(scene, _evidence):
    _horizontal_relation(scene).distance = perception.QSPATIAL_MAX_GAP_M + 1.0


def _disagree_with_evidence(scene, evidence):
    evidence["geometry"]["gap_m"] = _horizontal_relation(scene).distance + 0.5


def _drop_directed_evidence(_scene, evidence):
    evidence["geometry"]["directed_q05_m"] = None


def _corrupt_voxel_evidence(_scene, evidence):
    evidence["geometry"]["voxel_keys_sha256"] = ["not-a-sha", "also-not-a-sha"]


def _break_symmetric_estimator(scene, evidence):
    relation = _horizontal_relation(scene)
    skewed = min(float(value) for value in evidence["geometry"]["directed_q05_m"]) + 0.25
    relation.distance = skewed
    evidence["geometry"]["gap_m"] = skewed


def _horizontal_relation(scene):
    return next(
        relation for relation in scene.relations if relation.predicate == "horizontal_surface_gap"
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_drift_entity_provenance, "entity provenance drifted"),
        (_make_gap_nonfinite, "gap is nonfinite"),
        (_push_gap_out_of_range, "gap is outside frozen range"),
        (_disagree_with_evidence, "relation and evidence disagree"),
        (_drop_directed_evidence, "lacks directed gap evidence"),
        (_corrupt_voxel_evidence, "lacks deterministic voxel evidence"),
        (_break_symmetric_estimator, "violates symmetric gap estimator"),
    ],
)
def test_strict_validator_rejects_each_corrupted_geometry_claim(png_bytes, mutate, message):
    """Each fail-closed guard must reject a scene that is valid except for one field.

    These guards are the last line between a corrupted measurement and a sealed
    benchmark gate, so every rejection branch carries its own regression.
    """
    scene, evidence = _valid_qspatial_scene_and_evidence(png_bytes)
    mutate(scene, evidence)
    with pytest.raises(RuntimeError, match=message):
        perception.validate_qspatial_gap_scene(scene, evidence)
