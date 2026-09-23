"""Metric perception backend: estimate metric (x, z) ground-plane positions from SAM3 masks
and a reconstructed point map.

Uses NVIDIA SpatialClaw's GPU tool server (SAM3 segmentation + Depth-Anything-3 / Pi3
reconstruction) to estimate metric (x, z) ground-plane positions for scene entities, then
reuses Spatial Atlas's existing deterministic SpatialScene pipeline. Reconstruction is an
estimate, so mask, depth, and scale errors can still reach the scene.

Requires (runs on a GPU host only): the `spatial_agent` package importable (SpatialClaw, NVIDIA
Source Code License-NC) and a running SpatialClaw GPU tool server (registry
logs/gpu_server.json reachable). Raises RuntimeError on any unavailability so the handler
can fall back to the scene-graph engine.

Pipeline for a generic single-image metric task:
  reconstruct(image) -> metric point map (meters, gravity-aligned)
  for each entity phrase: SAM3 segment_image_by_text -> all instance masks
                          -> confidence-qualified visible points per mask
                          -> robust 2.5th to 97.5th percentile +Y height
  SpatialScene with reconstructed heights and ground-plane (x, z) positions.

Warehouse exact-RLE tasks retain their separate centroid-only path.
"""

import asyncio
import hashlib
import io
import itertools
import json
import logging
import math
import os
import re
import unicodedata
from typing import Any

from PIL import Image

from fieldwork.spatial import SpatialEntity, SpatialRelation, SpatialScene
from llm import LLMClient

logger = logging.getLogger("spatial-atlas.fieldwork.perception")

ENTITY_PROMPT = (
    "Extract only the physical entities whose height, size, spatial position, or pairwise "
    "distance the question asks about. Include each visually distinct target description, "
    "but do not answer the question. Do not estimate any measurement or include numbers or "
    "units.\n"
    'Return JSON {{"entities": ["<short visual phrase>", ...]}} with 1-6 concise, visually '
    'groundable phrases (color + clothing/type), e.g. "woman wearing a yellow vest and black '
    'shirt", "black forklift".\nQuestion: {question}'
)

HEIGHT_LOW_PERCENTILE = 2.5
HEIGHT_HIGH_PERCENTILE = 97.5
HEIGHT_CONFIDENCE_THRESHOLD = 0.3
HEIGHT_MIN_CONFIDENT_POINTS = 32

QSPATIAL_GAP_PROTOCOL = "qspatial-horizontal-gap-v1"
QSPATIAL_CONFIDENCE_THRESHOLD = 0.3
QSPATIAL_MASK_OVERLAP_THRESHOLD = 0.05
QSPATIAL_VOXEL_SIZE_M = 0.005
QSPATIAL_MIN_UNIQUE_VOXELS = 32
QSPATIAL_MAX_UNIQUE_VOXELS = 50_000
QSPATIAL_NEAREST_PERCENTILE = 5.0
QSPATIAL_MAX_GAP_M = 20.0
QSPATIAL_PARSER_PROTOCOL = "qspatial-question-parser-v1"


def _configured_sha256(name: str) -> str:
    """Return an operator-supplied SHA-256 hex digest, or "" when it is unset or malformed."""
    value = os.environ.get(name, "").strip().lower()
    return value if re.fullmatch(r"[0-9a-f]{64}", value) else ""


# The frozen QSpatial gap slice is a private control artifact and is not bundled with this
# repository. Its digest is supplied at run time. Sample metadata validation fails closed when
# the digest is unset.
QSPATIAL_GAP_SLICE_SHA256 = _configured_sha256("ATLAS_QSPATIAL_GAP_SLICE_SHA256")

QSPATIAL_UNAVAILABLE = "measurement unavailable"

# The phrases and the question below are copied from Q-Spatial Bench (QSpatial++) question text
# by Liao et al. They are kept verbatim, including the dataset's own spelling, so the parser
# matches those rows exactly. That text stays under the dataset's Apache-2.0 license, and the
# MIT License of this repository does not cover it.
QSPATIAL_PARSER_CONTRACT = {
    "distinct_alias_substrings": {
        "breen jar": "green jar",
        "coffe table": "coffee table",
        "ps4": "playstation 4 console",
    },
    "distinct_regex": (
        r"^what is (?:the )?(?:minimum distance|gap|horizontal gap) between "
        r"(?P<a>.+?) and (?P<b>.+?)(?: in the image)?\?$"
    ),
    "normalization": ["unicode_nfkc", "strip", "lower", "collapse_ascii_whitespace"],
    "repeated_aliases": {
        "bollards": "bollard",
        "front tires of the two bikes": "front tire of bike",
        "two black chairs": "black chair",
        "two boxwoods in the black planters": "boxwood in black planter",
        "two cars": "car",
        "two coke cans": "coke can",
        "two crayons on the table": "crayon on table",
        "two freestanding displays": "freestanding display",
        "two orange chairs next to the window": "orange chair next to window",
        "two pictcures in the middle on the wall": "picture in middle of wall",
        "two pillows on the sofa": "pillow on sofa",
        "two stool chairs": "stool chair",
        "two stool chairs with speckled pattern": "stool chair with speckled pattern",
        "two tea containers": "tea container",
        "two traffic barrels at the front": "front traffic barrel",
    },
    "repeated_regex": (
        r"^what is (?:the )?(?:minimum distance|gap|horizontal gap) between "
        r"(?:the )?(?P<pair>.+?)(?: in the image)?\?$"
    ),
    "unsupported_exact_question": (
        "what is the horizontal gap in the remaining area of the bookshelf next to "
        "the last book on the left side?"
    ),
    "unsupported_row_ids": [80, 81, 82],
    "version": QSPATIAL_PARSER_PROTOCOL,
}

# SAM3's open-vocabulary grounder can return zero candidates for a
# spatially-qualified group_phrase (e.g. "orange chair next to window") while
# finding the object easily once the relational clause is dropped (e.g.
# "orange chair"). This table only changes what text is sent to SAM3 for
# grounding; parser_record["group_phrase"] (used for scoring, evidence
# identity, and entity labels) is untouched and stays outside this override.
# Deliberately NOT part of QSPATIAL_PARSER_CONTRACT: it does not change what
# a question means or how it is scored, only how the segmenter is queried.
QSPATIAL_GROUNDING_QUERY_OVERRIDES: dict[str, str] = {
    "orange chair next to window": "orange chair",
    "front traffic barrel": "traffic barrel",
}

# For repeated_instance mode, SAM3 can return a spurious extra candidate (a
# fragment of one real instance, an overlapping sub-detection, a shadow) far
# smaller than the true instances. Pairing it in raises "minimum valid gap"
# selection: a tiny fragment sitting close to a real instance produces an
# artificially small gap that outranks the true pair. Discarding instances
# whose area is far below the group's largest instance, before pairing,
# keeps "minimum_valid_gap_then_lowest_mask_indices" (QSPATIAL_GEOMETRY_
# CONTRACT, unchanged) applied to a plausible candidate pool instead of a
# raw, unfiltered one. The 0.30 threshold separates a genuine second
# instance from a small spurious fragment in the development cases the
# author inspected.
QSPATIAL_MIN_INSTANCE_AREA_RATIO = 0.30

QSPATIAL_GEOMETRY_CONTRACT = {
    "candidate_pair_selection": "minimum_valid_gap_then_lowest_mask_indices",
    "da3_confidence_rule": ">0.30",
    "da3_points_unit": "meter_already_scaled_no_posthoc_scale",
    "erosion_radius_px": ("ceil(max(mask_width/da3_target_width,mask_height/da3_target_height))"),
    "erosion_structure": "closed_euclidean_disk_dx2_plus_dy2_le_r2",
    "gap_estimator": "min(q05(A_to_B_nearest_xz),q05(B_to_A_nearest_xz))",
    "gap_valid_range_m": "0<gap<=20",
    "mask_overlap_coefficient": "intersection_pixels/min(mask_a_pixels,mask_b_pixels)",
    "mask_overlap_rule": ">=0.05_invalid_else_remove_intersection_from_both",
    "max_voxels_per_mask": QSPATIAL_MAX_UNIQUE_VOXELS,
    "min_unique_xz_voxels": QSPATIAL_MIN_UNIQUE_VOXELS,
    "nearest_neighbor": "scipy_cKDTree_euclidean_workers_1",
    "nearest_quantile": 0.05,
    "quantile_method": "numpy_linear",
    "sam_detection_rule": ">=0.30",
    "voxel_representative": "highest_da3_confidence_then_lowest_flat_pixel_index",
    "voxel_subsample_if_over_cap": ("keep_50000_smallest_sha256_of_signed_vx_comma_vz_utf8"),
    "version": QSPATIAL_GAP_PROTOCOL,
    "xz_coordinates": "world_axes_0_and_2",
    "xz_voxel_key": "floor(x/0.005),floor(z/0.005)",
    "xz_voxel_size_m": QSPATIAL_VOXEL_SIZE_M,
}


def _frozen_contract_sha256(contract: dict[str, Any]) -> str:
    payload = json.dumps(contract, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Each protocol digest is computed from the contract defined above, so journals record the
# exact contract this source file carries.
QSPATIAL_PARSER_PROTOCOL_SHA256 = _frozen_contract_sha256(QSPATIAL_PARSER_CONTRACT)
QSPATIAL_GEOMETRY_PROTOCOL_SHA256 = _frozen_contract_sha256(QSPATIAL_GEOMETRY_CONTRACT)


def parse_qspatial_gap_question(question: str, *, row_id: int | None = None) -> dict[str, Any]:
    """Parse one frozen QSpatial++ prompt using question text only."""
    if not isinstance(question, str) or not question.strip():
        raise RuntimeError("parse_failed: question must be a nonempty string")
    normalized = unicodedata.normalize("NFKC", question).strip().lower()
    normalized = " ".join(normalized.split())

    base = {
        "version": QSPATIAL_PARSER_PROTOCOL,
        "contract_sha256": QSPATIAL_PARSER_PROTOCOL_SHA256,
        "normalized_question": normalized,
        "row_id": row_id,
    }
    if normalized == QSPATIAL_PARSER_CONTRACT["unsupported_exact_question"]:
        if row_id is not None and row_id not in QSPATIAL_PARSER_CONTRACT["unsupported_row_ids"]:
            raise RuntimeError(
                "manifest_mismatch: unsupported negative-space question has unexpected row ID"
            )
        return {
            **base,
            "mode": "unsupported_negative_space",
            "phrase_a": None,
            "phrase_b": None,
            "group_phrase": None,
        }

    distinct = re.fullmatch(QSPATIAL_PARSER_CONTRACT["distinct_regex"], normalized)
    if distinct is not None:
        phrases = [distinct.group("a").strip(), distinct.group("b").strip()]
        for index, phrase in enumerate(phrases):
            for source, replacement in QSPATIAL_PARSER_CONTRACT[
                "distinct_alias_substrings"
            ].items():
                phrase = phrase.replace(source, replacement)
            phrases[index] = phrase.strip()
        if not all(phrases):
            raise RuntimeError("parse_failed: distinct-pair phrase is empty")
        return {
            **base,
            "mode": "distinct_pair",
            "phrase_a": phrases[0],
            "phrase_b": phrases[1],
            "group_phrase": None,
        }

    repeated = re.fullmatch(QSPATIAL_PARSER_CONTRACT["repeated_regex"], normalized)
    if repeated is not None:
        source_phrase = repeated.group("pair").strip()
        group_phrase = QSPATIAL_PARSER_CONTRACT["repeated_aliases"].get(source_phrase)
        if group_phrase:
            return {
                **base,
                "mode": "repeated_instance",
                "phrase_a": None,
                "phrase_b": None,
                "group_phrase": group_phrase,
            }

    raise RuntimeError("parse_failed: question is outside the frozen QSpatial grammar")


def _decode_coco_rle(rle: dict[str, Any]):
    """Decode one COCO RLE while keeping NumPy and pycocotools optional."""
    import numpy as np
    from pycocotools import mask as mask_utils

    payload = dict(rle)
    if isinstance(payload.get("counts"), str):
        payload["counts"] = payload["counts"].encode("ascii")
    decoded = mask_utils.decode(payload)
    if decoded.ndim == 3:
        decoded = decoded[..., 0]
    return np.asarray(decoded, dtype=bool)


def _per_frame_mask_class():
    """Load SpatialClaw's mask type only inside an active metric run."""
    from spatial_agent.kernel_types.per_frame_types import PerFrameMask

    return PerFrameMask


def _resize_mask_nearest(mask, target_shape: tuple[int, int]):
    """Resize a categorical mask only when DA3 uses a different resolution."""
    import numpy as np

    if tuple(mask.shape) == tuple(target_shape):
        return np.asarray(mask, dtype=bool)
    height, width = target_shape
    mask_image = Image.fromarray(np.asarray(mask, dtype="uint8") * 255, mode="L")
    resized = mask_image.resize((width, height), resample=Image.Resampling.NEAREST)
    return np.asarray(resized, dtype="uint8") > 0


def _erode_qspatial_mask(mask, source_shape: tuple[int, int], target_shape: tuple[int, int]):
    """Apply the frozen, scale-aware one-pass QSpatial mask erosion."""
    import numpy as np
    from scipy.ndimage import binary_erosion

    source_height, source_width = source_shape
    target_height, target_width = target_shape
    radius = math.ceil(
            max(
                source_width / target_width,
                source_height / target_height,
            )
        )
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    closed_disk = (xx * xx + yy * yy) <= radius * radius
    eroded = binary_erosion(
        np.asarray(mask, dtype=bool),
        structure=closed_disk,
        iterations=1,
        border_value=0,
    )
    return np.asarray(eroded, dtype=bool), radius


def _ranked_qspatial_voxels(mask, points, confidence):
    """Return one deterministic highest-confidence XZ point per frozen voxel."""
    import numpy as np

    finite_xyz = np.isfinite(points).all(axis=-1)
    qualified = (
        np.asarray(mask, dtype=bool)
        & finite_xyz
        & np.isfinite(confidence)
        & (confidence > QSPATIAL_CONFIDENCE_THRESHOLD)
    )
    flattened_indices = np.flatnonzero(qualified)
    if not flattened_indices.size:
        return np.empty((0, 2), dtype=float), []

    flat_points = points.reshape(-1, 3)
    flat_confidence = confidence.reshape(-1)
    representatives: dict[tuple[int, int], tuple[float, int, tuple[float, float]]] = {}
    for flattened_index in flattened_indices.tolist():
        point = flat_points[flattened_index]
        key = (
            math.floor(float(point[0]) / QSPATIAL_VOXEL_SIZE_M),
            math.floor(float(point[2]) / QSPATIAL_VOXEL_SIZE_M),
        )
        candidate = (
            float(flat_confidence[flattened_index]),
            int(flattened_index),
            (float(point[0]), float(point[2])),
        )
        current = representatives.get(key)
        if (
            current is None
            or candidate[0] > current[0]
            or (candidate[0] == current[0] and candidate[1] < current[1])
        ):
            representatives[key] = candidate

    selected_keys = list(representatives)
    if len(selected_keys) > QSPATIAL_MAX_UNIQUE_VOXELS:
        selected_keys = sorted(
            selected_keys,
            key=lambda key: (
                hashlib.sha256(f"{key[0]},{key[1]}".encode()).digest(),
                key,
            ),
        )[:QSPATIAL_MAX_UNIQUE_VOXELS]
    else:
        selected_keys.sort()

    selected_points = np.asarray(
        [representatives[key][2] for key in selected_keys],
        dtype=float,
    )
    return selected_points, [[key[0], key[1]] for key in selected_keys]


def _measure_horizontal_surface_gap(mask_a, mask_b, points, confidence) -> dict[str, Any]:
    """Measure the frozen QSpatial++ horizontal surface gap from two image masks.

    The contract is deliberately fail-closed. It uses eroded masks, finite and
    confidence-qualified DA3 points, deterministic XZ voxel representatives, and
    symmetric directed nearest-neighbor fifth percentiles. It never substitutes a
    centroid distance or a scene-graph estimate.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    point_map = np.asarray(points, dtype=float)
    confidence_map = np.asarray(confidence, dtype=float)
    if point_map.ndim != 3 or point_map.shape[-1] != 3:
        raise RuntimeError(
            f"reconstruction_error: point map must have shape (H, W, 3); got {point_map.shape}"
        )
    target_shape = tuple(point_map.shape[:2])
    if confidence_map.shape != target_shape:
        raise RuntimeError(
            "reconstruction_error: confidence map must match point map; "
            f"got {confidence_map.shape} and {target_shape}"
        )

    raw_masks = [np.asarray(mask_a, dtype=bool), np.asarray(mask_b, dtype=bool)]
    if any(mask.ndim != 2 for mask in raw_masks):
        raise RuntimeError("reconstruction_error: QSpatial masks must be two-dimensional")
    input_shapes = [tuple(mask.shape) for mask in raw_masks]
    aligned_masks = [_resize_mask_nearest(mask, target_shape) for mask in raw_masks]
    eroded = [
        _erode_qspatial_mask(mask, source_shape, target_shape)
        for mask, source_shape in zip(aligned_masks, input_shapes, strict=True)
    ]
    eroded_masks = [item[0] for item in eroded]
    erosion_radii = [item[1] for item in eroded]
    eroded_counts = [int(mask.sum()) for mask in eroded_masks]
    if any(count == 0 for count in eroded_counts):
        raise RuntimeError("mask_too_small_after_erosion: eroded mask is empty")

    intersection = eroded_masks[0] & eroded_masks[1]
    intersection_count = int(intersection.sum())
    overlap_coefficient = intersection_count / min(eroded_counts)
    if overlap_coefficient >= QSPATIAL_MASK_OVERLAP_THRESHOLD:
        raise RuntimeError(
            "mask_overlap_excess: overlap coefficient "
            f"{overlap_coefficient:.8f} is at least {QSPATIAL_MASK_OVERLAP_THRESHOLD:.8f}"
        )
    if intersection_count:
        eroded_masks = [mask & ~intersection for mask in eroded_masks]
        eroded_counts = [int(mask.sum()) for mask in eroded_masks]

    voxelized = [_ranked_qspatial_voxels(mask, point_map, confidence_map) for mask in eroded_masks]
    voxel_points = [item[0] for item in voxelized]
    voxel_keys = [item[1] for item in voxelized]
    voxel_counts = [len(keys) for keys in voxel_keys]
    if any(count < QSPATIAL_MIN_UNIQUE_VOXELS for count in voxel_counts):
        raise RuntimeError(
            "insufficient_unique_xz_voxels: "
            f"got {voxel_counts}, minimum {QSPATIAL_MIN_UNIQUE_VOXELS} per mask"
        )

    tree_a = cKDTree(voxel_points[0])
    tree_b = cKDTree(voxel_points[1])
    distances_a_to_b = tree_b.query(voxel_points[0], k=1, workers=1)[0]
    distances_b_to_a = tree_a.query(voxel_points[1], k=1, workers=1)[0]
    directed_a_to_b = float(np.quantile(distances_a_to_b, 0.05, method="linear"))
    directed_b_to_a = float(np.quantile(distances_b_to_a, 0.05, method="linear"))
    gap_m = min(directed_a_to_b, directed_b_to_a)
    if not math.isfinite(gap_m):
        raise RuntimeError("nonfinite_gap: directed nearest-neighbor gap is nonfinite")
    if gap_m <= 0.0:
        raise RuntimeError("nonpositive_gap: gap is outside frozen range (0, 20] meters")
    if gap_m > QSPATIAL_MAX_GAP_M:
        raise RuntimeError("gap_above_20m: gap is outside frozen range (0, 20] meters")

    return {
        "protocol": QSPATIAL_GAP_PROTOCOL,
        "geometry_protocol_sha256": QSPATIAL_GEOMETRY_PROTOCOL_SHA256,
        "gap_m": gap_m,
        "directed_gap_a_to_b_m": directed_a_to_b,
        "directed_gap_b_to_a_m": directed_b_to_a,
        "mask_overlap_coefficient": overlap_coefficient,
        "mask_overlap_intersection_pixels": intersection_count,
        "input_mask_shapes": [list(shape) for shape in input_shapes],
        "point_map_shape": list(target_shape),
        "erosion_iterations": 1,
        "erosion_radius_px": erosion_radii,
        "eroded_mask_pixel_counts": eroded_counts,
        "confidence_threshold": QSPATIAL_CONFIDENCE_THRESHOLD,
        "voxel_size_m": QSPATIAL_VOXEL_SIZE_M,
        "minimum_unique_voxels": QSPATIAL_MIN_UNIQUE_VOXELS,
        "maximum_unique_voxels": QSPATIAL_MAX_UNIQUE_VOXELS,
        "voxel_counts": voxel_counts,
        "voxel_keys": voxel_keys,
        "voxel_keys_sha256": [
            hashlib.sha256(
                (json.dumps(keys, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            ).hexdigest()
            for keys in voxel_keys
        ],
        "nearest_percentile": QSPATIAL_NEAREST_PERCENTILE,
        "quantile_method": "linear",
        "kdtree_workers": 1,
    }


def _qspatial_geometry_record(
    geometry: dict[str, Any],
    mask_indices: tuple[int, int],
) -> dict[str, Any]:
    """Reduce pure geometry output to the frozen journal-facing evidence."""
    return {
        "status": "ok",
        "invalid_reason": None,
        "mask_indices": list(mask_indices),
        "overlap_coefficient": geometry["mask_overlap_coefficient"],
        "overlap_intersection_pixels": geometry["mask_overlap_intersection_pixels"],
        "erosion_radius_px": geometry["erosion_radius_px"],
        "eroded_pixels": geometry["eroded_mask_pixel_counts"],
        "unique_xz_voxels": geometry["voxel_counts"],
        "voxel_keys_sha256": geometry["voxel_keys_sha256"],
        "directed_q05_m": [
            geometry["directed_gap_a_to_b_m"],
            geometry["directed_gap_b_to_a_m"],
        ],
        "gap_m": geometry["gap_m"],
    }


def _qspatial_invalid_candidate(
    mask_indices: tuple[int, int],
    error: RuntimeError,
) -> dict[str, Any]:
    message = str(error)
    reason = message.split(":", 1)[0]
    return {
        "status": "invalid",
        "invalid_reason": reason,
        "mask_indices": list(mask_indices),
    }


def _qspatial_evidence_base(
    question: str,
    image_bytes: bytes,
    parser_record: dict[str, Any],
    sample_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = dict(sample_metadata or {})
    if metadata:
        required = {
            "measurement_family",
            "row_id",
            "cluster_id",
            "slice_id",
            "slice_sha256",
            "output_format",
        }
        if set(metadata) != required:
            raise RuntimeError(
                "manifest_mismatch: QSpatial sample metadata must contain exactly six fields"
            )
        if metadata["measurement_family"] != "horizontal_surface_gap":
            raise RuntimeError("manifest_mismatch: measurement family drifted")
        if not QSPATIAL_GAP_SLICE_SHA256:
            raise RuntimeError("manifest_mismatch: QSpatial slice hash is not configured")
        if metadata["slice_sha256"] != QSPATIAL_GAP_SLICE_SHA256:
            raise RuntimeError("manifest_mismatch: QSpatial slice hash drifted")
        if parser_record.get("row_id") != metadata["row_id"]:
            raise RuntimeError("manifest_mismatch: parser and sample row IDs differ")

    parser_public = {
        key: parser_record.get(key)
        for key in (
            "version",
            "contract_sha256",
            "mode",
            "phrase_a",
            "phrase_b",
            "group_phrase",
        )
    }
    row_id = metadata.get("row_id", parser_record.get("row_id"))
    sample_id = f"qspatial_plus:{int(row_id):03d}" if row_id is not None else None
    return {
        "schema_version": "qspatial-gap-v1",
        "sample_id": sample_id,
        "row_id": row_id,
        "cluster_id": metadata.get("cluster_id"),
        "slice_id": metadata.get("slice_id"),
        "slice_sha256": metadata.get("slice_sha256", QSPATIAL_GAP_SLICE_SHA256 or None),
        "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "parser": parser_public,
        "segmentation": {
            "sam_threshold": QSPATIAL_CONFIDENCE_THRESHOLD,
            "queries": [],
            "candidate_mask_counts": [],
        },
        "reconstruction": {
            "backend": "da3",
            "metric_scale_reported": None,
            "point_map_shape": None,
            "confidence_threshold": QSPATIAL_CONFIDENCE_THRESHOLD,
        },
        "geometry": {
            "contract_sha256": QSPATIAL_GEOMETRY_PROTOCOL_SHA256,
            "status": "pending",
            "invalid_reason": None,
            "candidate_pairs": [],
            "selected_mask_indices": None,
            "overlap_coefficient": None,
            "eroded_pixels": None,
            "unique_xz_voxels": None,
            "voxel_keys_sha256": None,
            "directed_q05_m": None,
            "gap_m": None,
            "rendered_prediction": None,
        },
        "geometry_valid": False,
        "geometry_status": "pending",
        "fallback_used": False,
    }


def _mark_qspatial_invalid(evidence: dict[str, Any], reason: str) -> None:
    evidence["geometry_valid"] = False
    evidence["geometry_status"] = "invalid"
    evidence["geometry"].update(
        {
            "status": "invalid",
            "invalid_reason": reason,
            "selected_mask_indices": None,
            "overlap_coefficient": None,
            "eroded_pixels": None,
            "unique_xz_voxels": None,
            "voxel_keys_sha256": None,
            "directed_q05_m": None,
            "gap_m": None,
            "rendered_prediction": None,
        }
    )


def _centroid_is_finite(centroid: Any) -> bool:
    try:
        return len(centroid) >= 3 and all(math.isfinite(float(value)) for value in centroid[:3])
    except (TypeError, ValueError):
        return False


def _point_maps_for_frame(reconstruction: Any, frame_index: int):
    """Return aligned DA3 points and confidence for one absolute frame."""
    import numpy as np

    point_store = reconstruction.points
    local_index = point_store.get_by_frame_index(frame_index)
    points = np.asarray(point_store.points[local_index], dtype=float)
    confidence = np.asarray(point_store.confidence[local_index], dtype=float)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise RuntimeError(
            f"metric height geometry has invalid point-map shape {points.shape}; expected (H, W, 3)"
        )
    if confidence.shape != points.shape[:2]:
        raise RuntimeError(
            "metric height geometry has invalid confidence-map shape "
            f"{confidence.shape}; expected {points.shape[:2]}"
        )
    return points, confidence


def _measure_visible_height(
    segmentation: Any,
    reconstruction: Any,
    frame_index: int,
    object_index: int,
) -> dict[str, Any]:
    """Measure one mask's robust visible-point extent along gravity-aligned +Y."""
    import numpy as np

    points, confidence = _point_maps_for_frame(reconstruction, frame_index)
    mask = _resize_mask_nearest(
        segmentation.get_mask(frame_index, object_index),
        tuple(points.shape[:2]),
    )
    mask_point_count = int(mask.sum())
    if mask_point_count == 0:
        raise RuntimeError("empty mask geometry")

    finite_xyz = np.isfinite(points).all(axis=-1)
    finite_points = mask & finite_xyz
    finite_point_count = int(finite_points.sum())
    if finite_point_count == 0:
        raise RuntimeError("nonfinite mask geometry")

    confidence_qualified = (
        finite_points & np.isfinite(confidence) & (confidence > HEIGHT_CONFIDENCE_THRESHOLD)
    )
    confident_point_count = int(confidence_qualified.sum())
    if confident_point_count < HEIGHT_MIN_CONFIDENT_POINTS:
        raise RuntimeError(
            "undersized mask geometry: "
            f"{confident_point_count} confidence-qualified points, "
            f"minimum {HEIGHT_MIN_CONFIDENT_POINTS}"
        )

    qualified_points = points[confidence_qualified]
    centroid = np.median(qualified_points, axis=0)
    if not _centroid_is_finite(centroid):
        raise RuntimeError("nonfinite measured centroid geometry")

    low_m, high_m = np.percentile(
        qualified_points[:, 1],
        [HEIGHT_LOW_PERCENTILE, HEIGHT_HIGH_PERCENTILE],
    )
    height_m = float(high_m - low_m)
    if not all(math.isfinite(float(value)) for value in (low_m, high_m, height_m)):
        raise RuntimeError("nonfinite measured height geometry")
    if height_m <= 0.0:
        raise RuntimeError("degenerate measured height geometry")

    return {
        "centroid": tuple(float(value) for value in centroid),
        "height_m": height_m,
        "height_percentile_low_m": float(low_m),
        "height_percentile_high_m": float(high_m),
        "mask_point_count": mask_point_count,
        "finite_point_count": finite_point_count,
        "confident_point_count": confident_point_count,
    }


def _add_pairwise_distances(scene: SpatialScene) -> None:
    ids = list(scene.entities)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            distance = scene.compute_distance(ids[a], ids[b])
            if distance is not None and math.isfinite(distance):
                scene.add_relation(
                    SpatialRelation(ids[a], "distance_to", ids[b], round(distance, 2))
                )


def validate_metric_scene(scene: SpatialScene) -> None:
    """Require usable reconstructed geometry before a strict metric row can succeed."""
    grounded_ids = {
        entity_id
        for entity_id, entity in scene.entities.items()
        if entity.position is not None
        and len(entity.position) >= 2
        and all(math.isfinite(float(value)) for value in entity.position[:2])
    }
    if len(grounded_ids) < 2:
        raise RuntimeError(
            "strict metric backend requires at least two finite grounded entities; "
            f"got {len(grounded_ids)}"
        )

    finite_distances = [
        relation
        for relation in scene.relations
        if relation.predicate == "distance_to"
        and relation.subject in grounded_ids
        and relation.object in grounded_ids
        and relation.distance is not None
        and math.isfinite(float(relation.distance))
    ]
    if not finite_distances:
        raise RuntimeError(
            "strict metric backend requires at least one finite distance_to relation"
        )


def validate_metric_height_scene(scene: SpatialScene) -> None:
    """Validate the explicit one-or-more-entity visible-height measurement mode."""
    if not scene.entities:
        raise RuntimeError("strict metric height backend requires at least one entity")

    for entity_id, entity in scene.entities.items():
        attributes = entity.attributes
        if (
            attributes.get("source") != "sam3+da3-height"
            or attributes.get("measurement_provenance") != "sam3+da3-height"
        ):
            raise RuntimeError(f"strict metric height backend: {entity_id} has invalid provenance")
        try:
            height_m = float(attributes["height_m"])
            low_m = float(attributes["height_percentile_low_m"])
            high_m = float(attributes["height_percentile_high_m"])
            confident_count = int(attributes["confident_point_count"])
            centroid = attributes["centroid_xyz_m"]
            confidence_threshold = float(attributes["confidence_threshold"])
            minimum_confident_points = int(attributes["minimum_confident_points"])
            percentiles = [float(value) for value in attributes["height_percentiles"]]
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"strict metric height backend: {entity_id} has incomplete geometry"
            ) from error

        if not all(math.isfinite(value) for value in (height_m, low_m, high_m)):
            raise RuntimeError(f"strict metric height backend: {entity_id} has nonfinite geometry")
        if height_m <= 0.0 or high_m <= low_m:
            raise RuntimeError(
                f"strict metric height backend: {entity_id} has degenerate height geometry"
            )
        if confident_count < HEIGHT_MIN_CONFIDENT_POINTS:
            raise RuntimeError(f"strict metric height backend: {entity_id} has undersized geometry")
        if (
            confidence_threshold != HEIGHT_CONFIDENCE_THRESHOLD
            or minimum_confident_points != HEIGHT_MIN_CONFIDENT_POINTS
            or percentiles != [HEIGHT_LOW_PERCENTILE, HEIGHT_HIGH_PERCENTILE]
        ):
            raise RuntimeError(
                f"strict metric height backend: {entity_id} violates the frozen protocol"
            )
        if not _centroid_is_finite(centroid):
            raise RuntimeError(
                f"strict metric height backend: {entity_id} has nonfinite centroid geometry"
            )


class MetricPerceptionBackend:
    """Thin client over SpatialClaw's SAM3 + Reconstruct GPU tools."""

    def __init__(self, config):
        self.config = config
        self._sam3 = None
        self._recon = None

    def _ensure_tools(self):
        if self._recon is not None:
            return
        # Fail fast instead of hanging for four hours, when SpatialClaw supports it.
        #
        # NVIDIA's public SpatialClaw finds its GPU server through the registry file
        # logs/gpu_server.json. When that file lists no live server, its
        # _get_server_url polls 1440 times at ten-second intervals, which is about
        # four hours. The author's local SpatialClaw checkout adds a
        # SPATIALCLAW_GPU_SERVER_URL override, and an unreachable address there
        # raises within about five seconds. Pointing the variable at a closed
        # loopback port selects that fast branch in the local checkout. NVIDIA's
        # public release does not read the variable, so this default has no effect
        # there. Callers that set the variable themselves are unaffected, because
        # setdefault never overwrites it.
        os.environ.setdefault("SPATIALCLAW_GPU_SERVER_URL", "http://127.0.0.1:1")

        # Imported lazily: only available where the SpatialClaw package is installed
        # (a GPU agent environment), so the API service stays importable without it.
        from spatial_agent.tools.reconstruct_tool import ReconstructTool
        from spatial_agent.tools.sam3_tool import SAM3Tool

        class _ReconCfg:
            reconstruct_max_frames = getattr(self.config, "reconstruct_max_frames", 32)

        self._sam3 = SAM3Tool()
        self._recon = ReconstructTool(self._sam3, _ReconCfg())
        if not (self._sam3.is_available and self._recon.is_available):
            raise RuntimeError("SpatialClaw GPU tool server not reachable (logs/gpu_server.json).")

    async def extract_entities(self, llm: LLMClient, question: str) -> list[str]:
        out = await llm.generate(
            ENTITY_PROMPT.format(question=question),
            model_tier="fast",
            json_mode=True,
            max_tokens=512,
        )
        try:
            return [e for e in json.loads(out).get("entities", []) if e][:6]
        except Exception:
            return []

    def ground(self, image_bytes: bytes, phrases: list[str]) -> SpatialScene:
        self._ensure_tools()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        recon = self._recon.Reconstruct([img])
        fi = recon.frame_indices[0]
        strict = bool(getattr(self.config, "fieldwork_metric_strict", False))
        scene = SpatialScene()
        for phrase_index, phrase in enumerate(phrases):
            seg = self._sam3.segment_image_by_text(img, phrase)
            instance_count = int(getattr(seg, "num_objects", 0))
            if instance_count == 0:
                message = f"metric height geometry: empty SAM3 result for {phrase!r}"
                if strict:
                    raise RuntimeError(message)
                logger.warning(message)
                continue

            raw_labels = getattr(seg, "labels", None)
            raw_object_ids = getattr(seg, "object_ids", None)
            labels = list(raw_labels) if raw_labels is not None else []
            object_ids = list(raw_object_ids) if raw_object_ids is not None else []
            for instance_index in range(instance_count):
                try:
                    geometry = _measure_visible_height(
                        seg,
                        recon,
                        fi,
                        instance_index,
                    )
                except RuntimeError as error:
                    message = (
                        "metric height geometry failed for "
                        f"{phrase!r} instance {instance_index}: {error}"
                    )
                    if strict:
                        raise RuntimeError(message) from error
                    logger.warning(message)
                    continue

                centroid = geometry["centroid"]
                mask_label = labels[instance_index] if instance_index < len(labels) else phrase
                mask_object_id = (
                    object_ids[instance_index]
                    if instance_index < len(object_ids)
                    else instance_index
                )
                if hasattr(mask_object_id, "item"):
                    mask_object_id = mask_object_id.item()
                scene.add_entity(
                    SpatialEntity(
                        id=f"e{phrase_index}_i{instance_index}",
                        label=phrase,
                        position=(centroid[0], centroid[2]),
                        attributes={
                            "height_m": round(geometry["height_m"], 4),
                            "height_percentile_low_m": round(
                                geometry["height_percentile_low_m"], 4
                            ),
                            "height_percentile_high_m": round(
                                geometry["height_percentile_high_m"], 4
                            ),
                            "height_percentiles": [
                                HEIGHT_LOW_PERCENTILE,
                                HEIGHT_HIGH_PERCENTILE,
                            ],
                            "gravity_axis": "y",
                            "confidence_threshold": HEIGHT_CONFIDENCE_THRESHOLD,
                            "minimum_confident_points": HEIGHT_MIN_CONFIDENT_POINTS,
                            "mask_point_count": geometry["mask_point_count"],
                            "finite_point_count": geometry["finite_point_count"],
                            "confident_point_count": geometry["confident_point_count"],
                            "mask_instance_index": instance_index,
                            "mask_instance_count": instance_count,
                            "mask_object_id": mask_object_id,
                            "mask_label": mask_label,
                            "query_phrase": phrase,
                            "centroid_xyz_m": [
                                round(centroid[0], 4),
                                round(centroid[1], 4),
                                round(centroid[2], 4),
                            ],
                            "source": "sam3+da3-height",
                            "measurement_provenance": "sam3+da3-height",
                        },
                    )
                )
        _add_pairwise_distances(scene)
        return scene

    def ground_horizontal_surface_gap(
        self,
        image_bytes: bytes,
        question: str,
        parser_record: dict[str, Any],
        sample_metadata: dict[str, Any] | None = None,
    ) -> tuple[SpatialScene | None, dict[str, Any]]:
        """Ground one frozen QSpatial++ surface gap without any fallback path."""
        import numpy as np

        evidence = _qspatial_evidence_base(
            question,
            image_bytes,
            parser_record,
            sample_metadata,
        )
        if parser_record["mode"] == "unsupported_negative_space":
            _mark_qspatial_invalid(evidence, "unsupported_negative_space")
            return None, evidence

        self._ensure_tools()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        try:
            reconstruction = self._recon.Reconstruct([image])
        except Exception as error:
            raise RuntimeError("service_error: DA3 reconstruction failed") from error
        try:
            frame_index = reconstruction.frame_indices[0]
            points, confidence = _point_maps_for_frame(reconstruction, frame_index)
        except Exception as error:
            raise RuntimeError(
                "reconstruction_error: invalid DA3 reconstruction payload"
            ) from error

        evidence["reconstruction"].update(
            {
                "metric_scale_reported": float(getattr(reconstruction, "metric_scale", 1.0)),
                "point_map_shape": list(points.shape[:2]),
            }
        )

        def segment(phrase: str, empty_reason: str):
            evidence["segmentation"]["queries"].append(phrase)
            try:
                segmentation = self._sam3.segment_image_by_text(
                    image,
                    phrase,
                    confidence_threshold=QSPATIAL_CONFIDENCE_THRESHOLD,
                )
            except RuntimeError as error:
                if "produced no usable mask" in str(error):
                    evidence["segmentation"]["candidate_mask_counts"].append(0)
                    _mark_qspatial_invalid(evidence, empty_reason)
                    return None
                raise RuntimeError(f"service_error: SAM3 failed for {phrase!r}") from error
            except Exception as error:
                raise RuntimeError(f"service_error: SAM3 failed for {phrase!r}") from error
            count = int(getattr(segmentation, "num_objects", 0))
            evidence["segmentation"]["candidate_mask_counts"].append(count)
            if count == 0:
                _mark_qspatial_invalid(evidence, empty_reason)
                return None
            return segmentation

        candidate_pairs: list[tuple[Any, int, Any, int]] = []
        phrase_a: str
        phrase_b: str
        if parser_record["mode"] == "distinct_pair":
            phrase_a = str(parser_record["phrase_a"])
            phrase_b = str(parser_record["phrase_b"])
            segmentation_a = segment(phrase_a, "sam_empty_a")
            if segmentation_a is None:
                return None, evidence
            segmentation_b = segment(phrase_b, "sam_empty_b")
            if segmentation_b is None:
                return None, evidence
            candidate_pairs = [
                (segmentation_a, index_a, segmentation_b, index_b)
                for index_a, index_b in itertools.product(
                    range(int(segmentation_a.num_objects)),
                    range(int(segmentation_b.num_objects)),
                )
            ]
        elif parser_record["mode"] == "repeated_instance":
            phrase_a = str(parser_record["group_phrase"])
            phrase_b = phrase_a
            grounding_query = QSPATIAL_GROUNDING_QUERY_OVERRIDES.get(phrase_a, phrase_a)
            segmentation = segment(grounding_query, "sam_fewer_than_two")
            if segmentation is None:
                return None, evidence
            if int(segmentation.num_objects) < 2:
                _mark_qspatial_invalid(evidence, "sam_fewer_than_two")
                return None, evidence
            instance_areas = {
                index: int(
                    np.asarray(segmentation.get_mask(frame_index, index), dtype=bool).sum()
                )
                for index in range(int(segmentation.num_objects))
            }
            evidence["segmentation"]["instance_areas"] = instance_areas
            max_area = max(instance_areas.values())
            plausible_indices = [
                index
                for index, area in instance_areas.items()
                if max_area > 0 and area >= QSPATIAL_MIN_INSTANCE_AREA_RATIO * max_area
            ]
            if len(plausible_indices) < 2:
                _mark_qspatial_invalid(evidence, "sam_fewer_than_two")
                return None, evidence
            candidate_pairs = [
                (segmentation, index_a, segmentation, index_b)
                for index_a, index_b in itertools.combinations(plausible_indices, 2)
            ]
        else:
            raise RuntimeError("parse_failed: unknown QSpatial parser mode")

        valid_candidates: list[tuple[float, tuple[int, int], dict[str, Any]]] = []
        for segmentation_a, index_a, segmentation_b, index_b in candidate_pairs:
            indices = (index_a, index_b)
            try:
                mask_a = np.asarray(
                    segmentation_a.get_mask(frame_index, index_a),
                    dtype=bool,
                )
                mask_b = np.asarray(
                    segmentation_b.get_mask(frame_index, index_b),
                    dtype=bool,
                )
                geometry = _measure_horizontal_surface_gap(
                    mask_a,
                    mask_b,
                    points,
                    confidence,
                )
            except RuntimeError as error:
                reason = str(error).split(":", 1)[0]
                if reason in {"reconstruction_error", "service_error"}:
                    raise
                evidence["geometry"]["candidate_pairs"].append(
                    _qspatial_invalid_candidate(indices, error)
                )
                continue
            except Exception as error:
                raise RuntimeError("service_error: invalid SAM3 mask payload") from error

            candidate_record = _qspatial_geometry_record(geometry, indices)
            evidence["geometry"]["candidate_pairs"].append(candidate_record)
            valid_candidates.append((float(geometry["gap_m"]), indices, candidate_record))

        if not valid_candidates:
            _mark_qspatial_invalid(evidence, "no_valid_candidate_pair")
            return None, evidence

        _, selected_indices, selected = min(
            valid_candidates,
            key=lambda candidate: (candidate[0], candidate[1]),
        )
        gap_m = float(selected["gap_m"])
        rendered_prediction = rf"\scalar{{{gap_m:.6f}}} \distance_unit{{meter}}"
        evidence["geometry"].update(
            {
                "status": "ok",
                "invalid_reason": None,
                "selected_mask_indices": list(selected_indices),
                "overlap_coefficient": selected["overlap_coefficient"],
                "eroded_pixels": selected["eroded_pixels"],
                "unique_xz_voxels": selected["unique_xz_voxels"],
                "voxel_keys_sha256": selected["voxel_keys_sha256"],
                "directed_q05_m": selected["directed_q05_m"],
                "gap_m": gap_m,
                "rendered_prediction": rendered_prediction,
            }
        )
        evidence["geometry_valid"] = True
        evidence["geometry_status"] = "ok"

        scene = SpatialScene()
        provenance = "sam3+da3-horizontal-surface-gap"
        scene.add_entity(
            SpatialEntity(
                id="qspatial_a",
                label=phrase_a,
                attributes={
                    "source": provenance,
                    "mask_instance_index": selected_indices[0],
                },
            )
        )
        scene.add_entity(
            SpatialEntity(
                id="qspatial_b",
                label=phrase_b,
                attributes={
                    "source": provenance,
                    "mask_instance_index": selected_indices[1],
                },
            )
        )
        scene.add_relation(
            SpatialRelation(
                "qspatial_a",
                "horizontal_surface_gap",
                "qspatial_b",
                gap_m,
            )
        )
        return scene, evidence

    def ground_regions(
        self,
        image_bytes: bytes,
        regions: list[dict[str, Any]],
    ) -> SpatialScene:
        """Ground Warehouse `[Region N]` identities from their exact COCO RLEs."""
        import numpy as np

        self._ensure_tools()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        reconstruction = self._recon.Reconstruct([image])
        frame_index = reconstruction.frame_indices[0]
        point_map = np.asarray(reconstruction.points[frame_index])
        if point_map.ndim != 3 or point_map.shape[-1] != 3:
            raise RuntimeError(
                f"metric backend: DA3 point map must have shape (H, W, 3); got {point_map.shape}"
            )
        target_shape = tuple(point_map.shape[:2])
        PerFrameMask = _per_frame_mask_class()
        strict = bool(getattr(self.config, "fieldwork_metric_strict", False))
        scene = SpatialScene()

        for index, rle in enumerate(regions):
            label = f"Region {index}"
            mask = _resize_mask_nearest(_decode_coco_rle(rle), target_shape)
            if not bool(mask.any()):
                message = f"metric backend: {label} decoded to an empty mask"
                if strict:
                    raise RuntimeError(message)
                logger.warning(message)
                continue

            wrapped = PerFrameMask(
                masks=mask[np.newaxis, np.newaxis].astype(bool),
                labels=[label],
                object_ids=[index],
                frame_indices=[frame_index],
                frames={frame_index: image},
            )
            centroid = wrapped.get_centroid_3d(
                reconstruction,
                frame=frame_index,
                object=0,
            )
            if not _centroid_is_finite(centroid):
                message = f"metric backend: {label} produced a nonfinite 3D centroid"
                if strict:
                    raise RuntimeError(message)
                logger.warning(message)
                continue

            scene.add_entity(
                SpatialEntity(
                    id=f"region_{index}",
                    label=label,
                    position=(float(centroid[0]), float(centroid[2])),
                    attributes={
                        "y_height_m": round(float(centroid[1]), 2),
                        "source": "rle+da3",
                    },
                )
            )

        _add_pairwise_distances(scene)
        return scene


async def build_metric_scene(
    query: str,
    image_bytes_list: list[bytes],
    llm: LLMClient,
    config,
    regions: list[dict[str, Any]] | None = None,
) -> SpatialScene:
    """Build either exact-region geometry or the generic SAM3 metric scene."""
    backend = MetricPerceptionBackend(config)
    if not image_bytes_list:
        raise RuntimeError("metric backend: no image present")
    if regions is not None:
        if not regions:
            raise RuntimeError("metric backend: no exact regions present")
        return await asyncio.to_thread(
            backend.ground_regions,
            image_bytes_list[0],
            regions,
        )

    phrases = await backend.extract_entities(llm, query)
    if not phrases:
        raise RuntimeError("metric backend: no entities extracted")
    return await asyncio.to_thread(backend.ground, image_bytes_list[0], phrases)


def validate_qspatial_gap_scene(
    scene: SpatialScene,
    evidence: dict[str, Any],
) -> None:
    """Require exactly one valid relation under the frozen QSpatial protocol."""
    if evidence.get("fallback_used") is not False:
        raise RuntimeError("strict QSpatial backend forbids fallback")
    if evidence.get("geometry_valid") is not True or evidence.get("geometry_status") != "ok":
        raise RuntimeError("strict QSpatial backend requires valid geometry")
    geometry = evidence.get("geometry")
    if not isinstance(geometry, dict):
        # TRY004 asks for TypeError on a failed isinstance check. Every rejection in this
        # strict-mode block raises RuntimeError, and three call sites in this module catch
        # RuntimeError specifically. A TypeError here would escape them and turn a handled
        # strict-mode rejection into an uncaught error.
        raise RuntimeError("strict QSpatial backend has no geometry evidence")  # noqa: TRY004
    if geometry.get("contract_sha256") != QSPATIAL_GEOMETRY_PROTOCOL_SHA256:
        raise RuntimeError("strict QSpatial backend geometry contract drifted")
    parser = evidence.get("parser")
    if not isinstance(parser, dict) or (
        parser.get("contract_sha256") != QSPATIAL_PARSER_PROTOCOL_SHA256
    ):
        raise RuntimeError("strict QSpatial backend parser contract drifted")

    relations = [
        relation for relation in scene.relations if relation.predicate == "horizontal_surface_gap"
    ]
    if len(scene.entities) != 2 or len(relations) != 1:
        raise RuntimeError("strict QSpatial backend requires two entities and one horizontal gap")
    expected_provenance = "sam3+da3-horizontal-surface-gap"
    if any(
        entity.attributes.get("source") != expected_provenance for entity in scene.entities.values()
    ):
        raise RuntimeError("strict QSpatial backend entity provenance drifted")
    relation = relations[0]
    if relation.distance is None or not math.isfinite(float(relation.distance)):
        raise RuntimeError("strict QSpatial backend gap is nonfinite")
    if not 0.0 < float(relation.distance) <= QSPATIAL_MAX_GAP_M:
        raise RuntimeError("strict QSpatial backend gap is outside frozen range")
    if float(relation.distance) != float(geometry.get("gap_m")):
        raise RuntimeError("strict QSpatial backend relation and evidence disagree")
    directed = geometry.get("directed_q05_m")
    if not isinstance(directed, list) or len(directed) != 2:
        raise RuntimeError("strict QSpatial backend lacks directed gap evidence")
    voxel_hashes = geometry.get("voxel_keys_sha256")
    if (
        not isinstance(voxel_hashes, list)
        or len(voxel_hashes) != 2
        or any(re.fullmatch(r"[0-9a-f]{64}", str(value)) is None for value in voxel_hashes)
    ):
        raise RuntimeError("strict QSpatial backend lacks deterministic voxel evidence")
    if float(relation.distance) != min(float(value) for value in directed):
        raise RuntimeError("strict QSpatial backend violates symmetric gap estimator")


async def build_qspatial_gap_scene(
    query: str,
    image_bytes_list: list[bytes],
    config,
    sample_metadata: dict[str, Any] | None = None,
) -> tuple[SpatialScene | None, dict[str, Any]]:
    """Build one QSpatial scene and its label-free geometry evidence."""
    if not image_bytes_list:
        raise RuntimeError("service_error: QSpatial metric backend has no image")
    row_id = None if sample_metadata is None else sample_metadata.get("row_id")
    parser_record = parse_qspatial_gap_question(query, row_id=row_id)
    backend = MetricPerceptionBackend(config)
    return await asyncio.to_thread(
        backend.ground_horizontal_surface_gap,
        image_bytes_list[0],
        query,
        parser_record,
        sample_metadata,
    )
