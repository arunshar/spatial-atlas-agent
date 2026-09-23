"""Run Spatial Atlas directly on a SpatialClaw benchmark.

The driver deliberately imports SpatialClaw's benchmark factory and scoring helpers from a
checkout supplied at runtime. This keeps dataset parsing and metrics identical across the
Spatial Atlas and SpatialClaw-native runs.

Example:
    uv run python eval_bench.py \
      --benchmark omni3d \
      --data-root /path/containing/Omni3D-Bench \
      --engine metric \
      --subsample 5 \
      --output-dir work_dir/omni3d_metric \
      --spatialclaw-root ../SpatialClaw
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import mimetypes
import os
import random
import re
import sys
import time
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Sequence

import litellm

# The repository uses a src layout but this root-level script is run directly.
_SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from config import Config  # noqa: E402 - src bootstrap must run before local imports
from fieldwork.handler import FieldWorkHandler  # noqa: E402
from llm import LLMClient  # noqa: E402

SUBSAMPLE_SEED = 42
PREDICTIONS_FILENAME = "predictions.jsonl"
SUMMARY_FILENAME = "results_summary.json"
MANIFEST_FILENAME = "run_manifest.json"
PRESCORE_SCHEMA = "label_free_prescore_v1"
NORMAL_SCHEMA = "ground_truth_scored_v1"
QSPATIAL_METRIC_PROTOCOL = "qspatial-horizontal-gap-v1"
MEASUREMENT_UNAVAILABLE = "measurement unavailable"


def _configured_sha256(name: str) -> str:
    """Return an operator-supplied SHA-256 hex digest, or "" when it is unset or malformed."""
    value = os.environ.get(name, "").strip().lower()
    return value if re.fullmatch(r"[0-9a-f]{64}", value) else ""


# The frozen control mapping and its image manifest are private control artifacts and are not
# bundled with this repository. Their digests are supplied at run time. Loading the mapping
# fails closed when either digest is unset.
QSPATIAL_CONTROL_MAPPING_SHA256 = _configured_sha256("ATLAS_QSPATIAL_CONTROL_MAPPING_SHA256")
QSPATIAL_IMAGE_MANIFEST_SHA256 = _configured_sha256("ATLAS_QSPATIAL_IMAGE_MANIFEST_SHA256")

QSPATIAL_CONTROL_IMAGE_COUNT = 84
EXPECTED_LITELLM_VERSION = "1.92.0"
LITELLM_FLUSH_TIMEOUT_SECONDS = 60.0

_PRESCORE_METADATA_FIELDS = (
    "measurement_family",
    "row_id",
    "cluster_id",
    "slice_id",
    "slice_sha256",
    "output_format",
)
_FORBIDDEN_PRESCORE_EXACT_KEYS = {
    "gt",
    "ground_truth",
    "answer",
    "answer_value",
    "answer_unit",
    "result",
    "results",
    "score",
    "scores",
    "label",
    "labels",
    "target",
    "targets",
    "gold",
    "gold_answer",
    "gold_value",
    "expected_answer",
    "expected_value",
    "expected_unit",
    "groundtruth",
    "truth",
    "reference_answer",
    "reference_value",
    "correct_answer",
    "correct_value",
}
_FORBIDDEN_PRESCORE_PREFIXES = (
    "actual",
    "annotation",
    "answer",
    "correct",
    "expected",
    "gold",
    "groundtruth",
    "gt",
    "label",
    "reference",
    "supervision",
    "target",
    "truth",
    "true",
)


class NoOpUpdater:
    """TaskUpdater-compatible sink for direct handler calls."""

    async def update_status(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _normalize_question_types(values: Iterable[str] | None) -> list[str]:
    normalized: list[str] = []
    for value in values or []:
        normalized.extend(part.strip() for part in value.split(",") if part.strip())
    return normalized


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, help="SpatialClaw benchmark registry key")
    parser.add_argument(
        "--data-root",
        default="data",
        help="Dataset root passed to SpatialClaw's BenchmarkFactory (default: data)",
    )
    parser.add_argument(
        "--engine",
        choices=("scenegraph", "metric"),
        required=True,
        help="Spatial Atlas FieldWork engine",
    )
    parser.add_argument(
        "--question-type",
        dest="question_types",
        action="append",
        default=[],
        help="Question-type filter. Repeat the flag or pass comma-separated values.",
    )
    parser.add_argument(
        "--subsample",
        type=_positive_int,
        default=None,
        metavar="N",
        help=f"Shuffle deterministically with seed {SUBSAMPLE_SEED}, then keep N samples",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for JSONL and summary output",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip only successful sample IDs already present in predictions.jsonl",
    )
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=1,
        metavar="N",
        help="Number of independent Config/LLMClient/FieldWorkHandler workers (default: 1)",
    )
    parser.add_argument(
        "--spatialclaw-root",
        default=os.environ.get("SPATIALCLAW_ROOT"),
        help="SpatialClaw checkout root (or set SPATIALCLAW_ROOT)",
    )
    parser.add_argument(
        "--image-mode",
        choices=("correct", "question-only", "shuffled"),
        default="correct",
        help=("Image ablation for label-free QSpatial runs. Shuffled requires --control-mapping."),
    )
    parser.add_argument(
        "--control-mapping",
        default=None,
        help="Frozen QSpatial cyclic-next mapping JSON for --image-mode shuffled",
    )
    args = parser.parse_args(argv)
    if not args.spatialclaw_root:
        parser.error("--spatialclaw-root or SPATIALCLAW_ROOT is required")
    if args.image_mode == "question-only" and args.engine != "scenegraph":
        parser.error("--image-mode question-only requires --engine scenegraph")
    if args.image_mode == "shuffled" and args.engine != "metric":
        parser.error("--image-mode shuffled requires --engine metric")
    if args.image_mode == "shuffled" and not args.control_mapping:
        parser.error("--image-mode shuffled requires --control-mapping")
    if args.image_mode != "shuffled" and args.control_mapping:
        parser.error("--control-mapping is valid only with --image-mode shuffled")
    args.question_types = _normalize_question_types(args.question_types)
    return args


def _load_spatialclaw_modules(root: str | os.PathLike[str]) -> tuple[ModuleType, ModuleType]:
    checkout = Path(root).expanduser().resolve()
    required = (
        checkout / "spatial_agent" / "evals" / "factory.py",
        checkout / "spatial_agent" / "evals" / "scoring.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SpatialClaw checkout is missing: {', '.join(missing)}")

    checkout_str = str(checkout)
    while checkout_str in sys.path:
        sys.path.remove(checkout_str)
    sys.path.insert(0, checkout_str)

    factory = importlib.import_module("spatial_agent.evals.factory")
    scoring = importlib.import_module("spatial_agent.evals.scoring")
    return factory, scoring


def _select_samples(benchmark: Any, subsample: int | None) -> list[Any]:
    samples = list(benchmark)
    if subsample is None:
        return samples
    rng = random.Random(SUBSAMPLE_SEED)
    rng.shuffle(samples)
    return samples[:subsample]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    return str(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_qspatial_control_mapping(
    path: str | os.PathLike[str],
    image_root: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Validate the frozen cyclic-next mapping and every referenced image."""
    if not (QSPATIAL_CONTROL_MAPPING_SHA256 and QSPATIAL_IMAGE_MANIFEST_SHA256):
        raise ValueError(
            "QSpatial control mapping digests are not configured: set "
            "ATLAS_QSPATIAL_CONTROL_MAPPING_SHA256 and ATLAS_QSPATIAL_IMAGE_MANIFEST_SHA256"
        )
    mapping_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid QSpatial control mapping: {error}") from error
    required = {
        "algorithm": "utf8-byte-sorted-cyclic-next",
        "count": QSPATIAL_CONTROL_IMAGE_COUNT,
        "files_verified": True,
        "image_manifest_sha256": QSPATIAL_IMAGE_MANIFEST_SHA256,
        "kind": "qspatial_negative_control_mapping",
        "mapping_sha256": QSPATIAL_CONTROL_MAPPING_SHA256,
        "schema_version": 1,
    }
    mismatches = {
        key: payload.get(key) for key, expected in required.items() if payload.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"QSpatial control mapping metadata drifted: {mismatches}")
    mappings = payload.get("mappings")
    if not isinstance(mappings, list) or len(mappings) != QSPATIAL_CONTROL_IMAGE_COUNT:
        raise ValueError("QSpatial control mapping must contain exactly 84 pairs")
    if _canonical_sha256(mappings) != QSPATIAL_CONTROL_MAPPING_SHA256:
        raise ValueError("QSpatial control mapping pair hash drifted")

    root = image_root.expanduser().resolve()
    lookup: dict[str, dict[str, str]] = {}
    digest_cache: dict[str, str] = {}
    expected_fields = {
        "source_path",
        "source_sha256",
        "control_path",
        "control_sha256",
    }
    for index, raw in enumerate(mappings):
        if not isinstance(raw, Mapping) or set(raw) != expected_fields:
            raise ValueError(f"QSpatial control pair {index} has invalid fields")
        record = {key: str(raw[key]) for key in expected_fields}
        for path_key, digest_key in (
            ("source_path", "source_sha256"),
            ("control_path", "control_sha256"),
        ):
            relative = Path(record[path_key])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"QSpatial control pair {index} has an unsafe path")
            disk_path = (root / relative).resolve()
            try:
                disk_path.relative_to(root)
            except ValueError as error:
                raise ValueError(f"QSpatial control pair {index} escapes the image root") from error
            if not disk_path.is_file():
                raise FileNotFoundError(f"QSpatial control image not found: {disk_path}")
            cache_key = str(disk_path)
            actual = digest_cache.get(cache_key)
            if actual is None:
                actual = _sha256_file(disk_path)
                digest_cache[cache_key] = actual
            if actual != record[digest_key]:
                raise ValueError(f"QSpatial control image digest drifted: {disk_path}")
        if record["source_path"] in lookup:
            raise ValueError("QSpatial control mapping repeats a source image")
        if record["source_path"] == record["control_path"]:
            raise ValueError("QSpatial control mapping contains a fixed point")
        lookup[record["source_path"]] = record
    if len(lookup) != QSPATIAL_CONTROL_IMAGE_COUNT:
        raise ValueError("QSpatial control mapping source count drifted")
    if set(lookup) != {record["control_path"] for record in lookup.values()}:
        raise ValueError("QSpatial control mapping is not a bijection")
    return lookup, {
        "path": str(mapping_path),
        "sha256": _sha256_file(mapping_path),
        "mapping_sha256": QSPATIAL_CONTROL_MAPPING_SHA256,
        "image_manifest_sha256": QSPATIAL_IMAGE_MANIFEST_SHA256,
    }


def _forbidden_prescore_key(key: Any) -> bool:
    raw = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key).strip())
    normalized = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    tokens = normalized.split("_")
    compact = "".join(tokens)
    return (
        normalized in _FORBIDDEN_PRESCORE_EXACT_KEYS
        or "answer" in tokens
        or "label" in tokens
        or "result" in tokens
        or "score" in tokens
        or "scores" in tokens
        or compact.startswith(_FORBIDDEN_PRESCORE_PREFIXES)
        or normalized.endswith(("_score", "_scores", "_ground_truth", "_groundtruth"))
    )


def _assert_label_free(value: Any, description: str, path: str = "$") -> None:
    """Reject label-bearing fields anywhere in a prescore payload."""
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _forbidden_prescore_key(key):
                raise ValueError(f"{description} contains forbidden label field at {path}.{key}")
            _assert_label_free(nested, description, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _assert_label_free(nested, description, f"{path}[{index}]")


def _sample_journal_ground_truth(sample: Any) -> bool:
    value = getattr(sample, "journal_ground_truth", True)
    if not isinstance(value, bool):
        raise ValueError(
            "sample journal_ground_truth must be boolean; "
            f"sample {getattr(sample, 'sample_id', '?')} has {value!r}"
        )
    return value


def _prescore_metadata(sample: Any) -> dict[str, Any]:
    provider = getattr(sample, "prediction_metadata", None)
    if not callable(provider):
        raise ValueError(
            f"label-free sample {getattr(sample, 'sample_id', '?')} "
            "must implement prediction_metadata()"
        )
    supplied = provider()
    if not isinstance(supplied, Mapping):
        raise ValueError("prediction_metadata() must return a mapping")
    actual_fields = set(supplied)
    expected_fields = set(_PRESCORE_METADATA_FIELDS)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        unexpected = sorted(actual_fields - expected_fields)
        raise ValueError(
            "prediction metadata must contain exactly the six safe fields; "
            f"missing={missing}, unexpected={unexpected}"
        )
    metadata = {field: supplied[field] for field in _PRESCORE_METADATA_FIELDS}
    _assert_label_free(metadata, "prediction metadata")
    return metadata


def _preflight_journal_schema(
    selected_samples: Sequence[Any],
) -> tuple[str, dict[int, dict[str, Any]]]:
    ground_truth_flags = [_sample_journal_ground_truth(sample) for sample in selected_samples]
    if len(set(ground_truth_flags)) != 1:
        raise ValueError(
            "benchmark selection mixes normal and label-free prescore samples; "
            "use separate output directories and runs"
        )
    if ground_truth_flags[0]:
        return NORMAL_SCHEMA, {}
    metadata_by_object = {id(sample): _prescore_metadata(sample) for sample in selected_samples}
    return PRESCORE_SCHEMA, metadata_by_object


def _record_journal_schema(record: Mapping[str, Any]) -> str:
    has_prescore = "prediction_text" in record
    has_normal = any(field in record for field in ("answer", "gt", "score"))
    if has_prescore and has_normal:
        raise ValueError("journal row mixes normal and label-free prescore fields")
    if has_prescore:
        _assert_label_free(record, "prescore journal row")
        return PRESCORE_SCHEMA
    if has_normal:
        return NORMAL_SCHEMA
    raise ValueError("journal row has neither a normal nor label-free prescore schema")


def _validate_resume_record_schemas(
    records: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    expected_schema: str,
) -> None:
    row_values = records.values() if isinstance(records, Mapping) else records
    schemas = {_record_journal_schema(record) for record in row_values}
    if len(schemas) > 1:
        raise ValueError("resume journal mixes normal and label-free prescore rows")
    if schemas and schemas != {expected_schema}:
        raise ValueError(
            f"resume journal schema mismatch: stored={next(iter(schemas))!r}, "
            f"requested={expected_schema!r}"
        )


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
                sample_id = record["sample_id"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            records[str(sample_id)] = record
    return records


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Read every valid identified row without collapsing duplicate sample IDs."""
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(record, dict) and record.get("sample_id") is not None:
                rows.append(record)
    return rows


def _repair_jsonl_tail(path: Path) -> bool:
    """Remove one invalid trailing fragment, or terminate a valid final record."""
    if not path.is_file():
        return False
    data = path.read_bytes()
    if not data:
        return False

    lines = data.splitlines(keepends=True)
    offset = 0
    for index, line in enumerate(lines):
        payload = line.rstrip(b"\r\n")
        try:
            json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if index != len(lines) - 1:
                raise ValueError(
                    f"JSONL journal has an invalid non-trailing record at byte {offset}"
                )
            descriptor = os.open(path, os.O_WRONLY)
            try:
                os.ftruncate(descriptor, offset)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return True
        offset += len(line)

    if not data.endswith((b"\n", b"\r")):
        descriptor = os.open(path, os.O_APPEND | os.O_WRONLY)
        try:
            if os.write(descriptor, b"\n") != 1:
                raise OSError("short write while terminating JSONL record")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return True
    return False


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a small JSON control or summary file."""
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(
            payload,
            default=_json_default,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(record, default=_json_default, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError(f"short JSONL write: {written}/{len(payload)} bytes")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


_USAGE_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "num_calls",
    "estimated_cost_usd",
)


def _usage_snapshot(llm: Any) -> dict[str, int | float] | None:
    tracker = getattr(llm, "cost_tracker", None)
    stats = getattr(tracker, "stats", None)
    if stats is None:
        return None
    snapshot: dict[str, int | float] = {}
    for field in _USAGE_FIELDS:
        value = stats.get(field, 0) if isinstance(stats, Mapping) else getattr(stats, field, 0)
        snapshot[field] = value or 0
    return snapshot


def _usage_delta(
    before: Mapping[str, int | float] | None,
    after: Mapping[str, int | float] | None,
) -> dict[str, int | float] | None:
    if before is None or after is None:
        return None
    return {field: after.get(field, 0) - before.get(field, 0) for field in _USAGE_FIELDS}


def _llm_call_telemetry_cursor(llm: Any) -> int | None:
    tracker = getattr(llm, "cost_tracker", None)
    cursor_method = getattr(tracker, "telemetry_cursor", None)
    records_method = getattr(tracker, "telemetry_since", None)
    if cursor_method is None and records_method is None:
        return None
    if not callable(cursor_method) or not callable(records_method):
        raise TypeError("cost tracker must expose both telemetry_cursor and telemetry_since")
    cursor = cursor_method()
    if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
        raise ValueError("cost tracker returned an invalid telemetry cursor")
    return cursor


def _llm_call_telemetry_delta(llm: Any, cursor: int | None) -> list[dict[str, Any]] | None:
    if cursor is None:
        return None
    tracker = getattr(llm, "cost_tracker", None)
    records_method = getattr(tracker, "telemetry_since", None)
    if not callable(records_method):
        raise TypeError("cost tracker telemetry_since is unavailable")
    records = records_method(cursor)
    if not isinstance(records, list) or any(not isinstance(record, Mapping) for record in records):
        raise TypeError("cost tracker telemetry_since must return a list of mappings")
    return [dict(record) for record in records]


def _sample_observability(
    llm: Any,
    before_usage: Mapping[str, int | float] | None,
    telemetry_cursor: int | None,
) -> tuple[dict[str, int | float] | None, list[dict[str, Any]] | None]:
    usage = _usage_delta(before_usage, _usage_snapshot(llm))
    telemetry = _llm_call_telemetry_delta(llm, telemetry_cursor)
    if telemetry is not None:
        if usage is None:
            raise ValueError("LLM call telemetry requires aggregate usage statistics")
        num_calls = usage.get("num_calls")
        if isinstance(num_calls, bool) or not isinstance(num_calls, int) or num_calls < 0:
            raise ValueError("aggregate usage num_calls must be a nonnegative integer")
        if len(telemetry) != num_calls:
            raise ValueError(
                "LLM call telemetry count does not match usage num_calls: "
                f"{len(telemetry)} != {num_calls}"
            )
    return usage, telemetry


def _infer_output_format(sample: Any) -> str:
    explicit = getattr(sample, "output_format", None)
    if explicit:
        return str(explicit)
    answer_type = str(getattr(sample, "answer_type", "")).lower()
    question_type = str(getattr(sample, "question_type", "")).lower()
    if answer_type in {"float", "int", "number", "numeric"}:
        return "number"
    if question_type in {"count", "distance", "mcq"}:
        return "number"
    return "text"


# This function is derived from spatial_agent/entrypoints/run.py in NVIDIA SpatialClaw.
# Copyright (c) 2026-Present, NVIDIA Corporation & affiliates. All rights reserved.
# NVIDIA licenses it under the NVIDIA Source Code License-NC, and the full text is in
# LICENSES/NVIDIA-Source-Code-License-NC.txt. It may be used only for non-commercial
# scientific research, and the MIT License of this repository does not cover it.
def _benchmark_instruction(sample: Any, data_specific_prompt: str) -> str:
    """Build the instruction in the text format of NVIDIA SpatialClaw's run loop.

    The format follows spatial_agent/entrypoints/run.py in NVIDIA SpatialClaw, so that Atlas
    runs and SpatialClaw-native runs see the same prompt. The format comes from NVIDIA
    SpatialClaw.
    """
    instruction = f"{sample.question}\n\n{data_specific_prompt}"
    choices = getattr(sample, "choices", None)
    if isinstance(choices, dict):
        for letter, choice in choices.items():
            instruction += f"\n{letter}. {choice}"
    elif isinstance(choices, list):
        for index, choice in enumerate(choices):
            instruction += f"\n{chr(65 + index)}. {choice}"
    return instruction


def _build_goal(sample: Any, image_name: str, data_specific_prompt: str) -> str:
    instruction = _benchmark_instruction(sample, data_specific_prompt)
    return (
        f"# Question\n{instruction}\n\n"
        f"# Input Data\n{image_name}\n\n"
        f"# Output Format\n{_infer_output_format(sample)}\n"
    )


def _resolve_single_image(sample: Any) -> Path:
    ensure_loaded = getattr(sample, "ensure_frames_loaded", None)
    if callable(ensure_loaded):
        ensure_loaded()
    images = list(getattr(sample, "images", None) or [])
    if len(images) != 1:
        raise ValueError(
            f"sample {getattr(sample, 'sample_id', '?')} must resolve to exactly one image; "
            f"got {len(images)}"
        )
    path = Path(images[0]).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"sample image not found: {path}")
    return path


async def _run_sample(
    sample: Any,
    benchmark: Any,
    handler: Any,
    llm: Any,
    engine: str,
    *,
    journal_schema: str,
    prescore_metadata: Mapping[str, Any] | None = None,
    image_mode: str = "correct",
    image_root: Path | None = None,
    control_lookup: Mapping[str, Mapping[str, str]] | None = None,
    control_mapping_sha256: str | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    before_usage = _usage_snapshot(llm)
    telemetry_cursor = _llm_call_telemetry_cursor(llm)
    image_path: Path | None = None
    source_image_path: Path | None = None
    input_image_sha256: str | None = None
    answer: str | None = None
    score: float | None = None
    error: str | None = None
    setattr(handler, "last_fieldwork_engine", None)
    setattr(handler, "last_metric_grounding", None)
    setattr(handler, "last_metric_evidence", None)
    metric_kwargs: dict[str, Any] = {}
    metric_evidence: dict[str, Any] | None = None

    try:
        source_image_path = _resolve_single_image(sample).resolve()
        image_bytes: bytes | None
        if image_mode == "correct":
            image_path = source_image_path
            image_bytes = image_path.read_bytes()
        elif image_mode == "question-only":
            image_path = None
            image_bytes = None
        elif image_mode == "shuffled":
            if image_root is None or control_lookup is None:
                raise ValueError("shuffled image mode has no validated control mapping")
            root = image_root.expanduser().resolve()
            try:
                relative_source = source_image_path.relative_to(root).as_posix()
            except ValueError as error:
                raise ValueError(
                    f"sample image is outside the QSpatial image root: {source_image_path}"
                ) from error
            mapping = control_lookup.get(relative_source)
            if mapping is None:
                raise ValueError(
                    f"sample image is absent from the frozen control mapping: {relative_source}"
                )
            image_path = (root / mapping["control_path"]).resolve()
            image_bytes = image_path.read_bytes()
            if _sha256_file(image_path) != mapping["control_sha256"]:
                raise ValueError(f"shuffled control image digest drifted: {image_path}")
        else:
            raise ValueError(f"unsupported image mode: {image_mode!r}")

        if image_path is not None and image_bytes is not None:
            input_image_sha256 = hashlib.sha256(image_bytes).hexdigest()
            mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
            file_parts = [(image_path.name, mime_type, image_bytes)]
            input_name = image_path.name
        else:
            file_parts = []
            input_name = "question-only-no-image"
        goal = _build_goal(
            sample,
            input_name,
            str(getattr(benchmark, "data_specific_prompt", "")),
        )
        if engine == "metric":
            source_image = getattr(sample, "source_image", None)
            regions = getattr(sample, "rles", None)
            if source_image and regions is not None:
                source_path = Path(source_image).expanduser()
                if not source_path.is_file():
                    raise FileNotFoundError(f"metric source image not found: {source_path}")
                metric_kwargs = {
                    "metric_image_bytes": source_path.read_bytes(),
                    "metric_regions": regions,
                }
            if journal_schema == PRESCORE_SCHEMA:
                if prescore_metadata is None:
                    raise ValueError("label-free metric sample has no safe metadata")
                if prescore_metadata.get("measurement_family") != "horizontal_surface_gap":
                    raise ValueError("label-free metric samples must use horizontal_surface_gap")
                metric_kwargs.update(
                    {
                        "metric_protocol": QSPATIAL_METRIC_PROTOCOL,
                        "metric_sample_metadata": dict(prescore_metadata),
                        "metric_question": str(sample.question),
                    }
                )
                if image_bytes is None:
                    raise ValueError("QSpatial metric mode requires an input image")
        answer = await handler.handle(
            goal,
            file_parts,
            NoOpUpdater(),
            **metric_kwargs,
        )
        if answer is None or not str(answer).strip():
            raise RuntimeError("handler returned an empty answer")
        if journal_schema == PRESCORE_SCHEMA:
            raw_evidence = getattr(handler, "last_metric_evidence", None)
            if raw_evidence is not None:
                if not isinstance(raw_evidence, Mapping):
                    raise TypeError("handler last_metric_evidence must be a mapping or None")
                metric_evidence = dict(raw_evidence)
                _assert_label_free(metric_evidence, "metric evidence")
    except Exception as exc:  # noqa: BLE001 - persist the failure and continue the run
        error = f"{type(exc).__name__}: {exc}"
        metric_evidence = None

    usage_delta, llm_call_telemetry = _sample_observability(
        llm,
        before_usage,
        telemetry_cursor,
    )
    used_engine = getattr(handler, "last_fieldwork_engine", None)
    if used_engine is None and error is None:
        used_engine = engine
    grounding_source = getattr(handler, "last_metric_grounding", None)
    if grounding_source is None and error is None and used_engine == "metric":
        grounding_source = "rle+da3" if "metric_regions" in metric_kwargs else "sam3+da3"
    if journal_schema == PRESCORE_SCHEMA:
        prediction_text = None if answer is None else str(answer)
        unavailable = (
            prediction_text is not None
            and prediction_text.strip().lower() == MEASUREMENT_UNAVAILABLE
        )
        if metric_evidence is None:
            geometry_valid = error is None and not unavailable
            geometry_status = (
                "runtime_error"
                if error is not None
                else "not_applicable"
                if engine != "metric"
                else "measurement_unavailable"
                if unavailable
                else "evidence_unavailable"
            )
            fallback_used = False
        else:
            geometry_valid = metric_evidence.get("geometry_valid")
            geometry_status = metric_evidence.get("geometry_status")
            fallback_used = metric_evidence.get("fallback_used")
            if not isinstance(geometry_valid, bool):
                error = "ValueError: metric evidence geometry_valid must be boolean"
                geometry_valid = False
            if not isinstance(geometry_status, str) or not geometry_status.strip():
                error = "ValueError: metric evidence geometry_status must be nonempty"
                geometry_status = "invalid_evidence"
            if fallback_used is not False:
                error = "ValueError: metric evidence fallback_used must be false"
                fallback_used = False
            if error is None and geometry_valid == unavailable:
                error = "ValueError: metric evidence geometry_valid conflicts with prediction_text"
        assert prescore_metadata is not None
        record = {
            "schema_version": 1,
            "sample_id": getattr(sample, "sample_id", None),
            "prediction_text": prediction_text,
            **dict(prescore_metadata),
            "usage": usage_delta,
            "llm_call_telemetry": llm_call_telemetry,
            "latency_seconds": round(time.perf_counter() - start, 6),
            "engine": engine,
            "requested_engine": engine,
            "used_engine": used_engine,
            "grounding_source": grounding_source,
            "question_type": getattr(sample, "question_type", None),
            "image_path": str(image_path) if image_path is not None else None,
            "source_image_path": (
                str(source_image_path) if source_image_path is not None else None
            ),
            "image_mode": image_mode,
            "input_image_sha256": input_image_sha256,
            "control_mapping_sha256": control_mapping_sha256,
            "geometry_valid": geometry_valid,
            "geometry_status": geometry_status,
            "fallback_used": fallback_used,
            "metric_evidence": metric_evidence,
            "error": error,
        }
        _assert_label_free(record, "prescore journal row")
        return record
    return {
        "sample_id": getattr(sample, "sample_id", None),
        "answer": answer,
        "gt": getattr(sample, "answer", None),
        "score": score,
        "usage": usage_delta,
        "llm_call_telemetry": llm_call_telemetry,
        "latency_seconds": round(time.perf_counter() - start, 6),
        "engine": engine,
        "requested_engine": engine,
        "used_engine": used_engine,
        "grounding_source": grounding_source,
        "question_type": getattr(sample, "question_type", None),
        "image_path": str(image_path) if image_path is not None else None,
        "error": error,
    }


def _score_record(benchmark: Any, sample: Any, record: dict[str, Any]) -> None:
    """Apply the benchmark's per-sample scorer in deterministic selection order."""
    if record.get("error") is not None:
        return
    try:
        record["score"] = benchmark.evaluate_single(sample, record["answer"])
    except Exception as exc:  # noqa: BLE001 - persist scorer failures like handler failures
        record["score"] = None
        record["error"] = f"{type(exc).__name__}: {exc}"


def _resolved_model_tiers(handler: Any) -> dict[str, str]:
    tiers = getattr(getattr(handler, "config", None), "model_tiers", None)
    if isinstance(tiers, Mapping):
        return {str(key): str(value) for key, value in tiers.items()}
    return {}


def _resolved_llm_enable_thinking(handler: Any) -> bool | None:
    """Return the resolved chat-template thinking mode used by this run."""
    config = getattr(handler, "config", None)
    value = getattr(config, "llm_enable_thinking", None)
    if value not in (True, False, None):
        raise ValueError(
            f"handler config llm_enable_thinking must be true, false, or None; got {value!r}"
        )
    return value


def _build_manifest(
    benchmark: Any,
    selected_samples: list[Any],
    handler: Any,
    *,
    engine: str,
    benchmark_name: str | None,
    data_root: str | os.PathLike[str] | None,
    question_types: Sequence[str] | None,
    model_tiers: Mapping[str, str] | None,
    concurrency: int = 1,
    journal_schema: str | None = None,
    image_mode: str = "correct",
    control_mapping_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_root = data_root
    if resolved_root is None:
        resolved_root = getattr(benchmark, "data_path", None)
    resolved_filters = question_types
    if resolved_filters is None:
        resolved_filters = getattr(benchmark, "question_type_filter", None)
    if journal_schema is None:
        flags = {_sample_journal_ground_truth(sample) for sample in selected_samples}
        if len(flags) != 1:
            raise ValueError("cannot build a manifest for mixed journal schemas")
        journal_schema = NORMAL_SCHEMA if flags.pop() else PRESCORE_SCHEMA
    return {
        "benchmark": benchmark_name or benchmark.__class__.__name__,
        "data_root": (
            str(Path(resolved_root).expanduser().resolve()) if resolved_root is not None else None
        ),
        "question_filters": [str(value) for value in (resolved_filters or [])],
        "engine": engine,
        "model_tiers": dict(model_tiers or _resolved_model_tiers(handler)),
        "llm_enable_thinking": _resolved_llm_enable_thinking(handler),
        "concurrency": concurrency,
        "journal_schema": journal_schema,
        "image_mode": image_mode,
        "control_mapping_artifact": (
            dict(control_mapping_artifact) if control_mapping_artifact is not None else None
        ),
        "seed": SUBSAMPLE_SEED,
        "selected_ids": [str(sample.sample_id) for sample in selected_samples],
    }


def _validate_manifest(path: Path, requested: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise ValueError(f"resume requires {MANIFEST_FILENAME}; start a fresh output directory")
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid resume manifest: {exc}") from exc
    mismatches = [key for key, value in requested.items() if stored.get(key) != value]
    if mismatches:
        details = ", ".join(
            f"{key}: stored={stored.get(key)!r}, requested={requested[key]!r}" for key in mismatches
        )
        raise ValueError(f"resume manifest mismatch: {details}")


_OPERATIONAL_USAGE_FIELDS = {
    "prompt_tokens": "mean_prompt_tokens",
    "completion_tokens": "mean_completion_tokens",
    "total_tokens": "mean_total_tokens",
    "num_calls": "mean_calls",
    "estimated_cost_usd": "mean_estimated_cost_usd",
}


def _record_completed(record: Mapping[str, Any]) -> bool:
    prediction_field = "prediction_text" if "prediction_text" in record else "answer"
    prediction = record.get(prediction_field)
    return (
        record.get("error") in (None, "")
        and prediction is not None
        and bool(str(prediction).strip())
    )


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _operational_group(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if _record_completed(record)]
    result: dict[str, Any] = {
        "sample_count": len(records),
        "completed_count": len(completed),
        "error_count": len(records) - len(completed),
        "mean_latency_seconds": _mean(
            [
                float(record["latency_seconds"])
                for record in records
                if isinstance(record.get("latency_seconds"), (int, float))
            ]
        ),
    }
    for source_name, output_name in _OPERATIONAL_USAGE_FIELDS.items():
        values = [
            float(usage[source_name])
            for record in records
            if isinstance((usage := record.get("usage")), Mapping)
            and isinstance(usage.get(source_name), (int, float))
        ]
        result[output_name] = _mean(values)
    return result


def _operational_summary(
    selected_samples: list[Any],
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    latest = [records[str(sample.sample_id)] for sample in selected_samples]
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in latest:
        question_type = str(record.get("question_type") or "unknown")
        grouped.setdefault(question_type, []).append(record)
    overall = _operational_group(latest)
    return {
        "selected_count": len(latest),
        "completed_count": overall["completed_count"],
        "error_count": overall["error_count"],
        "overall": overall,
        "per_question_type": {
            question_type: _operational_group(group_records)
            for question_type, group_records in sorted(grouped.items())
        },
    }


def _extend_results_summary(path: Path, operational: Mapping[str, Any]) -> None:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"official scorer did not write {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"official scorer wrote invalid summary JSON: {exc}") from exc
    if not isinstance(summary, dict):
        raise RuntimeError("official scorer summary must be a JSON object")
    summary["operational"] = operational
    _write_json(path, summary)


def _evaluate_selected(
    benchmark: Any,
    selected_samples: list[Any],
    records: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    scoring_module: ModuleType,
) -> dict[str, Any]:
    predictions = {
        str(sample.sample_id): (records.get(str(sample.sample_id), {}).get("answer") or "")
        for sample in selected_samples
    }
    original_data = benchmark.data
    benchmark.data = selected_samples
    try:
        results = benchmark.evaluate(predictions, output_dir=str(output_dir))
    finally:
        benchmark.data = original_data
    scoring_module.write_results_summary(str(output_dir), results)
    _extend_results_summary(
        output_dir / SUMMARY_FILENAME,
        _operational_summary(selected_samples, records),
    )
    return results


def _write_inference_only_summary(
    selected_samples: list[Any],
    records: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Write operational facts only, without loading or evaluating labels."""
    results = {
        "total_samples": len(selected_samples),
        "inference_only": True,
        "journal_schema": PRESCORE_SCHEMA,
        "operational": _operational_summary(selected_samples, records),
    }
    _write_json(output_dir / SUMMARY_FILENAME, results)
    return results


async def run_benchmark(
    benchmark: Any,
    handler: Any,
    llm: Any,
    scoring_module: ModuleType,
    *,
    engine: str,
    output_dir: str | os.PathLike[str],
    subsample: int | None = None,
    resume: bool = False,
    benchmark_name: str | None = None,
    data_root: str | os.PathLike[str] | None = None,
    question_types: Sequence[str] | None = None,
    model_tiers: Mapping[str, str] | None = None,
    concurrency: int = 1,
    worker_factory: Callable[[], tuple[Any, Any]] | None = None,
    image_mode: str = "correct",
    control_mapping_path: str | os.PathLike[str] | None = None,
    max_new_samples: int | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency <= 0:
        raise ValueError("concurrency must be a positive integer")
    if concurrency > 1 and worker_factory is None:
        raise ValueError("concurrency greater than 1 requires an independent worker_factory")
    if max_new_samples is not None and (
        isinstance(max_new_samples, bool)
        or not isinstance(max_new_samples, int)
        or max_new_samples <= 0
    ):
        raise ValueError("max_new_samples must be a positive integer")

    selected_samples = _select_samples(benchmark, subsample)
    if not selected_samples:
        raise ValueError("benchmark selection is empty; refusing to report a successful run")
    journal_schema, prescore_metadata_by_object = _preflight_journal_schema(selected_samples)
    if allow_partial and journal_schema != PRESCORE_SCHEMA:
        raise ValueError("partial benchmark execution is valid only for label-free prescore runs")
    if image_mode not in {"correct", "question-only", "shuffled"}:
        raise ValueError(f"unsupported image mode: {image_mode!r}")
    if image_mode != "correct" and journal_schema != PRESCORE_SCHEMA:
        raise ValueError("image ablations are valid only for label-free prescore runs")
    if image_mode == "question-only" and engine != "scenegraph":
        raise ValueError("question-only image mode requires the scenegraph engine")
    if image_mode == "shuffled" and engine != "metric":
        raise ValueError("shuffled image mode requires the metric engine")

    image_root: Path | None = None
    control_lookup: dict[str, dict[str, str]] | None = None
    control_mapping_artifact: dict[str, Any] | None = None
    if image_mode == "shuffled":
        if control_mapping_path is None:
            raise ValueError("shuffled image mode requires a control mapping")
        benchmark_data_path = getattr(benchmark, "data_path", None)
        if benchmark_data_path is None:
            raise ValueError("shuffled image mode requires benchmark.data_path")
        image_root = Path(benchmark_data_path).expanduser().resolve()
        control_lookup, control_mapping_artifact = _load_qspatial_control_mapping(
            control_mapping_path,
            image_root,
        )
    elif control_mapping_path is not None:
        raise ValueError("control mapping is valid only for shuffled image mode")

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    predictions_path = destination / PREDICTIONS_FILENAME
    manifest_path = destination / MANIFEST_FILENAME
    manifest = _build_manifest(
        benchmark,
        selected_samples,
        handler,
        engine=engine,
        benchmark_name=benchmark_name,
        data_root=data_root,
        question_types=question_types,
        model_tiers=model_tiers,
        concurrency=concurrency,
        journal_schema=journal_schema,
        image_mode=image_mode,
        control_mapping_artifact=control_mapping_artifact,
    )

    if resume:
        _validate_manifest(manifest_path, manifest)
        _validate_resume_record_schemas(_read_jsonl_rows(predictions_path), journal_schema)
        _repair_jsonl_tail(predictions_path)
        records = _read_jsonl(predictions_path)
        _validate_resume_record_schemas(_read_jsonl_rows(predictions_path), journal_schema)
    else:
        predictions_path.write_text("", encoding="utf-8")
        _write_json(manifest_path, manifest)
        records = {}

    completed_ids = (
        {sample_id for sample_id, record in records.items() if _record_completed(record)}
        if resume
        else set()
    )
    pending_samples = [
        sample for sample in selected_samples if str(sample.sample_id) not in completed_ids
    ]
    if max_new_samples is not None:
        pending_samples = pending_samples[:max_new_samples]
    if pending_samples:
        workers = [(handler, llm)]
        for _index in range(1, concurrency):
            assert worker_factory is not None
            worker = worker_factory()
            if not isinstance(worker, tuple) or len(worker) != 2:
                raise TypeError("worker_factory must return a (handler, llm) tuple")
            workers.append(worker)

        if concurrency > 1:
            handler_ids = [id(worker_handler) for worker_handler, _worker_llm in workers]
            llm_ids = [id(worker_llm) for _worker_handler, worker_llm in workers]
            if len(set(handler_ids)) != len(handler_ids):
                raise ValueError("worker_factory reused a FieldWorkHandler instance")
            if len(set(llm_ids)) != len(llm_ids):
                raise ValueError("worker_factory reused an LLMClient instance")
            for worker_handler, worker_llm in workers:
                bound_llm = getattr(worker_handler, "llm", worker_llm)
                if bound_llm is not worker_llm:
                    raise ValueError("worker handler is not bound to its paired LLMClient")
            worker_configs = [
                getattr(worker_handler, "config", None) for worker_handler, _worker_llm in workers
            ]
            concrete_configs = [config for config in worker_configs if config is not None]
            if len({id(config) for config in concrete_configs}) != len(concrete_configs):
                raise ValueError("worker_factory reused a Config instance")
            cost_trackers = [
                getattr(worker_llm, "cost_tracker", None) for _worker_handler, worker_llm in workers
            ]
            concrete_trackers = [tracker for tracker in cost_trackers if tracker is not None]
            if len({id(tracker) for tracker in concrete_trackers}) != len(concrete_trackers):
                raise ValueError("worker_factory reused an LLM cost tracker")

        queue: asyncio.Queue[Any] = asyncio.Queue()
        completed_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=len(workers))
        for sample in pending_samples:
            queue.put_nowait(sample)
        for _worker in workers:
            queue.put_nowait(None)

        async def worker_loop(worker_handler: Any, worker_llm: Any) -> None:
            while True:
                item = await queue.get()
                if item is None:
                    return
                sample = item
                try:
                    record = await _run_sample(
                        sample,
                        benchmark,
                        worker_handler,
                        worker_llm,
                        engine,
                        journal_schema=journal_schema,
                        prescore_metadata=prescore_metadata_by_object.get(id(sample)),
                        image_mode=image_mode,
                        image_root=image_root,
                        control_lookup=control_lookup,
                        control_mapping_sha256=(
                            None
                            if control_mapping_artifact is None
                            else str(control_mapping_artifact["mapping_sha256"])
                        ),
                    )
                except Exception as exc:  # pragma: no cover - fatal worker guard
                    await completed_queue.put((sample, None, exc))
                else:
                    await completed_queue.put((sample, record, None))

        worker_tasks = [
            asyncio.create_task(worker_loop(worker_handler, worker_llm))
            for worker_handler, worker_llm in workers
        ]
        try:
            for _completed in pending_samples:
                sample, record, fatal_error = await completed_queue.get()
                if fatal_error is not None:
                    raise fatal_error
                assert record is not None
                if journal_schema == NORMAL_SCHEMA:
                    _score_record(benchmark, sample, record)
                _append_jsonl(predictions_path, record)
                records[str(sample.sample_id)] = record
        finally:
            for task in worker_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*worker_tasks, return_exceptions=True)

    if journal_schema == PRESCORE_SCHEMA:
        if allow_partial:
            completed_samples = [
                sample for sample in selected_samples if str(sample.sample_id) in records
            ]
            results = {
                "total_samples": len(selected_samples),
                "inference_only": True,
                "journal_schema": PRESCORE_SCHEMA,
                "partial": True,
                "operational": _operational_summary(completed_samples, records),
            }
            _write_json(destination / SUMMARY_FILENAME, results)
        else:
            results = _write_inference_only_summary(
                selected_samples,
                records,
                destination,
            )
    else:
        results = _evaluate_selected(
            benchmark,
            selected_samples,
            records,
            destination,
            scoring_module,
        )
    unresolved = [
        str(sample.sample_id)
        for sample in selected_samples
        if not _record_completed(records.get(str(sample.sample_id), {}))
    ]
    if unresolved and not allow_partial:
        raise RuntimeError("benchmark run has unresolved sample errors: " + ", ".join(unresolved))
    return results


async def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    factory_module, scoring_module = _load_spatialclaw_modules(args.spatialclaw_root)
    benchmark = factory_module.BenchmarkFactory.create_benchmark(
        args.benchmark,
        data_root=args.data_root,
        question_type=args.question_types or None,
    )
    if benchmark is None:
        raise ValueError(f"benchmark {args.benchmark!r} resolved to None")

    concurrency = getattr(args, "concurrency", 1)

    def build_worker() -> tuple[Any, Any]:
        config = Config()
        config.fieldwork_engine = args.engine
        config.fieldwork_metric_strict = args.engine == "metric"
        llm = LLMClient(config)
        return FieldWorkHandler(config, llm), llm

    handler, llm = build_worker()
    config = handler.config
    return await run_benchmark(
        benchmark,
        handler,
        llm,
        scoring_module,
        engine=args.engine,
        output_dir=args.output_dir,
        subsample=args.subsample,
        resume=args.resume,
        benchmark_name=args.benchmark,
        data_root=args.data_root,
        question_types=args.question_types,
        model_tiers=getattr(config, "model_tiers", {}),
        concurrency=concurrency,
        worker_factory=build_worker if concurrency > 1 else None,
        image_mode=getattr(args, "image_mode", "correct"),
        control_mapping_path=getattr(args, "control_mapping", None),
    )


def _resolve_litellm_shutdown_boundary() -> tuple[Any, Callable[[], Any]]:
    """Resolve the exact process-global shutdown boundary frozen in uv.lock."""
    try:
        installed_version = package_version("litellm")
    except PackageNotFoundError as exc:
        raise RuntimeError("the benchmark requires the locked LiteLLM distribution") from exc
    if installed_version != EXPECTED_LITELLM_VERSION:
        raise RuntimeError(
            "unsupported LiteLLM shutdown boundary: "
            f"expected {EXPECTED_LITELLM_VERSION}, found {installed_version}"
        )

    try:
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    except ImportError as exc:
        raise RuntimeError("LiteLLM 1.92.0 logging worker API is unavailable") from exc

    flush = getattr(GLOBAL_LOGGING_WORKER, "flush", None)
    stop = getattr(GLOBAL_LOGGING_WORKER, "stop", None)
    close_clients = getattr(litellm, "close_litellm_async_clients", None)
    if not callable(flush) or not callable(stop) or not callable(close_clients):
        raise RuntimeError("LiteLLM 1.92.0 shutdown API is incompatible")
    return GLOBAL_LOGGING_WORKER, close_clients


async def _shutdown_litellm(
    worker: Any,
    close_clients: Callable[[], Any],
    *,
    flush_timeout_seconds: float = LITELLM_FLUSH_TIMEOUT_SECONDS,
) -> None:
    """Attempt every process-global LiteLLM cleanup step and report all failures."""
    errors: list[tuple[str, Exception]] = []
    try:
        await asyncio.wait_for(worker.flush(), timeout=flush_timeout_seconds)
    except Exception as exc:  # noqa: BLE001 - cleanup must continue after any step failure
        errors.append(("flush", exc))
    try:
        await worker.stop()
    except Exception as exc:  # noqa: BLE001 - client closure must still be attempted
        errors.append(("stop", exc))
    try:
        await close_clients()
    except Exception as exc:  # noqa: BLE001 - aggregate after all cleanup attempts
        errors.append(("close_clients", exc))

    if errors:
        summary = ", ".join(f"{step}={type(exc).__name__}" for step, exc in errors)
        error = RuntimeError(f"LiteLLM shutdown failed: {summary}")
        for step, exc in errors:
            error.add_note(f"{step} raised {type(exc).__name__}")
        raise error from errors[0][1]


async def _run_from_args_and_shutdown(args: argparse.Namespace) -> dict[str, Any]:
    """Run one CLI invocation and close LiteLLM within the same event loop.

    The worker API is internal to the exact LiteLLM release frozen in uv.lock.
    Keeping this lifecycle hook in the non-packaged benchmark driver avoids
    extending that internal dependency to normal Spatial Atlas library imports.
    This entry point owns process-global LiteLLM shutdown and must not run beside
    another LiteLLM consumer in the same process.
    """
    worker, close_clients = _resolve_litellm_shutdown_boundary()
    try:
        result = await run_from_args(args)
    except BaseException as primary_error:
        try:
            await _shutdown_litellm(worker, close_clients)
        except Exception as shutdown_error:  # noqa: BLE001 - preserve the primary failure
            shutdown_diagnostic = (
                "LiteLLM shutdown also failed: " f"{type(shutdown_error).__name__}"
            )
            primary_error.add_note(shutdown_diagnostic)
            primary_error._spatial_atlas_shutdown_diagnostic = shutdown_diagnostic
        raise
    await _shutdown_litellm(worker, close_clients)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        asyncio.run(_run_from_args_and_shutdown(args))
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports fatal setup/summary errors
        print(f"[fatal] {type(exc).__name__}: {exc}", file=sys.stderr)
        shutdown_diagnostic = getattr(exc, "_spatial_atlas_shutdown_diagnostic", None)
        if isinstance(shutdown_diagnostic, str):
            print(f"[cleanup] {shutdown_diagnostic}", file=sys.stderr)
        return 1
    print(f"[ok] predictions: {Path(args.output_dir) / PREDICTIONS_FILENAME}")
    print(f"[ok] summary: {Path(args.output_dir) / SUMMARY_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
