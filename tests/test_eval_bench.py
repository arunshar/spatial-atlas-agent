"""Hermetic tests for the SpatialClaw benchmark driver."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fakes import FakeFieldWorkHandler, FakeLLM

import eval_bench as driver
from cost.tracker import CostTracker

pytestmark = pytest.mark.unit


def test_configured_sha256_accepts_only_a_64_hex_digest(monkeypatch):
    name = "ATLAS_TEST_CONFIGURED_DIGEST"
    monkeypatch.delenv(name, raising=False)
    assert driver._configured_sha256(name) == ""
    monkeypatch.setenv(name, "  " + "CD" * 32 + "\n")
    assert driver._configured_sha256(name) == "cd" * 32
    monkeypatch.setenv(name, "cd" * 33)
    assert driver._configured_sha256(name) == ""


@pytest.mark.parametrize(
    ("mapping_digest", "manifest_digest"),
    [("", ""), ("1" * 64, ""), ("", "2" * 64)],
)
def test_control_mapping_fails_closed_without_configured_digests(
    monkeypatch, tmp_path, mapping_digest, manifest_digest
):
    monkeypatch.setattr(driver, "QSPATIAL_CONTROL_MAPPING_SHA256", mapping_digest)
    monkeypatch.setattr(driver, "QSPATIAL_IMAGE_MANIFEST_SHA256", manifest_digest)
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="digests are not configured"):
        driver._load_qspatial_control_mapping(mapping_path, tmp_path)


@dataclass
class FakeSample:
    sample_id: object
    question: str
    question_type: str
    images: list[str]
    answer: str
    answer_type: str = "str"
    output_format: str = ""
    choices: object = None


class LabelFreeSample:
    journal_ground_truth = False
    question_type = "horizontal_distance"
    output_format = (
        r"Return exactly: \scalar{<positive scalar>} "
        r"\distance_unit{centimeter|meter|foot|inch|mm}."
    )

    def __init__(self, sample_id, question, image, *, extra_metadata=None):
        self.sample_id = sample_id
        self.row_id = int(sample_id)
        self.question = question
        self.images = [image]
        self.measurement_family = "horizontal_surface_gap"
        self.cluster_id = f"cluster-{sample_id}"
        self.slice_id = "qspatial_gap97_v1"
        self.slice_sha256 = "7" * 64
        self._extra_metadata = extra_metadata or {}

    @property
    def answer(self):
        raise AssertionError("label-free inference accessed sample.answer")

    def prediction_metadata(self):
        return {
            "measurement_family": self.measurement_family,
            "row_id": self.row_id,
            "cluster_id": self.cluster_id,
            "slice_id": self.slice_id,
            "slice_sha256": self.slice_sha256,
            "output_format": self.output_format,
            **self._extra_metadata,
        }


class FakeBenchmark:
    data_specific_prompt = "Use the benchmark-specific normalization rules."

    def __init__(self, samples):
        self.data = list(samples)
        self.single_calls = []
        self.aggregate_calls = []

    def __iter__(self):
        return iter(self.data)

    def evaluate_single(self, sample, prediction):
        self.single_calls.append((sample.sample_id, prediction))
        return 1.0 if prediction == sample.answer else 0.0

    def evaluate(self, predictions, output_dir=None):
        call = {
            "predictions": dict(predictions),
            "output_dir": output_dir,
            "sample_ids": [sample.sample_id for sample in self.data],
        }
        self.aggregate_calls.append(call)
        scores = [
            float(predictions.get(str(sample.sample_id), "") == sample.answer)
            for sample in self.data
        ]
        return {
            "total_samples": len(self.data),
            "overall_accuracy": sum(scores) / len(scores) if scores else 0.0,
            "detailed_results": [{"score": score} for score in scores],
        }


class FakeScoring(ModuleType):
    def __init__(self):
        super().__init__("fake_scoring")
        self.calls = []

    def write_results_summary(self, output_dir, results):
        self.calls.append((output_dir, results))
        summary = {key: value for key, value in results.items() if key != "detailed_results"}
        path = Path(output_dir) / driver.SUMMARY_FILENAME
        path.write_text(json.dumps(summary), encoding="utf-8")


def _image(tmp_path: Path, png_bytes: bytes, name: str) -> str:
    path = tmp_path / name
    path.write_bytes(png_bytes)
    return str(path)


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _seed_manifest(output, benchmark, handler, engine):
    manifest = driver._build_manifest(
        benchmark,
        list(benchmark.data),
        handler,
        engine=engine,
        benchmark_name=None,
        data_root=None,
        question_types=None,
        model_tiers=None,
    )
    (output / driver.MANIFEST_FILENAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def test_parse_args_preserves_cli_selection_and_filter_order(monkeypatch, tmp_path):
    monkeypatch.setenv("SPATIALCLAW_ROOT", "/fake/spatialclaw")
    args = driver.parse_args(
        [
            "--benchmark",
            "warehouse",
            "--data-root",
            "/datasets",
            "--engine",
            "metric",
            "--question-type",
            "distance,count",
            "--question-type",
            "mcq",
            "--subsample",
            "5",
            "--output-dir",
            str(tmp_path),
            "--resume",
        ]
    )

    assert args.benchmark == "warehouse"
    assert args.data_root == "/datasets"
    assert args.engine == "metric"
    assert args.question_types == ["distance", "count", "mcq"]
    assert args.subsample == 5
    assert args.resume is True
    assert args.concurrency == 1
    assert args.spatialclaw_root == "/fake/spatialclaw"


def test_parse_args_requires_spatialclaw_root(monkeypatch, tmp_path):
    monkeypatch.delenv("SPATIALCLAW_ROOT", raising=False)
    with pytest.raises(SystemExit):
        driver.parse_args(
            [
                "--benchmark",
                "omni3d",
                "--engine",
                "scenegraph",
                "--output-dir",
                str(tmp_path),
            ]
        )


def test_parse_args_rejects_nonpositive_subsample(monkeypatch, tmp_path):
    monkeypatch.setenv("SPATIALCLAW_ROOT", "/fake/spatialclaw")
    with pytest.raises(SystemExit):
        driver.parse_args(
            [
                "--benchmark",
                "omni3d",
                "--engine",
                "scenegraph",
                "--subsample",
                "0",
                "--output-dir",
                str(tmp_path),
            ]
        )


def test_parse_args_accepts_positive_concurrency_and_rejects_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("SPATIALCLAW_ROOT", "/fake/spatialclaw")
    base = [
        "--benchmark",
        "omni3d",
        "--engine",
        "scenegraph",
        "--output-dir",
        str(tmp_path),
        "--concurrency",
    ]
    assert driver.parse_args([*base, "4"]).concurrency == 4
    with pytest.raises(SystemExit):
        driver.parse_args([*base, "0"])


def test_dynamic_import_prepends_checkout_and_loads_factory_and_scorer(monkeypatch, tmp_path):
    checkout = tmp_path / "SpatialClaw"
    evals = checkout / "spatial_agent" / "evals"
    evals.mkdir(parents=True)
    (evals / "factory.py").write_text("", encoding="utf-8")
    (evals / "scoring.py").write_text("", encoding="utf-8")
    factory = ModuleType("factory")
    scoring = ModuleType("scoring")
    imported = []

    def fake_import(name):
        imported.append(name)
        return factory if name.endswith("factory") else scoring

    monkeypatch.setattr(driver.importlib, "import_module", fake_import)
    checkout_str = str(checkout.resolve())
    try:
        loaded = driver._load_spatialclaw_modules(checkout)
        assert loaded == (factory, scoring)
        assert sys.path[0] == checkout_str
        assert imported == ["spatial_agent.evals.factory", "spatial_agent.evals.scoring"]
    finally:
        while checkout_str in sys.path:
            sys.path.remove(checkout_str)


def test_dynamic_import_rejects_incomplete_checkout(tmp_path):
    with pytest.raises(FileNotFoundError, match="factory.py"):
        driver._load_spatialclaw_modules(tmp_path)


def test_subsample_matches_spatialclaw_seed_42_shuffle_order():
    benchmark = FakeBenchmark(
        [FakeSample(i, f"q{i}", "distance", ["unused"], str(i)) for i in range(10)]
    )
    expected = list(benchmark.data)
    random.Random(42).shuffle(expected)

    selected_a = driver._select_samples(benchmark, 4)
    selected_b = driver._select_samples(benchmark, 4)

    assert [sample.sample_id for sample in selected_a] == [
        sample.sample_id for sample in expected[:4]
    ]
    assert [sample.sample_id for sample in selected_b] == [
        sample.sample_id for sample in expected[:4]
    ]
    assert driver._select_samples(benchmark, None) == benchmark.data
    assert driver._select_samples(benchmark, 99) == expected


@pytest.mark.asyncio
async def test_run_writes_schema_usage_score_and_benchmark_summary(tmp_path, png_bytes):
    image = _image(tmp_path, png_bytes, "sample.png")
    sample = FakeSample(7, "distance question", "distance", [image], "2.5", "float")
    benchmark = FakeBenchmark([sample])
    llm = FakeLLM()
    handler = FakeFieldWorkHandler(llm=llm, answers={"distance question": "2.5"})
    scoring = FakeScoring()
    output = tmp_path / "run"

    results = await driver.run_benchmark(
        benchmark,
        handler,
        llm,
        scoring,
        engine="metric",
        output_dir=output,
    )

    rows = _records(output / driver.PREDICTIONS_FILENAME)
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == {
        "sample_id",
        "answer",
        "gt",
        "score",
        "usage",
        "llm_call_telemetry",
        "latency_seconds",
        "engine",
        "requested_engine",
        "used_engine",
        "grounding_source",
        "question_type",
        "image_path",
        "error",
    }
    assert row["sample_id"] == 7
    assert row["answer"] == row["gt"] == "2.5"
    assert row["score"] == 1.0
    assert row["engine"] == row["requested_engine"] == row["used_engine"] == "metric"
    assert row["grounding_source"] == "sam3+da3"
    assert row["question_type"] == "distance"
    assert row["image_path"] == image
    assert row["error"] is None
    assert row["latency_seconds"] >= 0
    assert row["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
        "num_calls": 1,
        "estimated_cost_usd": 0.01,
    }
    assert row["llm_call_telemetry"] is None

    goal, file_parts = handler.calls[0]
    assert "# Question\ndistance question" in goal
    assert FakeBenchmark.data_specific_prompt in goal
    assert "# Output Format\nnumber" in goal
    assert file_parts[0][0:2] == ("sample.png", "image/png")
    assert file_parts[0][2] == png_bytes
    assert benchmark.single_calls == [(7, "2.5")]
    assert benchmark.aggregate_calls[0]["predictions"] == {"7": "2.5"}
    assert benchmark.aggregate_calls[0]["sample_ids"] == [7]
    assert benchmark.data == [sample]
    assert scoring.calls == [(str(output), results)]
    summary = json.loads((output / driver.SUMMARY_FILENAME).read_text())
    assert summary["total_samples"] == 1
    assert summary["overall_accuracy"] == 1.0
    assert summary["operational"]["completed_count"] == 1
    assert summary["operational"]["error_count"] == 0
    overall_ops = summary["operational"]["overall"]
    assert overall_ops["mean_latency_seconds"] >= 0
    assert overall_ops["mean_prompt_tokens"] == 10.0
    assert overall_ops["mean_completion_tokens"] == 2.0
    assert overall_ops["mean_total_tokens"] == 12.0
    assert overall_ops["mean_calls"] == 1.0
    assert overall_ops["mean_estimated_cost_usd"] == 0.01
    assert (
        summary["operational"]["per_question_type"]["distance"]["mean_estimated_cost_usd"] == 0.01
    )
    manifest = json.loads((output / driver.MANIFEST_FILENAME).read_text())
    assert manifest["engine"] == "metric"
    assert manifest["concurrency"] == 1
    assert manifest["seed"] == 42
    assert manifest["selected_ids"] == ["7"]


@pytest.mark.asyncio
async def test_sample_failure_is_persisted_and_later_samples_continue(tmp_path, png_bytes):
    samples = [
        FakeSample(
            index,
            question,
            "distance",
            [_image(tmp_path, png_bytes, f"{index}.png")],
            answer,
            "float",
        )
        for index, question, answer in (
            (1, "first", "1"),
            (2, "bad sample", "2"),
            (3, "third", "3"),
        )
    ]
    benchmark = FakeBenchmark(samples)
    llm = FakeLLM()
    handler = FakeFieldWorkHandler(
        llm=llm,
        answers={"first": "1", "third": "3"},
        fail_on={"bad sample"},
    )
    scoring = FakeScoring()
    output = tmp_path / "errors"

    with pytest.raises(RuntimeError, match="unresolved sample errors: 2"):
        await driver.run_benchmark(
            benchmark,
            handler,
            llm,
            scoring,
            engine="metric",
            output_dir=output,
        )

    rows = _records(output / driver.PREDICTIONS_FILENAME)
    assert [row["sample_id"] for row in rows] == [1, 2, 3]
    assert rows[0]["answer"] == "1"
    assert rows[1]["answer"] is None
    assert rows[1]["score"] is None
    assert rows[1]["error"] == "RuntimeError: forced failure: bad sample"
    assert rows[1]["used_engine"] is None
    assert rows[2]["answer"] == "3"
    assert benchmark.aggregate_calls[0]["predictions"] == {"1": "1", "2": "", "3": "3"}
    summary = json.loads((output / driver.SUMMARY_FILENAME).read_text())
    assert summary["operational"]["completed_count"] == 2
    assert summary["operational"]["error_count"] == 1


@pytest.mark.asyncio
async def test_concurrent_workers_overlap_with_isolated_state_and_ordered_output(
    tmp_path, png_bytes
):
    samples = [
        FakeSample(
            index,
            f"question {index}",
            "distance",
            [_image(tmp_path, png_bytes, f"concurrent-{index}.png")],
            str(index),
            "float",
        )
        for index in range(1, 6)
    ]
    benchmark = FakeBenchmark(samples)
    state = {
        "active": 0,
        "max_active": 0,
        "active_handlers": set(),
        "bindings": [],
    }
    handlers = []
    llms = []
    configs = []

    class DelayedHandler:
        def __init__(self, config, llm, worker_index):
            self.config = config
            self.llm = llm
            self.worker_index = worker_index
            self.last_fieldwork_engine = None
            self.last_metric_grounding = None

        async def handle(self, text, _file_parts, _updater, **_kwargs):
            handler_id = id(self)
            assert handler_id not in state["active_handlers"]
            state["active_handlers"].add(handler_id)
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            state["bindings"].append((handler_id, id(self.llm), id(self.config)))
            try:
                question = next(index for index in range(1, 6) if f"question {index}" in text)
                await asyncio.sleep(0.04 if question == 1 else 0.005)
                stats = self.llm.cost_tracker.stats
                stats.prompt_tokens += 10 + self.worker_index
                stats.completion_tokens += 2
                stats.total_tokens += 12 + self.worker_index
                stats.num_calls += 1
                stats.estimated_cost_usd += 0.01
                self.last_fieldwork_engine = "scenegraph"
                return str(question)
            finally:
                state["active"] -= 1
                state["active_handlers"].remove(handler_id)

    def build_worker():
        worker_index = len(handlers)
        config = SimpleNamespace(
            fieldwork_engine="scenegraph",
            fieldwork_metric_strict=False,
            model_tiers={"fast": "fake/model"},
            llm_enable_thinking=None,
        )
        llm = FakeLLM(config=config)
        handler = DelayedHandler(config, llm, worker_index)
        configs.append(config)
        llms.append(llm)
        handlers.append(handler)
        return handler, llm

    handler, llm = build_worker()
    output = tmp_path / "concurrent"
    await driver.run_benchmark(
        benchmark,
        handler,
        llm,
        FakeScoring(),
        engine="scenegraph",
        output_dir=output,
        concurrency=3,
        worker_factory=build_worker,
    )

    rows = _records(output / driver.PREDICTIONS_FILENAME)
    assert {row["sample_id"] for row in rows} == set(range(1, 6))
    assert rows[0]["sample_id"] != 1
    assert {sample_id for sample_id, _answer in benchmark.single_calls} == set(range(1, 6))
    assert benchmark.aggregate_calls[0]["sample_ids"] == [1, 2, 3, 4, 5]
    assert list(benchmark.aggregate_calls[0]["predictions"]) == ["1", "2", "3", "4", "5"]
    assert state["max_active"] > 1
    assert len(handlers) == len({id(value) for value in handlers}) == 3
    assert len(llms) == len({id(value) for value in llms}) == 3
    assert len(configs) == len({id(value) for value in configs}) == 3
    assert all(handler.llm is llm for handler, llm in zip(handlers, llms, strict=True))
    assert all(row["usage"]["num_calls"] == 1 for row in rows)
    assert all(
        row["usage"]["total_tokens"]
        == row["usage"]["prompt_tokens"] + row["usage"]["completion_tokens"]
        for row in rows
    )
    manifest = json.loads((output / driver.MANIFEST_FILENAME).read_text())
    assert manifest["concurrency"] == 3


@pytest.mark.asyncio
async def test_completed_rows_are_journaled_before_an_earlier_sample_finishes(tmp_path, png_bytes):
    samples = [
        FakeSample(
            index,
            f"durability {index}",
            "distance",
            [_image(tmp_path, png_bytes, f"durability-{index}.png")],
            str(index),
            "float",
        )
        for index in (1, 2)
    ]
    benchmark = FakeBenchmark(samples)
    slow_release = asyncio.Event()
    fast_finished = asyncio.Event()

    class DurabilityHandler:
        def __init__(self, config, llm):
            self.config = config
            self.llm = llm
            self.last_fieldwork_engine = None
            self.last_metric_grounding = None

        async def handle(self, text, _file_parts, _updater, **_kwargs):
            if "durability 1" in text:
                await slow_release.wait()
                answer = "1"
            else:
                answer = "2"
                fast_finished.set()
            self.last_fieldwork_engine = "scenegraph"
            return answer

    def build_worker():
        config = SimpleNamespace(
            fieldwork_engine="scenegraph",
            fieldwork_metric_strict=False,
            model_tiers={"fast": "fake/model"},
            llm_enable_thinking=None,
        )
        llm = FakeLLM(config=config)
        return DurabilityHandler(config, llm), llm

    handler, llm = build_worker()
    output = tmp_path / "durability"
    task = asyncio.create_task(
        driver.run_benchmark(
            benchmark,
            handler,
            llm,
            FakeScoring(),
            engine="scenegraph",
            output_dir=output,
            concurrency=2,
            worker_factory=build_worker,
        )
    )
    await asyncio.wait_for(fast_finished.wait(), timeout=1)
    prediction_path = output / driver.PREDICTIONS_FILENAME
    for _attempt in range(100):
        if prediction_path.is_file() and prediction_path.stat().st_size:
            break
        await asyncio.sleep(0)
    assert [row["sample_id"] for row in _records(prediction_path)] == [2]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_empty_handler_answer_is_explicit_unresolved_error(tmp_path, png_bytes):
    image = _image(tmp_path, png_bytes, "empty-answer.png")
    sample = FakeSample(1, "empty answer", "count", [image], "2", "int")
    benchmark = FakeBenchmark([sample])
    llm = FakeLLM()
    output = tmp_path / "empty-answer"

    with pytest.raises(RuntimeError, match="unresolved sample errors: 1"):
        await driver.run_benchmark(
            benchmark,
            FakeFieldWorkHandler(llm=llm, answers={"empty answer": "   "}),
            llm,
            FakeScoring(),
            engine="scenegraph",
            output_dir=output,
        )

    row = _records(output / driver.PREDICTIONS_FILENAME)[0]
    assert row["answer"] == "   "
    assert row["error"] == "RuntimeError: handler returned an empty answer"
    summary = json.loads((output / driver.SUMMARY_FILENAME).read_text())
    assert summary["operational"]["completed_count"] == 0
    assert summary["operational"]["error_count"] == 1


@pytest.mark.asyncio
async def test_resume_skips_successful_persisted_sample_id(tmp_path, png_bytes):
    samples = [
        FakeSample(
            index,
            f"question {index}",
            "count",
            [_image(tmp_path, png_bytes, f"resume-{index}.png")],
            str(index),
            "int",
        )
        for index in (1, 2)
    ]
    output = tmp_path / "resume"
    output.mkdir()
    prior = {
        "sample_id": 1,
        "answer": "1",
        "gt": "1",
        "score": 1.0,
        "engine": "scenegraph",
        "requested_engine": "scenegraph",
        "error": None,
    }
    (output / driver.PREDICTIONS_FILENAME).write_text(
        json.dumps(prior) + "\n",
        encoding="utf-8",
    )
    benchmark = FakeBenchmark(samples)
    llm = FakeLLM()
    handler = FakeFieldWorkHandler(llm=llm, answers={"question 2": "2"})
    _seed_manifest(output, benchmark, handler, "scenegraph")

    await driver.run_benchmark(
        benchmark,
        handler,
        llm,
        FakeScoring(),
        engine="scenegraph",
        output_dir=output,
        resume=True,
    )

    rows = _records(output / driver.PREDICTIONS_FILENAME)
    assert [row["sample_id"] for row in rows] == [1, 2]
    assert len(handler.calls) == 1
    assert "question 2" in handler.calls[0][0]
    assert benchmark.aggregate_calls[0]["predictions"] == {"1": "1", "2": "2"}


@pytest.mark.asyncio
async def test_resume_retries_persisted_error_row(tmp_path, png_bytes):
    image = _image(tmp_path, png_bytes, "retry-error.png")
    sample = FakeSample(1, "retry question", "count", [image], "2", "int")
    output = tmp_path / "retry-error"
    output.mkdir()
    prior = {
        "sample_id": 1,
        "answer": None,
        "gt": "2",
        "score": None,
        "engine": "scenegraph",
        "requested_engine": "scenegraph",
        "error": "RuntimeError: transient service failure",
    }
    prediction_path = output / driver.PREDICTIONS_FILENAME
    prediction_path.write_text(json.dumps(prior) + "\n", encoding="utf-8")
    benchmark = FakeBenchmark([sample])
    llm = FakeLLM()
    handler = FakeFieldWorkHandler(llm=llm, answers={"retry question": "2"})
    _seed_manifest(output, benchmark, handler, "scenegraph")

    await driver.run_benchmark(
        benchmark,
        handler,
        llm,
        FakeScoring(),
        engine="scenegraph",
        output_dir=output,
        resume=True,
    )

    rows = _records(prediction_path)
    assert len(rows) == 2
    assert rows[0]["error"] == "RuntimeError: transient service failure"
    assert rows[1]["answer"] == "2"
    assert rows[1]["error"] is None
    assert len(handler.calls) == 1
    assert benchmark.aggregate_calls[0]["predictions"] == {"1": "2"}


@pytest.mark.asyncio
async def test_resume_retries_legacy_blank_success_row(tmp_path, png_bytes):
    image = _image(tmp_path, png_bytes, "retry-blank.png")
    sample = FakeSample(1, "retry blank", "count", [image], "2", "int")
    output = tmp_path / "retry-blank"
    output.mkdir()
    prior = {
        "sample_id": 1,
        "answer": " ",
        "gt": "2",
        "score": 0.0,
        "engine": "scenegraph",
        "requested_engine": "scenegraph",
        "error": None,
    }
    prediction_path = output / driver.PREDICTIONS_FILENAME
    prediction_path.write_text(json.dumps(prior) + "\n", encoding="utf-8")
    benchmark = FakeBenchmark([sample])
    llm = FakeLLM()
    handler = FakeFieldWorkHandler(llm=llm, answers={"retry blank": "2"})
    _seed_manifest(output, benchmark, handler, "scenegraph")

    await driver.run_benchmark(
        benchmark,
        handler,
        llm,
        FakeScoring(),
        engine="scenegraph",
        output_dir=output,
        resume=True,
    )

    rows = _records(prediction_path)
    assert len(rows) == 2
    assert rows[-1]["answer"] == "2"
    assert rows[-1]["error"] is None


@pytest.mark.asyncio
async def test_resume_rejects_cross_engine_rows(tmp_path, png_bytes):
    image = _image(tmp_path, png_bytes, "engine-mismatch.png")
    sample = FakeSample(1, "q", "distance", [image], "1", "float")
    output = tmp_path / "engine-mismatch"
    output.mkdir()
    (output / driver.PREDICTIONS_FILENAME).write_text(
        json.dumps({"sample_id": 1, "answer": "1", "engine": "scenegraph"}) + "\n",
        encoding="utf-8",
    )

    benchmark = FakeBenchmark([sample])
    stored_handler = FakeFieldWorkHandler()
    _seed_manifest(output, benchmark, stored_handler, "scenegraph")

    with pytest.raises(ValueError, match="resume manifest mismatch.*engine"):
        await driver.run_benchmark(
            benchmark,
            FakeFieldWorkHandler(),
            object(),
            FakeScoring(),
            engine="metric",
            output_dir=output,
            resume=True,
        )


@pytest.mark.asyncio
async def test_resume_rejects_changed_concurrency(tmp_path, png_bytes):
    image = _image(tmp_path, png_bytes, "concurrency-mismatch.png")
    sample = FakeSample(1, "q", "distance", [image], "1", "float")
    output = tmp_path / "concurrency-mismatch"
    output.mkdir()
    benchmark = FakeBenchmark([sample])
    stored_handler = FakeFieldWorkHandler()
    _seed_manifest(output, benchmark, stored_handler, "scenegraph")
    llm = FakeLLM()
    requested_handler = FakeFieldWorkHandler(llm=llm)

    with pytest.raises(ValueError, match="resume manifest mismatch.*concurrency"):
        await driver.run_benchmark(
            benchmark,
            requested_handler,
            llm,
            FakeScoring(),
            engine="scenegraph",
            output_dir=output,
            resume=True,
            concurrency=2,
            worker_factory=lambda: (FakeFieldWorkHandler(), FakeLLM()),
        )


@pytest.mark.asyncio
async def test_resume_requires_manifest(tmp_path, png_bytes):
    image = _image(tmp_path, png_bytes, "missing-manifest.png")
    sample = FakeSample(1, "q", "count", [image], "1", "int")
    output = tmp_path / "missing-manifest"
    output.mkdir()
    (output / driver.PREDICTIONS_FILENAME).write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="resume requires run_manifest.json"):
        await driver.run_benchmark(
            FakeBenchmark([sample]),
            FakeFieldWorkHandler(),
            object(),
            FakeScoring(),
            engine="scenegraph",
            output_dir=output,
            resume=True,
        )


@pytest.mark.asyncio
async def test_resume_repairs_partial_tail_before_append(tmp_path, png_bytes):
    samples = [
        FakeSample(
            index,
            f"question {index}",
            "count",
            [_image(tmp_path, png_bytes, f"tail-{index}.png")],
            str(index),
            "int",
        )
        for index in (1, 2)
    ]
    benchmark = FakeBenchmark(samples)
    llm = FakeLLM()
    handler = FakeFieldWorkHandler(llm=llm, answers={"question 2": "2"})
    output = tmp_path / "tail-repair"
    output.mkdir()
    prior = {
        "sample_id": 1,
        "answer": "1",
        "gt": "1",
        "score": 1.0,
        "engine": "scenegraph",
        "requested_engine": "scenegraph",
        "error": None,
    }
    path = output / driver.PREDICTIONS_FILENAME
    path.write_bytes((json.dumps(prior) + "\n").encode() + b'{"sample_id":')
    _seed_manifest(output, benchmark, handler, "scenegraph")

    await driver.run_benchmark(
        benchmark,
        handler,
        llm,
        FakeScoring(),
        engine="scenegraph",
        output_dir=output,
        resume=True,
    )

    assert [row["sample_id"] for row in _records(path)] == [1, 2]


@pytest.mark.asyncio
async def test_nonresume_truncates_stale_predictions(tmp_path, png_bytes):
    output = tmp_path / "fresh"
    output.mkdir()
    prediction_path = output / driver.PREDICTIONS_FILENAME
    prediction_path.write_text('{"sample_id":"stale","answer":"x"}\n', encoding="utf-8")
    sample = FakeSample(
        "new", "new question", "count", [_image(tmp_path, png_bytes, "new.png")], "4", "int"
    )
    benchmark = FakeBenchmark([sample])
    llm = FakeLLM()

    await driver.run_benchmark(
        benchmark,
        FakeFieldWorkHandler(llm=llm, answers={"new question": "4"}),
        llm,
        FakeScoring(),
        engine="scenegraph",
        output_dir=output,
        resume=False,
    )

    assert [row["sample_id"] for row in _records(prediction_path)] == ["new"]


@pytest.mark.asyncio
async def test_warehouse_lazy_sample_hook_runs_before_image_read(tmp_path, png_bytes):
    overlay = _image(tmp_path, png_bytes, "warehouse-overlay.png")
    raw = tmp_path / "warehouse-raw.jpg"
    raw.write_bytes(b"raw-source-bytes")
    sample = FakeSample("w1", "warehouse distance", "distance", [], "6.36", "float")
    sample.source_image = str(raw)
    sample.rles = [{"size": [2, 2], "counts": "encoded"}]
    hook_calls = []

    def ensure_frames_loaded():
        hook_calls.append(True)
        sample.images = [overlay]

    sample.ensure_frames_loaded = ensure_frames_loaded
    benchmark = FakeBenchmark([sample])
    llm = FakeLLM()
    handler = FakeFieldWorkHandler(llm=llm, answers={"warehouse distance": "6.36"})

    await driver.run_benchmark(
        benchmark,
        handler,
        llm,
        FakeScoring(),
        engine="metric",
        output_dir=tmp_path / "warehouse-run",
    )

    assert hook_calls == [True]
    assert benchmark.single_calls == [("w1", "6.36")]
    assert handler.calls[0][1][0][2] == png_bytes
    assert handler.handle_kwargs[0] == {
        "metric_image_bytes": b"raw-source-bytes",
        "metric_regions": sample.rles,
    }
    row = _records(tmp_path / "warehouse-run" / driver.PREDICTIONS_FILENAME)[0]
    assert row["grounding_source"] == "rle+da3"


@pytest.mark.asyncio
async def test_multiple_images_becomes_persisted_error(tmp_path, png_bytes):
    images = [
        _image(tmp_path, png_bytes, "multi-a.png"),
        _image(tmp_path, png_bytes, "multi-b.png"),
    ]
    sample = FakeSample("multi", "q", "distance", images, "1", "float")
    benchmark = FakeBenchmark([sample])
    output = tmp_path / "multi-run"

    with pytest.raises(RuntimeError, match="unresolved sample errors: multi"):
        await driver.run_benchmark(
            benchmark,
            FakeFieldWorkHandler(),
            object(),
            FakeScoring(),
            engine="metric",
            output_dir=output,
        )

    row = _records(output / driver.PREDICTIONS_FILENAME)[0]
    assert row["score"] is None
    assert row["usage"] is None
    assert "must resolve to exactly one image" in row["error"]


@pytest.mark.parametrize("engine", ["scenegraph", "metric"])
@pytest.mark.asyncio
async def test_run_from_args_wires_factory_filters_and_strict_engine(
    monkeypatch, tmp_path, png_bytes, engine
):
    image = _image(tmp_path, png_bytes, f"{engine}.png")
    sample = FakeSample(1, "q", "distance", [image], "0", "float")
    benchmark = FakeBenchmark([sample])
    factory_calls = []

    class Factory:
        @staticmethod
        def create_benchmark(name, **kwargs):
            factory_calls.append((name, kwargs))
            return benchmark

    factory_module = SimpleNamespace(BenchmarkFactory=Factory)
    scoring = FakeScoring()
    loaded_roots = []

    def load_modules(root):
        loaded_roots.append(root)
        return factory_module, scoring

    class FakeConfig:
        pass

    handlers = []

    def handler_factory(config, llm):
        handler = FakeFieldWorkHandler(config=config, llm=llm)
        handlers.append(handler)
        return handler

    monkeypatch.setattr(driver, "_load_spatialclaw_modules", load_modules)
    monkeypatch.setattr(driver, "Config", FakeConfig)
    monkeypatch.setattr(driver, "LLMClient", FakeLLM)
    monkeypatch.setattr(driver, "FieldWorkHandler", handler_factory)
    args = argparse.Namespace(
        spatialclaw_root="/checkout",
        benchmark="warehouse",
        data_root="/datasets",
        question_types=["distance"],
        engine=engine,
        output_dir=str(tmp_path / f"args-{engine}"),
        subsample=None,
        resume=False,
        concurrency=2,
    )

    await driver.run_from_args(args)

    assert loaded_roots == ["/checkout"]
    assert factory_calls == [
        (
            "warehouse",
            {"data_root": "/datasets", "question_type": ["distance"]},
        )
    ]
    assert len(handlers) == 2
    assert len({id(handler) for handler in handlers}) == 2
    assert len({id(handler.llm) for handler in handlers}) == 2
    assert len({id(handler.config) for handler in handlers}) == 2
    assert all(handler.config.fieldwork_engine == engine for handler in handlers)
    assert all(
        handler.config.fieldwork_metric_strict is (engine == "metric") for handler in handlers
    )
    manifest = json.loads((tmp_path / f"args-{engine}" / driver.MANIFEST_FILENAME).read_text())
    assert manifest["concurrency"] == 2


@pytest.mark.asyncio
async def test_run_from_args_rejects_none_benchmark(monkeypatch, tmp_path):
    class Factory:
        @staticmethod
        def create_benchmark(*_args, **_kwargs):
            return None

    monkeypatch.setattr(
        driver,
        "_load_spatialclaw_modules",
        lambda _root: (SimpleNamespace(BenchmarkFactory=Factory), FakeScoring()),
    )
    args = argparse.Namespace(
        spatialclaw_root="/checkout",
        benchmark="none",
        data_root="data",
        question_types=[],
        engine="scenegraph",
        output_dir=str(tmp_path),
        subsample=None,
        resume=False,
    )

    with pytest.raises(ValueError, match="resolved to None"):
        await driver.run_from_args(args)


def test_output_format_and_json_helpers(tmp_path):
    explicit = FakeSample(1, "q", "x", [], "a", output_format="json")
    numeric = FakeSample(2, "q", "x", [], "1.2", answer_type="float")
    text = FakeSample(3, "q", "left_right", [], "left")

    assert driver._infer_output_format(explicit) == "json"
    assert driver._infer_output_format(numeric) == "number"
    assert driver._infer_output_format(text) == "text"
    assert driver._json_default(tmp_path) == str(tmp_path)
    assert driver._json_default(SimpleNamespace(item=lambda: 9)) == 9
    assert driver._json_default(object()).startswith("<object object at")
    assert driver._usage_snapshot(object()) is None
    assert driver._usage_delta(None, None) is None


def test_benchmark_instruction_matches_native_prompt_and_choices():
    mapping = FakeSample(
        1,
        "Which region?",
        "mcq",
        [],
        "B",
        choices={"A": "Region 0", "B": "Region 1"},
    )
    sequence = FakeSample(
        2,
        "Choose one",
        "mcq",
        [],
        "1",
        choices=["left", "right"],
    )

    assert driver._benchmark_instruction(mapping, "Normalize.") == (
        "Which region?\n\nNormalize.\nA. Region 0\nB. Region 1"
    )
    assert driver._benchmark_instruction(sequence, "Normalize.") == (
        "Choose one\n\nNormalize.\nA. left\nB. right"
    )


def test_manifest_captures_complete_run_identity(tmp_path):
    benchmark = FakeBenchmark(
        [
            FakeSample("a", "q", "distance", [], "1"),
            FakeSample("b", "q", "count", [], "2"),
        ]
    )
    handler = FakeFieldWorkHandler()
    manifest = driver._build_manifest(
        benchmark,
        benchmark.data,
        handler,
        engine="metric",
        benchmark_name="warehouse",
        data_root=tmp_path / "dataset",
        question_types=["distance", "count"],
        model_tiers={"fast": "provider/fast", "vision": "provider/vision"},
    )

    assert manifest == {
        "benchmark": "warehouse",
        "data_root": str((tmp_path / "dataset").resolve()),
        "question_filters": ["distance", "count"],
        "engine": "metric",
        "model_tiers": {"fast": "provider/fast", "vision": "provider/vision"},
        "llm_enable_thinking": None,
        "concurrency": 1,
        "journal_schema": driver.NORMAL_SCHEMA,
        "image_mode": "correct",
        "control_mapping_artifact": None,
        "seed": 42,
        "selected_ids": ["a", "b"],
    }


def test_manifest_records_resolved_disabled_thinking_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("ATLAS_ENABLE_THINKING", "false")
    config = driver.Config()
    benchmark = FakeBenchmark([FakeSample("a", "q", "count", [], "1")])

    manifest = driver._build_manifest(
        benchmark,
        benchmark.data,
        FakeFieldWorkHandler(config=config),
        engine="scenegraph",
        benchmark_name="omni3d",
        data_root=tmp_path,
        question_types=None,
        model_tiers=None,
    )

    assert config.llm_enable_thinking is False
    assert manifest["llm_enable_thinking"] is False


@pytest.mark.asyncio
async def test_resume_rejects_changed_thinking_mode(monkeypatch, tmp_path, png_bytes):
    sample = FakeSample(
        "a",
        "q",
        "count",
        [_image(tmp_path, png_bytes, "thinking-resume.png")],
        "1",
    )
    benchmark = FakeBenchmark([sample])
    output = tmp_path / "thinking-resume"
    output.mkdir()

    monkeypatch.setenv("ATLAS_ENABLE_THINKING", "false")
    stored_handler = FakeFieldWorkHandler(config=driver.Config())
    _seed_manifest(output, benchmark, stored_handler, "scenegraph")

    monkeypatch.setenv("ATLAS_ENABLE_THINKING", "true")
    requested_handler = FakeFieldWorkHandler(config=driver.Config())
    with pytest.raises(
        ValueError,
        match="resume manifest mismatch.*llm_enable_thinking",
    ):
        await driver.run_benchmark(
            benchmark,
            requested_handler,
            FakeLLM(config=requested_handler.config),
            FakeScoring(),
            engine="scenegraph",
            output_dir=output,
            resume=True,
        )


def test_repair_jsonl_tail_truncates_partial_fragment(tmp_path):
    path = tmp_path / "predictions.jsonl"
    complete = json.dumps({"sample_id": 1}) + "\n"
    path.write_bytes(complete.encode() + b'{"sample_id":')

    assert driver._repair_jsonl_tail(path) is True
    assert path.read_text() == complete


def test_repair_jsonl_tail_terminates_valid_final_record(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps({"sample_id": 1}), encoding="utf-8")

    assert driver._repair_jsonl_tail(path) is True
    assert path.read_bytes().endswith(b"\n")


@pytest.mark.asyncio
async def test_empty_selection_fails_before_creating_output(tmp_path):
    output = tmp_path / "empty-run"
    with pytest.raises(ValueError, match="selection is empty"):
        await driver.run_benchmark(
            FakeBenchmark([]),
            FakeFieldWorkHandler(),
            object(),
            FakeScoring(),
            engine="scenegraph",
            output_dir=output,
        )
    assert not output.exists()


def test_read_jsonl_ignores_malformed_and_missing_ids(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        "not-json\n" + json.dumps({"answer": "x"}) + "\n" + json.dumps({"sample_id": 4}),
        encoding="utf-8",
    )
    assert driver._read_jsonl(path) == {"4": {"sample_id": 4}}
    assert driver._read_jsonl(tmp_path / "missing.jsonl") == {}


def test_main_reports_success_and_fatal_error(monkeypatch, tmp_path, capsys):
    args = argparse.Namespace(output_dir=str(tmp_path))
    monkeypatch.setattr(driver, "parse_args", lambda _argv: args)
    shutdown_calls = []
    worker = object()

    async def close_clients():
        return None

    monkeypatch.setattr(
        driver,
        "_resolve_litellm_shutdown_boundary",
        lambda: (worker, close_clients),
    )

    async def shutdown(actual_worker, actual_close_clients):
        assert actual_worker is worker
        assert actual_close_clients is close_clients
        shutdown_calls.append("shutdown")

    monkeypatch.setattr(driver, "_shutdown_litellm", shutdown)

    async def success(_args):
        return {"ok": True}

    monkeypatch.setattr(driver, "run_from_args", success)
    assert driver.main([]) == 0
    assert "predictions.jsonl" in capsys.readouterr().out
    assert shutdown_calls == ["shutdown"]

    async def failure(_args):
        raise RuntimeError("boom")

    monkeypatch.setattr(driver, "run_from_args", failure)
    assert driver.main([]) == 1
    assert "RuntimeError: boom" in capsys.readouterr().err
    assert shutdown_calls == ["shutdown", "shutdown"]


@pytest.mark.asyncio
async def test_shutdown_litellm_drains_worker_before_closing_clients(monkeypatch):
    events = []

    async def flush():
        events.append("flush")

    async def stop():
        events.append("stop")

    async def close_clients():
        events.append("close_clients")

    worker = SimpleNamespace(flush=flush, stop=stop)
    await driver._shutdown_litellm(worker, close_clients)

    assert events == ["flush", "stop", "close_clients"]


@pytest.mark.asyncio
async def test_shutdown_litellm_awaits_queued_callbacks(monkeypatch):
    callbacks = []
    worker, _close = driver._resolve_litellm_shutdown_boundary()
    worker = worker.__class__()
    worker.start()

    async def callback():
        await asyncio.sleep(0)
        callbacks.append("awaited")

    async def close_clients():
        return None

    worker.enqueue(callback())
    await driver._shutdown_litellm(worker, close_clients)

    assert callbacks == ["awaited"]


def test_litellm_shutdown_boundary_requires_frozen_version(monkeypatch):
    monkeypatch.setattr(driver, "package_version", lambda _name: "1.91.0")

    with pytest.raises(RuntimeError, match="expected 1.92.0, found 1.91.0"):
        driver._resolve_litellm_shutdown_boundary()


@pytest.mark.asyncio
async def test_shutdown_litellm_attempts_all_steps_after_failures():
    events = []

    async def flush():
        events.append("flush")
        raise OSError("flush failed")

    async def stop():
        events.append("stop")
        raise RuntimeError("stop failed")

    async def close_clients():
        events.append("close_clients")
        raise ValueError("close failed")

    worker = SimpleNamespace(flush=flush, stop=stop)
    with pytest.raises(RuntimeError, match=(
        "flush=OSError, stop=RuntimeError, close_clients=ValueError"
    )):
        await driver._shutdown_litellm(worker, close_clients)

    assert events == ["flush", "stop", "close_clients"]


@pytest.mark.asyncio
async def test_shutdown_litellm_bounds_flush_and_continues_cleanup():
    events = []

    async def flush():
        events.append("flush")
        await asyncio.Event().wait()

    async def stop():
        events.append("stop")

    async def close_clients():
        events.append("close_clients")

    worker = SimpleNamespace(flush=flush, stop=stop)
    with pytest.raises(RuntimeError, match="flush=TimeoutError"):
        await driver._shutdown_litellm(worker, close_clients, flush_timeout_seconds=0.01)

    assert events == ["flush", "stop", "close_clients"]


@pytest.mark.asyncio
async def test_primary_benchmark_failure_survives_shutdown_failure(monkeypatch):
    async def fail_run(_args):
        raise RuntimeError("primary benchmark failure")

    async def fail_shutdown(_worker, _close_clients):
        raise OSError("shutdown failure")

    monkeypatch.setattr(driver, "run_from_args", fail_run)
    monkeypatch.setattr(driver, "_shutdown_litellm", fail_shutdown)
    monkeypatch.setattr(
        driver,
        "_resolve_litellm_shutdown_boundary",
        lambda: (object(), lambda: None),
    )

    with pytest.raises(RuntimeError, match="primary benchmark failure") as captured:
        await driver._run_from_args_and_shutdown(object())

    assert any("LiteLLM shutdown also failed" in note for note in captured.value.__notes__)
    assert captured.value._spatial_atlas_shutdown_diagnostic == (
        "LiteLLM shutdown also failed: OSError"
    )


def test_main_reports_primary_and_secondary_shutdown_failures(monkeypatch, tmp_path, capsys):
    args = argparse.Namespace(output_dir=str(tmp_path))

    async def fail_run(_args):
        raise RuntimeError("primary benchmark failure")

    async def fail_shutdown(_worker, _close_clients):
        raise OSError("shutdown failure")

    monkeypatch.setattr(driver, "parse_args", lambda _argv: args)
    monkeypatch.setattr(driver, "run_from_args", fail_run)
    monkeypatch.setattr(driver, "_shutdown_litellm", fail_shutdown)
    monkeypatch.setattr(
        driver,
        "_resolve_litellm_shutdown_boundary",
        lambda: (object(), lambda: None),
    )

    assert driver.main([]) == 1
    stderr = capsys.readouterr().err
    assert "[fatal] RuntimeError: primary benchmark failure" in stderr
    assert "[cleanup] LiteLLM shutdown also failed: OSError" in stderr


@pytest.mark.asyncio
async def test_successful_benchmark_surfaces_shutdown_failure(monkeypatch):
    async def successful_run(_args):
        return {"ok": True}

    async def fail_shutdown(_worker, _close_clients):
        raise OSError("shutdown failure")

    monkeypatch.setattr(driver, "run_from_args", successful_run)
    monkeypatch.setattr(driver, "_shutdown_litellm", fail_shutdown)
    monkeypatch.setattr(
        driver,
        "_resolve_litellm_shutdown_boundary",
        lambda: (object(), lambda: None),
    )

    with pytest.raises(OSError, match="shutdown failure"):
        await driver._run_from_args_and_shutdown(object())


@pytest.mark.asyncio
async def test_label_free_run_never_reads_ground_truth_or_scores_before_sealing(
    tmp_path, png_bytes
):
    image = _image(tmp_path, png_bytes, "qspatial.png")
    sample = LabelFreeSample(0, "gap question", image)
    benchmark = FakeBenchmark([sample])
    config = SimpleNamespace(
        fieldwork_engine="metric",
        model_tiers={"fast": "fake/model"},
        llm_enable_thinking=None,
    )
    llm = FakeLLM(config=config)

    class EvidenceHandler(FakeFieldWorkHandler):
        async def handle(self, text, file_parts, updater, **kwargs):
            prediction = await super().handle(text, file_parts, updater, **kwargs)
            self.last_metric_grounding = "sam3+da3-gap"
            self.last_metric_evidence = {
                "geometry_valid": True,
                "geometry_status": "ok",
                "fallback_used": False,
                "parser": {"category": "distinct_pair"},
                "segmentation": {"instance_counts": [1, 1]},
                "reconstruction": {"metric_scale": 1.0},
                "geometry": {"gap_m": 1.25, "rendered_prediction": "1.25 meter"},
            }
            return prediction

    handler = EvidenceHandler(
        config=config,
        llm=llm,
        answers={"gap question": "1.25 meter"},
    )
    scoring = FakeScoring()
    output = tmp_path / "qspatial-run"

    results = await driver.run_benchmark(
        benchmark,
        handler,
        llm,
        scoring,
        engine="metric",
        output_dir=output,
    )

    row = _records(output / driver.PREDICTIONS_FILENAME)[0]
    assert not {"answer", "gt", "score"}.intersection(row)
    assert row["prediction_text"] == "1.25 meter"
    assert row["geometry_valid"] is True
    assert row["geometry_status"] == "ok"
    assert row["fallback_used"] is False
    assert row["metric_evidence"]["geometry"]["gap_m"] == 1.25
    assert {field for field in driver._PRESCORE_METADATA_FIELDS if field in row} == set(
        driver._PRESCORE_METADATA_FIELDS
    )
    assert handler.handle_kwargs == [
        {
            "metric_protocol": driver.QSPATIAL_METRIC_PROTOCOL,
            "metric_sample_metadata": sample.prediction_metadata(),
            "metric_question": "gap question",
        }
    ]
    assert benchmark.single_calls == []
    assert benchmark.aggregate_calls == []
    assert scoring.calls == []
    assert results["inference_only"] is True
    assert results["operational"]["completed_count"] == 1
    assert json.loads((output / driver.MANIFEST_FILENAME).read_text())["journal_schema"] == (
        driver.PRESCORE_SCHEMA
    )
    driver._assert_label_free(row, "test row")


@pytest.mark.asyncio
async def test_label_free_row_contains_exact_sanitized_llm_call_delta(tmp_path, png_bytes):
    image = _image(tmp_path, png_bytes, "qspatial-telemetry.png")
    sample = LabelFreeSample(0, "telemetry gap", image)
    benchmark = FakeBenchmark([sample])
    config = SimpleNamespace(
        fieldwork_engine="scenegraph",
        fieldwork_metric_strict=False,
        model_tiers={"main": "fake/model"},
        llm_enable_thinking=None,
    )
    llm = SimpleNamespace(config=config, cost_tracker=CostTracker())

    class TelemetryHandler(FakeFieldWorkHandler):
        async def handle(self, text, file_parts, updater, **kwargs):
            await updater.update_status("working", "telemetry")
            self.calls.append((text, file_parts))
            self.handle_kwargs.append(kwargs)
            self.last_fieldwork_engine = "scenegraph"
            response = SimpleNamespace(
                id="provider-id-not-for-journal",
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content="1.25 meter", reasoning_content=None),
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=3,
                    completion_tokens=2,
                    total_tokens=5,
                    completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
                ),
                _hidden_params={"response_cost": 0.0},
            )
            self.llm.cost_tracker.track(
                response,
                role="main",
                model_tier="main",
                call_kind="generate_with_messages",
                configured_max_tokens=8192,
            )
            return "1.25 meter"

    output = tmp_path / "qspatial-telemetry-run"
    await driver.run_benchmark(
        benchmark,
        TelemetryHandler(config=config, llm=llm),
        llm,
        FakeScoring(),
        engine="scenegraph",
        output_dir=output,
    )

    row = _records(output / driver.PREDICTIONS_FILENAME)[0]
    assert row["usage"]["num_calls"] == 1
    assert len(row["llm_call_telemetry"]) == row["usage"]["num_calls"]
    call = row["llm_call_telemetry"][0]
    assert call["role"] == call["model_tier"] == "main"
    assert call["call_kind"] == "generate_with_messages"
    assert call["configured_max_tokens"] == 8192
    assert call["provider_finish_reason"] == "stop"
    assert call["message_content_empty"] is False
    assert call["reasoning_content_empty"] is True
    serialized = json.dumps(row, sort_keys=True)
    assert "provider-id-not-for-journal" not in serialized
    assert "1.25 meter" in serialized
    driver._assert_label_free(row, "telemetry test row")


def test_real_tracker_telemetry_count_mismatch_fails_closed():
    class BrokenTracker(CostTracker):
        def telemetry_since(self, cursor):
            super().telemetry_since(cursor)
            return []

    tracker = BrokenTracker()
    llm = SimpleNamespace(cost_tracker=tracker)
    before_usage = driver._usage_snapshot(llm)
    cursor = driver._llm_call_telemetry_cursor(llm)
    tracker.track(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="ok", reasoning_content=None),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        ),
        role="main",
        model_tier="main",
        call_kind="generate",
        configured_max_tokens=128,
    )

    with pytest.raises(ValueError, match="telemetry count does not match usage num_calls"):
        driver._sample_observability(llm, before_usage, cursor)


@pytest.mark.asyncio
async def test_label_free_metadata_allowlist_fails_before_output_creation(tmp_path, png_bytes):
    sample = LabelFreeSample(
        0,
        "gap question",
        _image(tmp_path, png_bytes, "qspatial-extra.png"),
        extra_metadata={"answer_unit": "meter"},
    )
    output = tmp_path / "metadata-rejected"

    with pytest.raises(ValueError, match="exactly the six safe fields"):
        await driver.run_benchmark(
            FakeBenchmark([sample]),
            FakeFieldWorkHandler(),
            object(),
            FakeScoring(),
            engine="metric",
            output_dir=output,
        )

    assert not output.exists()


@pytest.mark.asyncio
async def test_mixed_normal_and_prescore_samples_fail_before_output_creation(tmp_path, png_bytes):
    image = _image(tmp_path, png_bytes, "mixed-schema.png")
    samples = [
        FakeSample(1, "normal", "distance", [image], "1 meter"),
        LabelFreeSample(2, "prescore", image),
    ]
    output = tmp_path / "mixed-schema"

    with pytest.raises(ValueError, match="mixes normal and label-free prescore"):
        await driver.run_benchmark(
            FakeBenchmark(samples),
            FakeFieldWorkHandler(),
            object(),
            FakeScoring(),
            engine="metric",
            output_dir=output,
        )

    assert not output.exists()


@pytest.mark.asyncio
async def test_label_free_resume_uses_prediction_text_and_skips_completed_id(tmp_path, png_bytes):
    image = _image(tmp_path, png_bytes, "qspatial-resume.png")
    samples = [
        LabelFreeSample(0, "first gap", image),
        LabelFreeSample(1, "second gap", image),
    ]
    benchmark = FakeBenchmark(samples)
    output = tmp_path / "qspatial-resume"
    output.mkdir()
    prior = {
        "schema_version": 1,
        "sample_id": 0,
        "prediction_text": "1 meter",
        **samples[0].prediction_metadata(),
        "usage": {"num_calls": 1},
        "geometry_valid": True,
        "geometry_status": "not_applicable",
        "fallback_used": False,
        "metric_evidence": None,
        "error": None,
    }
    (output / driver.PREDICTIONS_FILENAME).write_text(json.dumps(prior) + "\n", encoding="utf-8")
    config = SimpleNamespace(
        fieldwork_engine="scenegraph",
        model_tiers={"fast": "fake/model"},
        llm_enable_thinking=None,
    )
    llm = FakeLLM(config=config)
    handler = FakeFieldWorkHandler(
        config=config,
        llm=llm,
        answers={"second gap": "2 meter"},
    )
    _seed_manifest(output, benchmark, handler, "scenegraph")

    await driver.run_benchmark(
        benchmark,
        handler,
        llm,
        FakeScoring(),
        engine="scenegraph",
        output_dir=output,
        resume=True,
    )

    rows = _records(output / driver.PREDICTIONS_FILENAME)
    assert [row["sample_id"] for row in rows] == [0, 1]
    assert rows[-1]["prediction_text"] == "2 meter"
    assert len(handler.calls) == 1
    assert "second gap" in handler.calls[0][0]
    assert benchmark.single_calls == []
    assert benchmark.aggregate_calls == []


@pytest.mark.asyncio
async def test_label_free_controlled_four_plus_four_preserves_full_manifest(tmp_path, png_bytes):
    image = _image(tmp_path, png_bytes, "qspatial-four-plus-four.png")
    samples = [LabelFreeSample(index, f"gap {index}", image) for index in range(8)]
    benchmark = FakeBenchmark(samples)
    output = tmp_path / "qspatial-four-plus-four"
    config = SimpleNamespace(
        fieldwork_engine="scenegraph",
        model_tiers={"fast": "fake/model"},
        llm_enable_thinking=None,
    )
    llm = FakeLLM(config=config)
    handler = FakeFieldWorkHandler(
        config=config,
        llm=llm,
        answers={f"gap {index}": f"{index + 1} meter" for index in range(8)},
    )

    partial = await driver.run_benchmark(
        benchmark,
        handler,
        llm,
        FakeScoring(),
        engine="scenegraph",
        output_dir=output,
        max_new_samples=4,
        allow_partial=True,
    )
    manifest_before = (output / driver.MANIFEST_FILENAME).read_bytes()
    assert json.loads(manifest_before)["selected_ids"] == [str(index) for index in range(8)]
    assert [row["sample_id"] for row in _records(output / driver.PREDICTIONS_FILENAME)] == [
        0,
        1,
        2,
        3,
    ]
    assert partial["partial"] is True
    assert partial["operational"]["completed_count"] == 4

    resumed = await driver.run_benchmark(
        benchmark,
        handler,
        llm,
        FakeScoring(),
        engine="scenegraph",
        output_dir=output,
        resume=True,
    )
    assert (output / driver.MANIFEST_FILENAME).read_bytes() == manifest_before
    assert [row["sample_id"] for row in _records(output / driver.PREDICTIONS_FILENAME)] == list(
        range(8)
    )
    assert resumed.get("partial") is None
    assert resumed["operational"]["completed_count"] == 8


@pytest.mark.asyncio
async def test_resume_rejects_mixed_historical_row_schemas_without_mutating_journal(
    tmp_path, png_bytes
):
    image = _image(tmp_path, png_bytes, "qspatial-mixed-history.png")
    sample = LabelFreeSample(0, "gap question", image)
    benchmark = FakeBenchmark([sample])
    output = tmp_path / "mixed-history"
    output.mkdir()
    config = SimpleNamespace(
        fieldwork_engine="scenegraph",
        model_tiers={},
        llm_enable_thinking=None,
    )
    handler = FakeFieldWorkHandler(config=config, llm=FakeLLM(config=config))
    _seed_manifest(output, benchmark, handler, "scenegraph")
    normal = {"sample_id": 0, "answer": "1 meter", "gt": "1 meter", "score": 1.0}
    prescore = {
        "sample_id": 0,
        "prediction_text": "1 meter",
        **sample.prediction_metadata(),
        "error": None,
    }
    journal = output / driver.PREDICTIONS_FILENAME
    original = (json.dumps(normal) + "\n" + json.dumps(prescore) + "\n").encode()
    journal.write_bytes(original)

    with pytest.raises(ValueError, match="mixes normal and label-free prescore rows"):
        await driver.run_benchmark(
            benchmark,
            handler,
            handler.llm,
            FakeScoring(),
            engine="scenegraph",
            output_dir=output,
            resume=True,
        )

    assert journal.read_bytes() == original


@pytest.mark.asyncio
async def test_label_free_recursively_rejects_forbidden_metric_evidence(tmp_path, png_bytes):
    image = _image(tmp_path, png_bytes, "qspatial-forbidden-evidence.png")
    sample = LabelFreeSample(0, "gap question", image)
    benchmark = FakeBenchmark([sample])
    config = SimpleNamespace(
        fieldwork_engine="metric",
        model_tiers={},
        llm_enable_thinking=None,
    )
    llm = FakeLLM(config=config)

    class UnsafeEvidenceHandler(FakeFieldWorkHandler):
        async def handle(self, text, file_parts, updater, **kwargs):
            prediction = await super().handle(text, file_parts, updater, **kwargs)
            self.last_metric_evidence = {
                "geometry_valid": True,
                "geometry_status": "ok",
                "fallback_used": False,
                "geometry": {"reference_value": 1.25},
            }
            return prediction

    handler = UnsafeEvidenceHandler(
        config=config,
        llm=llm,
        answers={"gap question": "1.25 meter"},
    )
    output = tmp_path / "forbidden-evidence"

    with pytest.raises(RuntimeError, match="unresolved sample errors: 0"):
        await driver.run_benchmark(
            benchmark,
            handler,
            llm,
            FakeScoring(),
            engine="metric",
            output_dir=output,
        )

    row = _records(output / driver.PREDICTIONS_FILENAME)[0]
    assert "forbidden label field" in row["error"]
    assert row["metric_evidence"] is None
    driver._assert_label_free(row, "persisted error row")


@pytest.mark.asyncio
async def test_question_only_mode_supplies_no_image_bytes(tmp_path, png_bytes):
    image = _image(tmp_path, png_bytes, "question-only-source.png")
    sample = LabelFreeSample(0, "question-only gap", image)
    benchmark = FakeBenchmark([sample])
    config = SimpleNamespace(
        fieldwork_engine="scenegraph",
        model_tiers={},
        llm_enable_thinking=False,
    )
    llm = FakeLLM(config=config)
    handler = FakeFieldWorkHandler(
        config=config,
        llm=llm,
        answers={"question-only gap": "1 meter"},
    )

    await driver.run_benchmark(
        benchmark,
        handler,
        llm,
        FakeScoring(),
        engine="scenegraph",
        image_mode="question-only",
        output_dir=tmp_path / "question-only-run",
    )

    assert handler.calls[0][1] == []
    row = _records(tmp_path / "question-only-run" / driver.PREDICTIONS_FILENAME)[0]
    assert row["image_mode"] == "question-only"
    assert row["image_path"] is None
    assert row["source_image_path"] == str(Path(image).resolve())
    assert row["input_image_sha256"] is None
    assert row["control_mapping_sha256"] is None


@pytest.mark.asyncio
async def test_shuffled_mode_supplies_only_frozen_control_image(monkeypatch, tmp_path, png_bytes):
    source = _image(tmp_path, png_bytes, "source.png")
    control_bytes = png_bytes + b"control"
    control = tmp_path / "control.png"
    control.write_bytes(control_bytes)
    sample = LabelFreeSample(0, "shuffled gap", source)
    benchmark = FakeBenchmark([sample])
    benchmark.data_path = str(tmp_path)
    mapping = {
        "source.png": {
            "source_path": "source.png",
            "source_sha256": driver._sha256_file(Path(source)),
            "control_path": "control.png",
            "control_sha256": driver._sha256_file(control),
        }
    }
    artifact = {
        "path": str(tmp_path / "mapping.json"),
        "sha256": "8" * 64,
        "mapping_sha256": driver.QSPATIAL_CONTROL_MAPPING_SHA256,
        "image_manifest_sha256": driver.QSPATIAL_IMAGE_MANIFEST_SHA256,
    }
    monkeypatch.setattr(
        driver,
        "_load_qspatial_control_mapping",
        lambda _path, _root: (mapping, artifact),
    )
    config = SimpleNamespace(
        fieldwork_engine="metric",
        model_tiers={},
        llm_enable_thinking=False,
    )
    llm = FakeLLM(config=config)
    handler = FakeFieldWorkHandler(
        config=config,
        llm=llm,
        answers={"shuffled gap": "2 meter"},
    )

    await driver.run_benchmark(
        benchmark,
        handler,
        llm,
        FakeScoring(),
        engine="metric",
        image_mode="shuffled",
        control_mapping_path=tmp_path / "mapping.json",
        output_dir=tmp_path / "shuffled-run",
    )

    assert handler.calls[0][1][0][2] == control_bytes
    row = _records(tmp_path / "shuffled-run" / driver.PREDICTIONS_FILENAME)[0]
    assert row["image_mode"] == "shuffled"
    assert row["image_path"] == str(control.resolve())
    assert row["source_image_path"] == str(Path(source).resolve())
    assert row["input_image_sha256"] == driver._sha256_file(control)
    assert row["control_mapping_sha256"] == driver.QSPATIAL_CONTROL_MAPPING_SHA256
    manifest = json.loads((tmp_path / "shuffled-run" / driver.MANIFEST_FILENAME).read_text())
    assert manifest["control_mapping_artifact"] == artifact


@pytest.mark.asyncio
async def test_image_ablation_rejects_normal_scored_samples_before_output(tmp_path, png_bytes):
    sample = FakeSample(
        0,
        "normal sample",
        "distance",
        [_image(tmp_path, png_bytes, "normal.png")],
        "1 meter",
    )
    output = tmp_path / "invalid-ablation"

    with pytest.raises(ValueError, match="label-free prescore"):
        await driver.run_benchmark(
            FakeBenchmark([sample]),
            FakeFieldWorkHandler(),
            object(),
            FakeScoring(),
            engine="scenegraph",
            image_mode="question-only",
            output_dir=output,
        )

    assert not output.exists()
