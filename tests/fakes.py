"""In-memory fakes for SpatialClaw's GPU tools (no GPU, no server, no network).

Used to test the metric perception backend's orchestration + geometry by behavior,
not by patching internals (fakes-over-mocks, per Google testing guidance).
"""

from typing import ClassVar

import numpy as np


class FakePerFrameMask:
    """Stand-in for SpatialClaw's PerFrameMask."""

    def __init__(
        self,
        num_objects=1,
        centroid=(0.0, 0.0, 0.0),
        masks=None,
        labels=None,
        object_ids=None,
    ):
        if masks is None:
            masks = np.ones((1, num_objects, 8, 8), dtype=bool)
        self.masks = np.asarray(masks, dtype=bool)
        self.labels = labels or [f"object_{index}" for index in range(num_objects)]
        self.object_ids = object_ids or list(range(num_objects))
        self.num_objects = num_objects
        self.frame_indices = [0]
        self._centroid = centroid

    def get_mask(self, frame, object=0):
        assert frame == 0
        return self.masks[0, object]

    def get_centroid_3d(self, recon, frame, object):
        return None if self.num_objects == 0 else self._centroid


class FakeSAM3:
    """Stand-in for SAM3Tool: maps a text phrase to a preset mask."""

    def __init__(self, by_phrase):
        self._by_phrase = by_phrase
        self.is_available = True
        self.calls = []

    def segment_image_by_text(self, img, phrase, **kwargs):
        self.calls.append((phrase, kwargs))
        return self._by_phrase[phrase]


class _FakePoints:
    def __init__(self, points, confidence):
        self.points = np.asarray(points, dtype=float)[np.newaxis]
        self.confidence = np.asarray(confidence, dtype=float)[np.newaxis]

    def __getitem__(self, frame_index):
        assert frame_index == 0
        return self.points[0]

    def get_by_frame_index(self, frame_index):
        assert frame_index == 0
        return 0


class _FakeRecon:
    frame_indices: ClassVar[list[int]] = [0]
    metric_scale = 1.0

    def __init__(self, points, confidence):
        self.points = _FakePoints(points, confidence)


class FakeReconstructTool:
    """Stand-in for ReconstructTool."""

    is_available = True

    def __init__(self, points=None, confidence=None):
        if points is None:
            rows, columns = np.indices((8, 8))
            points = np.stack(
                [
                    columns.astype(float),
                    np.linspace(0.0, 1.0, 64).reshape(8, 8),
                    rows.astype(float),
                ],
                axis=-1,
            )
        if confidence is None:
            confidence = np.ones(np.asarray(points).shape[:2], dtype=float)
        self.points = points
        self.confidence = confidence
        self.calls = 0

    def Reconstruct(self, frames):
        self.calls += 1
        return _FakeRecon(self.points, self.confidence)


class FakeUsageStats:
    """Mutable token/cost totals with the same fields as CostTracker.stats."""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.num_calls = 0
        self.estimated_cost_usd = 0.0


class FakeLLM:
    """LLM stand-in exposing cumulative usage without making model calls."""

    def __init__(self, config=None):
        self.config = config
        self.cost_tracker = type("FakeCostTracker", (), {"stats": FakeUsageStats()})()


class FakeFieldWorkHandler:
    """Behavioral FieldWorkHandler fake for the benchmark driver."""

    def __init__(self, config=None, llm=None, answers=None, fail_on=()):
        self.config = config
        self.llm = llm
        self.answers = answers or {}
        self.fail_on = set(fail_on)
        self.calls = []
        self.handle_kwargs = []
        self.last_fieldwork_engine = None
        self.last_metric_grounding = None

    async def handle(self, text, file_parts, updater, **kwargs):
        await updater.update_status("working", "fake")
        self.calls.append((text, file_parts))
        self.handle_kwargs.append(kwargs)
        engine = getattr(self.config, "fieldwork_engine", None)
        if engine == "metric" or "metric_regions" in kwargs:
            self.last_fieldwork_engine = "metric"
            self.last_metric_grounding = "rle+da3" if "metric_regions" in kwargs else "sam3+da3"
        for marker in self.fail_on:
            if marker in text:
                raise RuntimeError(f"forced failure: {marker}")
        stats = getattr(getattr(self.llm, "cost_tracker", None), "stats", None)
        if stats is not None:
            stats.prompt_tokens += 10
            stats.completion_tokens += 2
            stats.total_tokens += 12
            stats.num_calls += 1
            stats.estimated_cost_usd += 0.01
        for marker, answer in self.answers.items():
            if marker in text:
                return answer
        return "0"
