"""
Spatial Atlas: FieldWorkArena Domain Handler

Main entry point for FieldWorkArena tasks. Orchestrates:
1. Goal parsing (structured task extraction)
2. Multimodal file processing (images, PDFs, videos, text)
3. Spatial scene graph construction
4. Entropy-guided reasoning
5. Output format matching

Receives: text goal + file attachments from green agent
Returns: formatted answer text
"""

import logging
from typing import Any

from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState
from a2a.utils import new_agent_text_message

from config import Config
from fieldwork.formatter import AnswerFormatter
from fieldwork.parser import GoalParser
from fieldwork.reasoner import FieldWorkReasoner
from fieldwork.spatial import SpatialAnalyzer
from fieldwork.vision import VisionPipeline
from llm import LLMClient

logger = logging.getLogger("spatial-atlas.fieldwork")


class FieldWorkHandler:
    """Handle FieldWorkArena benchmark tasks."""

    def __init__(self, config: Config, llm: LLMClient):
        self.config = config
        self.llm = llm
        self.parser = GoalParser()
        self.vision = VisionPipeline(llm, max_video_frames=config.max_video_frames)
        self.spatial = SpatialAnalyzer(llm)
        self.reasoner = FieldWorkReasoner(config, llm)
        self.formatter = AnswerFormatter()
        self.last_fieldwork_engine: str | None = None
        self.last_metric_grounding: str | None = None
        self.last_metric_evidence: dict[str, Any] | None = None

    async def handle(
        self,
        text: str,
        file_parts: list[tuple[str, str, str | bytes]],
        updater: TaskUpdater,
        metric_image_bytes: bytes | None = None,
        metric_regions: list[dict[str, Any]] | None = None,
        metric_protocol: str | None = None,
        metric_sample_metadata: dict[str, Any] | None = None,
        metric_question: str | None = None,
    ) -> str:
        """
        Process a FieldWorkArena task end-to-end.

        Args:
            text: Goal text from green agent (# Question / # Input Data / # Output Format)
            file_parts: List of (name, mime_type, data) for attached files
            updater: A2A task updater for progress reporting
            metric_image_bytes: Optional raw image for reconstructed geometry. The normal
                file parts remain available to the vision pipeline.
            metric_regions: Optional exact COCO RLE regions whose identities must be
                preserved instead of rediscovered with SAM3.
            metric_protocol: Optional frozen metric protocol identifier. QSpatial
                horizontal-gap rows use ``qspatial-horizontal-gap-v1``.
            metric_sample_metadata: Optional label-free benchmark metadata covered by
                the frozen QSpatial manifest.
            metric_question: Raw benchmark question for the frozen QSpatial grammar.
                The goal text wraps the question with instruction suffixes, so the
                frozen parser and evidence hashes must consume this exact string.

        Returns:
            Formatted answer string
        """
        # 1. Parse the goal string
        task = self.parser.parse(text)
        logger.info(f"Task parsed: query='{task.query[:80]}...', files={len(file_parts)}")

        # 2. Process all file attachments into text context
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Processing {len(file_parts)} input file(s)..."),
        )

        file_contexts = []
        for name, mime, data in file_parts:
            context = await self.vision.process_file(name, mime, data)
            file_contexts.append(context)

        logger.info(f"Processed {len(file_contexts)} files into context")

        # 3. Build spatial scene graph
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Building spatial scene graph..."),
        )

        # 3a. Optional metric perception: estimate 3D positions from masks and a
        # reconstructed point map through SpatialClaw (exact benchmark RLEs or SAM3,
        # plus Depth-Anything-3), so the LLM does not supply the coordinates.
        # Falls back to the scene-graph engine on any unavailability.
        scene = None
        self.last_fieldwork_engine = "scenegraph"
        self.last_metric_grounding = None
        self.last_metric_evidence = None
        if getattr(self.config, "fieldwork_engine", "scenegraph") == "metric":
            qspatial_metric = metric_protocol == "qspatial-horizontal-gap-v1"
            self.last_metric_grounding = (
                "sam3+da3-horizontal-surface-gap"
                if qspatial_metric
                else "rle+da3"
                if metric_regions is not None
                else "sam3+da3"
            )
            try:
                if metric_image_bytes is not None:
                    image_bytes = [metric_image_bytes]
                else:
                    image_bytes = [
                        (data if isinstance(data, bytes) else self.vision._decode_data(data))
                        for (_name, mime, data) in file_parts
                        if mime.startswith("image/")
                    ]
                if qspatial_metric:
                    from fieldwork.perception import (
                        QSPATIAL_UNAVAILABLE,
                        build_qspatial_gap_scene,
                        validate_qspatial_gap_scene,
                    )

                    if not isinstance(metric_question, str) or not metric_question.strip():
                        raise RuntimeError(
                            "parse_failed: QSpatial metric mode requires the raw "
                            "benchmark question, not the composed goal text"
                        )
                    scene, evidence = await build_qspatial_gap_scene(
                        metric_question,
                        image_bytes,
                        self.config,
                        sample_metadata=metric_sample_metadata,
                    )
                    self.last_metric_evidence = evidence
                    self.last_fieldwork_engine = "metric"
                    if scene is None:
                        logger.info(
                            "QSpatial metric grounding unavailable: %s",
                            evidence["geometry"]["invalid_reason"],
                        )
                        return QSPATIAL_UNAVAILABLE
                    validate_qspatial_gap_scene(scene, evidence)
                else:
                    from fieldwork.perception import build_metric_scene, validate_metric_scene

                    scene = await build_metric_scene(
                        task.query,
                        image_bytes,
                        self.llm,
                        self.config,
                        regions=metric_regions,
                    )
                    if getattr(self.config, "fieldwork_metric_strict", False):
                        validate_metric_scene(scene)
                self.last_fieldwork_engine = "metric"
                logger.info(
                    "metric perception (%s): %d grounded entities",
                    self.last_metric_grounding,
                    scene.entity_count,
                )
            except Exception as e:
                self.last_fieldwork_engine = "metric_failed"
                if qspatial_metric or getattr(self.config, "fieldwork_metric_strict", False):
                    logger.error("metric backend failed in strict evaluation mode: %s", e)
                    raise
                logger.warning("metric backend unavailable (%s); falling back to scene-graph", e)
                scene = None

        if scene is None:
            self.last_fieldwork_engine = "scenegraph"
            scene = await self.spatial.build_scene(task.query, file_contexts)
        logger.info(
            f"Scene: {scene.entity_count} entities, "
            f"{len(scene.relations)} relations, "
            f"{scene.violation_count} violations"
        )

        # 4. Reason over evidence to produce answer
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Reasoning over evidence..."),
        )

        answer = await self.reasoner.reason(
            query=task.query,
            file_contexts=file_contexts,
            scene=scene,
            output_format=task.output_format,
        )

        # 5. Format answer to match expected output
        formatted = self.formatter.format_answer(answer, task.output_format)
        logger.info(f"Answer formatted ({len(formatted)} chars): {formatted[:200]}...")

        return formatted
