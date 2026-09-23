"""
Spatial Atlas: local object detection preprocessing

Uses Florence-2 (or falls back to a simple prompt-based approach) to extract
structured object counts and bounding boxes BEFORE sending to the LLM.

This addresses a known VLM weakness: large language models are notoriously
bad at precise counting and spatial measurement. By running a dedicated
detection model first, we feed deterministic structured data to the reasoner.

The detector is optional. If the model cannot be loaded (no GPU or dependencies),
the pipeline falls back gracefully to pure VLM analysis.
"""

import importlib.util
import io
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("spatial-atlas.fieldwork.detector")


# Detection results
@dataclass
class DetectedObject:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # x1, y1, x2, y2 normalized
    area_fraction: float = 0.0  # fraction of image area


@dataclass
class DetectionResult:
    objects: list[DetectedObject] = field(default_factory=list)
    object_counts: dict[str, int] = field(default_factory=dict)
    total_objects: int = 0
    ppe_detected: dict[str, bool] = field(default_factory=dict)  # hard_hat, vest, goggles, gloves
    raw_caption: str = ""

    def to_structured_text(self) -> str:
        """Convert detection results to structured text for the LLM."""
        if not self.objects and not self.raw_caption:
            return ""

        lines = ["## Pre-Analyzed Detection Results (deterministic)"]

        if self.object_counts:
            lines.append(f"Total objects detected: {self.total_objects}")
            lines.append("Object counts:")
            for label, count in sorted(self.object_counts.items()):
                lines.append(f"  - {label}: {count}")

        if self.ppe_detected:
            lines.append("PPE status:")
            for item, present in sorted(self.ppe_detected.items()):
                status = "PRESENT" if present else "NOT DETECTED"
                lines.append(f"  - {item}: {status}")

        if self.raw_caption:
            lines.append(f"Scene caption: {self.raw_caption}")

        return "\n".join(lines)


class LocalDetector:
    """
    Local object detection using Florence-2 or fallback approaches.

    Priority:
    1. Florence-2 (if transformers and torch are available), best quality
    2. Structured VLM prompt (if only LLM is available), good quality
    3. Skip (return empty), graceful fallback
    """

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = None
        self._available = None  # tri-state: None=unchecked, True, False

    def _check_availability(self) -> bool:
        """Check if Florence-2 can be loaded."""
        if self._available is not None:
            return self._available

        if all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers")):
            self._available = True
            logger.info("Florence-2 dependencies available")
        else:
            self._available = False
            logger.info("Florence-2 not available (no torch/transformers), using VLM fallback")

        return self._available

    async def _load_model(self):
        """Lazy-load Florence-2 model."""
        if self._model is not None:
            return

        if not self._check_availability():
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

            model_id = "microsoft/Florence-2-base"
            model_revision = "5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac"
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Loading Florence-2 on {self._device}...")
            self._processor = AutoProcessor.from_pretrained(
                model_id,
                revision=model_revision,
                trust_remote_code=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=model_revision,
                trust_remote_code=True,
            ).to(self._device)
            self._model.eval()
            logger.info("Florence-2 loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load Florence-2: {e}")
            self._available = False
            self._model = None

    async def detect(self, image_bytes: bytes) -> DetectionResult:
        """
        Run object detection on an image.

        Returns DetectionResult with counts, bounding boxes, and PPE status.
        Falls back gracefully if model is unavailable.
        """
        await self._load_model()

        if self._model is None:
            return DetectionResult()  # Empty result, pipeline continues with pure VLM

        try:
            return await self._detect_florence(image_bytes)
        except Exception as e:
            logger.warning(f"Florence-2 detection failed: {e}")
            return DetectionResult()

    async def _detect_florence(self, image_bytes: bytes) -> DetectionResult:
        """Run Florence-2 detection pipeline."""
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = img.size
        img_area = width * height

        result = DetectionResult()

        # Task 1: Dense captioning for overall scene understanding
        caption = await self._florence_infer(img, "<MORE_DETAILED_CAPTION>")
        result.raw_caption = caption

        # Task 2: Object detection with bounding boxes
        od_result = await self._florence_infer(img, "<OD>")
        if isinstance(od_result, dict) and "bboxes" in od_result:
            labels = od_result.get("labels", [])
            bboxes = od_result.get("bboxes", [])

            for i, (label, bbox) in enumerate(zip(labels, bboxes)):
                label_clean = label.strip().lower()
                x1, y1, x2, y2 = bbox
                area = abs((x2 - x1) * (y2 - y1))

                obj = DetectedObject(
                    label=label_clean,
                    confidence=0.8,  # Florence-2 doesn't return confidence scores
                    bbox=(x1 / width, y1 / height, x2 / width, y2 / height),
                    area_fraction=area / img_area if img_area > 0 else 0,
                )
                result.objects.append(obj)
                result.object_counts[label_clean] = result.object_counts.get(label_clean, 0) + 1

        result.total_objects = len(result.objects)

        # PPE detection from object labels
        ppe_keywords = {
            "hard_hat": ["hard hat", "helmet", "safety helmet"],
            "safety_vest": ["vest", "safety vest", "hi-vis", "high visibility"],
            "goggles": ["goggles", "safety glasses", "glasses"],
            "gloves": ["gloves", "safety gloves"],
            "mask": ["mask", "face mask", "respirator"],
        }

        for ppe_item, keywords in ppe_keywords.items():
            found = any(any(kw in obj.label for kw in keywords) for obj in result.objects)
            result.ppe_detected[ppe_item] = found

        logger.info(
            f"Detection: {result.total_objects} objects, "
            f"PPE: {sum(result.ppe_detected.values())}/{len(result.ppe_detected)} items"
        )
        return result

    async def _florence_infer(self, image, task: str):
        """Run a single Florence-2 inference task."""
        import torch

        inputs = self._processor(text=task, images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=1024,
                num_beams=3,
            )

        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(
            generated_text, task=task, image_size=image.size
        )

        # parsed is a dict like {task: result}
        return parsed.get(task, generated_text)


# Singleton detector instance
_detector: LocalDetector | None = None


def get_detector() -> LocalDetector:
    """Get or create the singleton detector."""
    global _detector
    if _detector is None:
        _detector = LocalDetector()
    return _detector
