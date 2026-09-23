"""
Spatial Atlas: multimodal vision pipeline

Processes all file types from FieldWorkArena:
- Images (JPEG) → Vision model detailed description
- PDFs → Text extraction via pypdf
- Videos (MP4) → Frame extraction + per-frame analysis
- Text files → Direct inclusion

Each file is converted to a rich text context for downstream reasoning.
"""

import base64
import io
import logging
import tempfile
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

from llm import LLMClient

logger = logging.getLogger("spatial-atlas.fieldwork.vision")

# Vision prompt optimized for spatial/safety analysis
VISION_PROMPT = (
    "Describe this image in detail for a safety and operations analyst. Include:\n"
    "1. ALL objects, people, vehicles, and equipment, with their approximate positions "
    "(left, right, center, foreground, background, distances if estimable)\n"
    "2. Safety equipment present or absent (PPE: hard hats, safety vests, goggles, gloves)\n"
    "3. Environment type (factory floor, warehouse aisle, loading dock, retail area)\n"
    "4. Any safety violations or hazards (blocked exits, missing PPE, proximity to machinery)\n"
    "5. Spatial relationships between objects (A is near B, C is blocking D)\n"
    "6. Estimated distances and measurements where possible\n"
    "7. Any text, signs, labels, or markings visible\n"
    "8. Activities being performed and their safety implications\n"
    "Be precise about quantities, positions, and spatial relationships."
)

VIDEO_FRAME_PROMPT = (
    "Describe this video frame from a workplace environment. Focus on:\n"
    "- Objects, people, and their positions\n"
    "- Activities being performed\n"
    "- Safety equipment and any violations\n"
    "- Changes from previous frames (if context suggests movement)\n"
    "Be concise but precise about spatial details."
)


class VisionPipeline:
    """Process multimodal file attachments into text context."""

    def __init__(self, llm: LLMClient, max_video_frames: int = 30):
        self.llm = llm
        self.max_video_frames = max_video_frames
        # Lazy-loaded local detector for structured preprocessing
        self._detector = None

    async def process_file(self, name: str, mime_type: str, data: str | bytes) -> str:
        """Convert any file attachment to text context."""
        logger.info(f"Processing file: {name} ({mime_type})")

        if mime_type.startswith("image/"):
            return await self._process_image(name, data)
        elif mime_type == "application/pdf" or (name and name.endswith(".pdf")):
            return self._process_pdf(name, data)
        elif mime_type.startswith("video/") or (name and name.endswith(".mp4")):
            return await self._process_video(name, data)
        elif mime_type.startswith("text/") or (name and name.endswith(".txt")):
            return self._process_text(name, data)
        else:
            logger.warning(f"Unsupported file type: {mime_type} for {name}")
            return f"[Unsupported file: {name} ({mime_type})]"

    async def _process_image(self, name: str, data: str | bytes) -> str:
        """Analyze image with local detector + vision model."""
        image_bytes = self._decode_data(data)

        # Ensure valid JPEG
        try:
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            image_bytes = buf.getvalue()
        except Exception as e:
            logger.warning(f"Image preprocessing failed for {name}: {e}")

        # Step 1: Run local detector for structured object counts (if available)
        detection_text = ""
        try:
            from fieldwork.detector import get_detector

            detector = get_detector()
            detection = await detector.detect(image_bytes)
            detection_text = detection.to_structured_text()
            if detection_text:
                logger.info(f"Local detection for {name}: {detection.total_objects} objects")
        except Exception as e:
            logger.debug(f"Local detection skipped for {name}: {e}")

        # Step 2: VLM analysis (enriched with detection context if available)
        prompt = VISION_PROMPT
        if detection_text:
            prompt = (
                f"{VISION_PROMPT}\n\n"
                f"IMPORTANT: A detection model has already identified the following objects. "
                f"Use these EXACT counts; do NOT re-count:\n{detection_text}"
            )

        description = await self.llm.vision_analyze(
            image_bytes=image_bytes,
            prompt=prompt,
        )

        parts = [f"[Image: {name}]"]
        if detection_text:
            parts.append(detection_text)
        parts.append(description)
        return "\n".join(parts)

    def _process_pdf(self, name: str, data: str | bytes) -> str:
        """Extract text from PDF."""
        pdf_bytes = self._decode_data(data)
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text_parts = []
            for i, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"--- Page {i} ---\n{page_text}")
            full_text = "\n\n".join(text_parts) if text_parts else "[Empty PDF]"
            return f"[PDF: {name}]\n{full_text}"
        except Exception as e:
            logger.error(
                "PDF extraction failed [exception_type=%s]",
                type(e).__name__,
            )
            return f"[PDF: {name}] Error: extraction failed"

    async def _process_video(self, name: str, data: str | bytes) -> str:
        """Extract frames from video and analyze key frames."""
        video_bytes = self._decode_data(data)

        try:
            frames = self._extract_video_frames(video_bytes)
            if not frames:
                return f"[Video: {name}] No frames extracted"

            descriptions = [f"Video file: {name} ({len(frames)} frames extracted)"]

            # Analyze key frames (not all, to save tokens)
            analysis_frames = self._select_key_frames(frames)
            for timestamp, frame_bytes in analysis_frames:
                desc = await self.llm.vision_analyze(
                    image_bytes=frame_bytes,
                    prompt=f"Frame at {timestamp}. {VIDEO_FRAME_PROMPT}",
                )
                descriptions.append(f"[Frame at {timestamp}]: {desc}")

            return f"[Video: {name}]\n" + "\n\n".join(descriptions)
        except Exception as e:
            logger.error(
                "Video processing failed [exception_type=%s]",
                type(e).__name__,
            )
            return f"[Video: {name}] Error: processing failed"

    def _process_text(self, name: str, data: str | bytes) -> str:
        """Read text file content."""
        raw = self._decode_data(data)
        try:
            text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        except UnicodeDecodeError:
            text = raw.decode("latin-1") if isinstance(raw, bytes) else str(raw)
        return f"[Text: {name}]\n{text}"

    def _decode_data(self, data: str | bytes) -> bytes:
        """Decode base64 string or return raw bytes."""
        if isinstance(data, str):
            # Strip data URI prefix if present
            if data.startswith("data:"):
                data = data.split(",", 1)[1]
            data = data.replace(" ", "").replace("\n", "").replace("\r", "")
            return base64.b64decode(data)
        return data

    def _extract_video_frames(
        self, video_bytes: bytes, seconds_per_frame: int = 2
    ) -> list[tuple[str, bytes]]:
        """Extract frames from video at specified interval."""
        import cv2

        # Write to temp file for OpenCV
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            temp_path = f.name

        frames = []
        try:
            cap = cv2.VideoCapture(temp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30

            skip = int(fps * seconds_per_frame)
            # Limit frames
            if skip > 0 and total_frames / skip > self.max_video_frames:
                skip = max(1, total_frames // self.max_video_frames)

            curr = 0
            while curr < total_frames and len(frames) < self.max_video_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, curr)
                ok, frame = cap.read()
                if not ok:
                    break

                # Convert to JPEG bytes
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)

                timestamp = self._format_timestamp(curr / fps)
                frames.append((timestamp, buf.getvalue()))
                curr += skip

            cap.release()
        finally:
            Path(temp_path).unlink(missing_ok=True)

        return frames

    def _select_key_frames(
        self, frames: list[tuple[str, bytes]], max_analyzed: int = 10
    ) -> list[tuple[str, bytes]]:
        """Select a subset of frames to analyze (evenly spaced)."""
        if len(frames) <= max_analyzed:
            return frames
        step = len(frames) / max_analyzed
        indices = [int(i * step) for i in range(max_analyzed)]
        return [frames[i] for i in indices]

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
