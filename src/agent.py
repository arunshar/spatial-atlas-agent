"""
Spatial Atlas: core orchestrator agent

THE BRAIN: Receives A2A messages, classifies domain (FieldWorkArena vs MLE-Bench),
routes to the appropriate handler, and returns formatted artifacts.
"""

import base64
import logging
import uuid

from a2a.server.tasks import TaskUpdater
from a2a.types import (
    DataPart,
    FilePart,
    FileWithBytes,
    Message,
    Part,
    TaskState,
    TextPart,
)
from a2a.utils import new_agent_text_message

from budgeted_llm import BudgetedLLMClient
from config import Config
from fieldwork.handler import FieldWorkHandler
from mlebench.handler import MLEBenchHandler

logger = logging.getLogger("spatial-atlas.agent")


class AgentTaskError(RuntimeError):
    """Public, sanitized failure raised after server-side diagnostics are recorded."""

    def __init__(self, reference: str):
        self.reference = reference
        super().__init__(f"Spatial Atlas could not complete this task. Reference: {reference}")


class Agent:
    def __init__(self):
        self.config = Config()
        self.llm = BudgetedLLMClient(self.config)
        self.cost_tracker = self.llm.cost_tracker
        self.fieldwork = FieldWorkHandler(self.config, self.llm)
        self.mlebench = MLEBenchHandler(self.config, self.llm)
        self.messages: list[dict] = []

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Main entry point: classify domain and route to handler."""
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Spatial Atlas processing..."),
        )

        try:
            # Parse message into text and file parts
            text_parts, file_parts = self._parse_message(message)
            full_text = "\n".join(text_parts)

            # Classify which benchmark is calling
            domain = self._classify_domain(full_text, file_parts)
            logger.info(f"Domain classified as: {domain}")

            if domain == "mlebench":
                await self._handle_mlebench(full_text, file_parts, updater)
            else:
                await self._handle_fieldwork(full_text, file_parts, updater)

            logger.info(f"Task completed. {self.cost_tracker.summary()}")

        except Exception as e:
            failure_id = uuid.uuid4().hex[:12]
            logger.error(
                "Agent task failed [reference=%s, exception_type=%s]",
                failure_id,
                type(e).__name__,
            )
            public_message = f"Spatial Atlas could not complete this task. Reference: {failure_id}"
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=f"Error: {public_message}"))],
                name="Error",
            )
            raise AgentTaskError(failure_id) from e

    async def _handle_fieldwork(
        self,
        text: str,
        file_parts: list[tuple[str, str, str | bytes]],
        updater: TaskUpdater,
    ) -> None:
        """Handle FieldWorkArena tasks."""
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Analyzing field work task with spatial reasoning..."),
        )

        result = await self.fieldwork.handle(text, file_parts, updater)

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=result))],
            name="Analysis",
        )

    async def _handle_mlebench(
        self,
        text: str,
        file_parts: list[tuple[str, str, str | bytes]],
        updater: TaskUpdater,
    ) -> None:
        """Handle MLE-Bench tasks."""
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Starting ML engineering pipeline..."),
        )

        csv_bytes, summary = await self.mlebench.handle(text, file_parts, updater)

        parts = [Part(root=TextPart(text=summary))]

        if csv_bytes:
            b64_csv = base64.b64encode(csv_bytes).decode("ascii")
            parts.append(
                Part(
                    root=FilePart(
                        file=FileWithBytes(
                            bytes=b64_csv,
                            name="submission.csv",
                            mime_type="text/csv",
                        )
                    )
                )
            )

        await updater.add_artifact(parts=parts, name="Submission")

    def _parse_message(
        self, message: Message
    ) -> tuple[list[str], list[tuple[str, str, str | bytes]]]:
        """
        Parse A2A message into text content and file attachments.

        Returns:
            text_parts: List of text strings from the message
            file_parts: List of (name, mime_type, data) tuples for file attachments
        """
        text_parts: list[str] = []
        file_parts: list[tuple[str, str, str | bytes]] = []

        for part in message.parts:
            root = part.root
            if isinstance(root, TextPart):
                text_parts.append(root.text)
            elif isinstance(root, DataPart):
                # Inline JSON data
                import json

                text_parts.append(json.dumps(root.data, indent=2))
            elif isinstance(root, FilePart):
                file_data = root.file
                if isinstance(file_data, FileWithBytes):
                    name = file_data.name or "unknown"
                    mime = file_data.mime_type or "application/octet-stream"
                    data = file_data.bytes  # base64-encoded string or bytes
                    file_parts.append((name, mime, data))

        return text_parts, file_parts

    def _classify_domain(
        self,
        text: str,
        file_parts: list[tuple[str, str, str | bytes]],
    ) -> str:
        """
        Classify which benchmark is calling based on message content.

        Detection strategy:
        1. MLE-Bench: has competition.tar.gz attachment
        2. FieldWorkArena: has '# Question' + '# Output Format' in text
        3. MLE-Bench: text mentions Kaggle/MLE-bench/competition
        4. Default: FieldWorkArena
        """
        # Check for MLE-Bench tar file
        for name, mime, _data in file_parts:
            if name and ("competition" in name.lower() or name.endswith(".tar.gz")):
                return "mlebench"
            if mime and "gzip" in mime:
                return "mlebench"

        # Check for FieldWorkArena goal format
        text_lower = text.lower()
        if "# question" in text_lower and "# output format" in text_lower:
            return "fieldwork"

        # Check for MLE-Bench keywords
        mle_keywords = ["kaggle", "mle-bench", "competition", "submission.csv", "train a model"]
        if any(kw in text_lower for kw in mle_keywords):
            return "mlebench"

        # Default to fieldwork (more common in research agent track)
        return "fieldwork"
