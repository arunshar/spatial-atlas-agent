"""
Spatial Atlas: FieldWorkArena Goal Parser

Parses the structured goal string sent by the FWA green agent into
a structured task object: query, input files, and output format.

Goal format:
    # Question
    {query}

    # Input Data
    {file1.jpg}
    {file2.pdf}

    # Output Format
    {expected format description}
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger("spatial-atlas.fieldwork.parser")


@dataclass
class FieldWorkTask:
    query: str
    input_files: list[str] = field(default_factory=list)
    output_format: str = ""
    raw_goal: str = ""


class GoalParser:
    """Parse FieldWorkArena goal strings into structured tasks."""

    # Section header patterns
    SECTION_PATTERN = re.compile(
        r"^#\s+(Question|Input Data|Output Format)\s*$",
        re.MULTILINE | re.IGNORECASE,
    )

    def parse(self, goal_text: str) -> FieldWorkTask:
        """Parse a goal string into a FieldWorkTask."""
        task = FieldWorkTask(raw_goal=goal_text, query="")

        # Split into sections
        sections = self._split_sections(goal_text)

        task.query = sections.get("question", "").strip()
        task.output_format = sections.get("output format", "").strip()

        # Parse input data section into file names
        input_data = sections.get("input data", "").strip()
        if input_data:
            task.input_files = [line.strip() for line in input_data.splitlines() if line.strip()]

        # Fallback: if no structured sections found, treat entire text as query
        if not task.query:
            task.query = goal_text.strip()
            logger.warning("No structured goal format detected, using full text as query")

        logger.info(
            f"Parsed task: query={task.query[:100]}..., "
            f"files={len(task.input_files)}, "
            f"format={task.output_format[:50]}..."
        )
        return task

    def _split_sections(self, text: str) -> dict[str, str]:
        """Split goal text by # headers into named sections."""
        sections: dict[str, str] = {}
        matches = list(self.SECTION_PATTERN.finditer(text))

        if not matches:
            return sections

        for i, match in enumerate(matches):
            section_name = match.group(1).lower()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections[section_name] = text[start:end]

        return sections
