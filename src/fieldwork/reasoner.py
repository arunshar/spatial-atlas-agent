"""
Spatial Atlas: FieldWork Reasoning Engine

Combines spatial scene analysis, file evidence, and entropy-guided
reasoning to produce accurate answers for FieldWorkArena tasks.
"""

import logging

from config import Config
from entropy.engine import EntropyEngine
from fieldwork.spatial import SpatialScene
from llm import LLMClient

logger = logging.getLogger("spatial-atlas.fieldwork.reasoner")


REASONING_SYSTEM_PROMPT = """You are an expert field operations and safety analyst.
You analyze factory, warehouse, and retail environments for safety compliance,
operational efficiency, and spatial awareness.

Key principles:
- Answer ONLY based on the provided evidence
- Be precise with counts, measurements, and distances
- Use the computed spatial analysis when available (these are deterministic calculations)
- Match the expected output format exactly
- If the expected output is JSON, respond with ONLY valid JSON
- If you cannot determine something from the evidence, say "N/A"
- Be concise; do not add unnecessary explanation unless the format requires it"""


class FieldWorkReasoner:
    """Reason over spatial scenes and evidence to produce answers."""

    def __init__(self, config: Config, llm: LLMClient):
        self.config = config
        self.llm = llm
        self.entropy = EntropyEngine(llm)

    async def reason(
        self,
        query: str,
        file_contexts: list[str],
        scene: SpatialScene | None,
        output_format: str,
    ) -> str:
        """
        Produce an answer by reasoning over evidence + spatial facts.

        Args:
            query: The question to answer
            file_contexts: Text descriptions of all input files
            scene: Spatial scene graph with computed facts
            output_format: Expected output format from green agent

        Returns:
            The answer string
        """
        # Build comprehensive evidence
        evidence = "\n\n".join(file_contexts)

        # Add deterministically computed spatial facts
        spatial_section = ""
        if scene and scene.entity_count > 0:
            spatial_facts = scene.to_fact_sheet()
            spatial_section = f"\n\n## Computed Spatial Analysis (deterministic)\n{spatial_facts}"

        # Construct the reasoning prompt
        prompt = f"""## Question
{query}

## Evidence from Input Files
{evidence[:12000]}
{spatial_section}

## Expected Output Format
{output_format}

## Instructions
Answer the question based on the evidence above.
- If the output format specifies JSON, respond with ONLY valid JSON (no markdown, no explanation)
- If the output format specifies a number, respond with ONLY the number
- If the output format specifies yes/no, respond with ONLY "yes" or "no"
- Use the computed spatial analysis for distances, counts, and violations
- Be precise and concise"""

        # Check confidence and potentially reflect
        answer = await self.llm.generate(
            prompt,
            model_tier="strong",
            system_prompt=REASONING_SYSTEM_PROMPT,
            max_tokens=4096,
        )

        # Entropy-guided confidence check: refine if low confidence
        if self.config.max_reflection_rounds > 0:
            confidence = await self.entropy.estimate_confidence(
                answer=answer,
                evidence=evidence[:2000],
                query=query,
            )
            logger.info(f"Answer confidence: {confidence:.2f}")

            if confidence < 0.6:
                logger.info("Low confidence: reflecting and refining answer")
                answer = await self._refine_answer(
                    query, answer, evidence[:6000], spatial_section, output_format
                )

        return answer.strip()

    async def _refine_answer(
        self,
        query: str,
        initial_answer: str,
        evidence: str,
        spatial_section: str,
        output_format: str,
    ) -> str:
        """Reflect on and refine a low-confidence answer."""
        prompt = f"""Your initial answer to a question may not be fully accurate.
Review it and provide a corrected answer.

## Question
{query}

## Initial Answer
{initial_answer}

## Evidence
{evidence}
{spatial_section}

## Expected Output Format
{output_format}

## Instructions
- Identify any errors or gaps in the initial answer
- Provide a corrected, complete answer
- Match the expected output format exactly
- Respond ONLY with the corrected answer, no meta-commentary"""

        refined = await self.llm.generate(
            prompt,
            model_tier="strong",
            system_prompt=REASONING_SYSTEM_PROMPT,
            max_tokens=4096,
        )
        return refined
