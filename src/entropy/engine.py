"""
Spatial Atlas: Entropy-Guided Reasoning Engine

Estimates information gain to optimize reasoning trajectories.
Builds on the entropy-guided approach from Sprint 1.

Core idea: Before each reasoning step, estimate which action
would maximize information gain (reduce uncertainty the most).
This leads to more efficient trajectories and better cost-efficiency scores.
"""

import json
import logging
from typing import Any

logger = logging.getLogger("spatial-atlas.entropy")


class EntropyEngine:
    """Estimate information gain to guide reasoning decisions."""

    def __init__(self, llm):
        self.llm = llm

    async def select_best_action(
        self,
        knowledge_state: dict[str, Any],
        candidates: list[str],
        query: str,
    ) -> tuple[str, str]:
        """
        Given current knowledge and candidate actions, pick the highest info-gain action.

        Returns:
            Tuple of (selected_action, reasoning)
        """
        if len(candidates) <= 1:
            return candidates[0] if candidates else "", "Only one option available"

        state_summary = json.dumps(knowledge_state, indent=2, default=str)
        candidate_list = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))

        prompt = f"""You are optimizing an information-gathering trajectory.

Goal: Answer this question: {query}

Current knowledge state:
{state_summary}

Candidate next actions:
{candidate_list}

For each action, estimate how much it would reduce uncertainty about answering the question.
Rate each from 1-10 (10 = most informative, reduces uncertainty the most).

Return JSON:
{{"rankings": [{{"action_index": 0, "info_gain": 8, "reason": "..."}}]}}"""

        try:
            result = await self.llm.generate(prompt, model_tier="fast", json_mode=True)
            parsed = json.loads(result)
            rankings = parsed.get("rankings", [])

            if rankings:
                best = max(rankings, key=lambda r: r.get("info_gain", 0))
                idx = best.get("action_index", 0)
                reason = best.get("reason", "Highest estimated information gain")
                if 0 <= idx < len(candidates):
                    logger.info(f"Entropy: selected action {idx} (gain={best.get('info_gain')})")
                    return candidates[idx], reason
        except Exception as e:
            logger.warning(f"Entropy engine fallback (error: {e})")

        # Fallback: first candidate
        return candidates[0], "Fallback selection"

    async def estimate_confidence(
        self,
        answer: str,
        evidence: str,
        query: str,
    ) -> float:
        """
        Estimate confidence in an answer given the evidence.
        Returns 0.0-1.0 confidence score.
        Used to decide whether to reflect/refine.
        """
        prompt = f"""Rate your confidence that this answer correctly addresses the question.

Question: {query}

Evidence summary (truncated):
{evidence[:2000]}

Proposed answer:
{answer}

Rate confidence from 0.0 (no confidence) to 1.0 (fully confident).
Consider: Does the evidence support the answer? Are there gaps?

Return JSON: {{"confidence": 0.85, "gaps": ["list of any gaps"]}}"""

        try:
            result = await self.llm.generate(prompt, model_tier="fast", json_mode=True)
            parsed = json.loads(result)
            return float(parsed.get("confidence", 0.5))
        except Exception:
            return 0.5  # Default moderate confidence
