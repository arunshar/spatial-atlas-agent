"""
Spatial Atlas: Output Format Matcher

Ensures answers match the expected output format for FieldWorkArena scoring.
Critical because the green agent uses exact_match, must_include, json_match, etc.
A correct answer in the wrong format scores 0.
"""

import json
import logging
import re

logger = logging.getLogger("spatial-atlas.fieldwork.formatter")


# Output-format keywords that mean "this is a yes/no question."
# FieldWorkArena green agents phrase this many ways; missing any of them
# causes a silent score 0 on boolean questions.
_BOOLEAN_KEYWORDS = (
    "yes/no",
    "yes or no",
    "y/n",
    "boolean",
    "bool",
    "true/false",
    "true or false",
)

# Fenced code block grabber (```json ... ``` or ``` ... ```).
# Non-greedy so we don't span multiple blocks; we'll iterate over matches.
_FENCED_CODE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class AnswerFormatter:
    """Format raw LLM answers to match expected output formats."""

    def format_answer(self, raw_answer: str, output_format: str) -> str:
        """
        Ensure answer matches the expected output format.

        Args:
            raw_answer: The raw answer from the reasoner
            output_format: The expected format description from the green agent

        Returns:
            Formatted answer string
        """
        fmt_lower = output_format.lower()
        answer = raw_answer.strip()

        # JSON format
        if "json" in fmt_lower or output_format.strip().startswith("{"):
            return self._format_json(answer)

        # Numeric format
        if any(kw in fmt_lower for kw in ["number", "count", "integer", "how many"]):
            return self._format_numeric(answer)

        # Boolean format (see _BOOLEAN_KEYWORDS above for the full list).
        if any(kw in fmt_lower for kw in _BOOLEAN_KEYWORDS):
            return self._format_boolean(answer, fmt_lower)

        # List format
        if "list" in fmt_lower or "comma" in fmt_lower:
            return self._format_list(answer)

        # Strip markdown artifacts
        answer = self._strip_markdown(answer)

        return answer

    def _format_json(self, answer: str) -> str:
        """
        Extract and clean JSON from answer.

        Strategy (try each; return the first one that parses):
          1. Parse the whole answer.
          2. Unwrap any fenced code blocks (```json ... ```) and parse each.
          3. Scan the answer for balanced {...} or [...] substrings and
             try every candidate, preferring the longest one. A plain
             non-greedy or greedy regex either grabs too little or spans
             unrelated braces; balanced scanning is the only reliable fix.
        """
        # 1. Whole answer.
        try:
            return json.dumps(json.loads(answer))
        except json.JSONDecodeError:
            pass

        # 2. Fenced code blocks.
        for match in _FENCED_CODE_RE.finditer(answer):
            candidate = match.group(1).strip()
            try:
                return json.dumps(json.loads(candidate))
            except json.JSONDecodeError:
                continue

        # 3. Balanced-brace substrings, longest first.
        candidates = sorted(
            _iter_balanced_substrings(answer),
            key=len,
            reverse=True,
        )
        for candidate in candidates:
            try:
                return json.dumps(json.loads(candidate))
            except json.JSONDecodeError:
                continue

        logger.warning(f"Could not extract JSON from answer: {answer[:200]}")
        return answer

    def _format_numeric(self, answer: str) -> str:
        """Extract numeric value from answer."""
        # Try whole answer as number
        try:
            val = float(answer)
            return str(int(val)) if val == int(val) else str(val)
        except ValueError:
            pass

        # Extract first number from text
        numbers = re.findall(r'-?\d+\.?\d*', answer)
        if numbers:
            val = float(numbers[0])
            return str(int(val)) if val == int(val) else str(val)

        return answer

    def _format_boolean(self, answer: str, fmt_lower: str = "") -> str:
        """
        Normalize to the boolean vocabulary requested by fmt_lower.

        Vocab selection:
          - If fmt_lower mentions 'true' or 'bool' (but not 'yes'), return 'true'/'false'.
          - Otherwise default to 'yes'/'no'.

        Resolution:
          1. Exact match against known affirmatives/negatives.
          2. Starts-with check for short leading tokens ('yes, because...').
          3. Fall back to raw answer so nothing is lost silently.
        """
        use_true_false = (
            ("true" in fmt_lower or "bool" in fmt_lower) and "yes" not in fmt_lower
        )
        yes_token, no_token = ("true", "false") if use_true_false else ("yes", "no")

        answer_lower = answer.lower().strip().rstrip(".!?")
        affirmatives = {"yes", "true", "correct", "affirmative", "y", "t", "1"}
        negatives = {"no", "false", "incorrect", "negative", "n", "f", "0"}

        if answer_lower in affirmatives:
            return yes_token
        if answer_lower in negatives:
            return no_token

        # Token scan: find the FIRST affirmative or negative whole-word token
        # in the answer. Handles both "Yes, because..." and "The answer is true."
        # Whole-word matching avoids 'no' matching 'not' or 'none'.
        for token in re.findall(r"[a-z0-9]+", answer_lower):
            if token in affirmatives:
                return yes_token
            if token in negatives:
                return no_token

        return answer

    def _format_list(self, answer: str) -> str:
        """Clean up list formatting."""
        # Remove bullet points and numbering
        lines = answer.strip().split("\n")
        items = []
        for line in lines:
            cleaned = re.sub(r'^\s*[-*\d.)\]]+\s*', '', line).strip()
            if cleaned:
                items.append(cleaned)
        return ", ".join(items) if items else answer

    def _strip_markdown(self, answer: str) -> str:
        """Remove markdown formatting artifacts."""
        # Remove code blocks
        answer = re.sub(r'```[\s\S]*?```', lambda m: m.group().strip('`').strip(), answer)
        # Remove inline code
        answer = re.sub(r'`([^`]+)`', r'\1', answer)
        # Remove bold/italic
        answer = re.sub(r'\*\*([^*]+)\*\*', r'\1', answer)
        answer = re.sub(r'\*([^*]+)\*', r'\1', answer)
        return answer.strip()


def _iter_balanced_substrings(text: str):
    """
    Yield every balanced {..} or [..] substring of text.

    Depth-counting scanner, not regex. Correctly handles nested braces and
    ignores braces that appear inside JSON string literals (so `{"a":"}"}`
    parses as one object, not two). Does not validate JSON; callers still
    try json.loads() on each yielded candidate.
    """
    openers = {"{": "}", "[": "]"}
    closers = {"}", "]"}

    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in openers:
            match_close = openers[ch]
            depth = 1
            j = i + 1
            in_string = False
            escape = False
            while j < n and depth > 0:
                cj = text[j]
                if in_string:
                    if escape:
                        escape = False
                    elif cj == "\\":
                        escape = True
                    elif cj == '"':
                        in_string = False
                elif cj == '"':
                    in_string = True
                elif cj in openers:
                    depth += 1
                elif cj in closers:
                    depth -= 1
                    if depth == 0 and cj != match_close:
                        # Mismatched close (e.g. { ... ] ); bail out of this start.
                        break
                j += 1
            if depth == 0:
                yield text[i:j]
        i += 1
