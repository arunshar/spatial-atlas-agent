"""
Spatial Atlas: token and cost budget tracker

Tracks cumulative token usage and estimated cost across LLM calls.
"""

import hashlib
import logging
import threading
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("spatial-atlas.cost")
_RESPONSE_ID_HASH_DOMAIN = b"spatial-atlas-llm-response-id-v1\0"


@dataclass
class UsageStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    num_calls: int = 0
    estimated_cost_usd: float = 0.0


class CostTracker:
    def __init__(self, max_tokens: int = 150_000):
        self.max_tokens = max_tokens
        self.stats = UsageStats()
        self._lock = threading.Lock()
        self._call_records: list[dict[str, Any]] = []

    def track(
        self,
        response: Any,
        *,
        role: str,
        model_tier: str,
        call_kind: str,
        configured_max_tokens: int,
    ) -> None:
        """Track aggregate usage plus a sanitized record for one provider response."""
        role = self._nonempty_metadata(role, "role")
        model_tier = self._nonempty_metadata(model_tier, "model_tier")
        call_kind = self._nonempty_metadata(call_kind, "call_kind")
        if isinstance(configured_max_tokens, bool) or not isinstance(configured_max_tokens, int):
            raise TypeError("configured_max_tokens must be an integer")
        if configured_max_tokens <= 0:
            raise ValueError("configured_max_tokens must be positive")

        usage = self._field(response, "usage")
        prompt = self._token_count(self._field(usage, "prompt_tokens"))
        completion = self._token_count(self._field(usage, "completion_tokens"))
        provider_total = self._optional_token_count(self._field(usage, "total_tokens"))
        total = prompt + completion
        recorded_total = total if provider_total is None else provider_total
        completion_details = self._field(usage, "completion_tokens_details")
        reasoning = self._token_count(self._field(completion_details, "reasoning_tokens"))

        choices = self._field(response, "choices", [])
        choice = choices[0] if choices else None
        message = self._field(choice, "message")
        finish_reason_value = self._field(choice, "finish_reason")
        finish_reason = self._safe_finish_reason(finish_reason_value)
        response_id_value = self._field(response, "id")
        if response_id_value is not None and (
            not isinstance(response_id_value, str) or not response_id_value
        ):
            raise ValueError("provider response ID must be a nonempty string or null")
        response_id_sha256 = (
            None
            if response_id_value is None
            else hashlib.sha256(
                _RESPONSE_ID_HASH_DOMAIN + response_id_value.encode("utf-8")
            ).hexdigest()
        )
        hidden_params = self._field(response, "_hidden_params", {})
        cost = self._cost(self._field(hidden_params, "response_cost"))
        record = {
            "schema_version": 1,
            "role": role,
            "model_tier": model_tier,
            "call_kind": call_kind,
            "configured_max_tokens": configured_max_tokens,
            "provider_finish_reason": finish_reason,
            "token_usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": recorded_total,
                "reasoning_tokens": reasoning,
            },
            "response_id_sha256": response_id_sha256,
            "message_content_empty": self._content_is_empty(self._field(message, "content")),
            "reasoning_content_empty": self._content_is_empty(
                self._field(message, "reasoning_content")
            ),
        }

        with self._lock:
            self.stats.prompt_tokens += prompt
            self.stats.completion_tokens += completion
            self.stats.total_tokens += total
            self.stats.num_calls += 1
            self.stats.estimated_cost_usd += cost
            self._call_records.append(record)
            num_calls = self.stats.num_calls
            cumulative_tokens = self.stats.total_tokens

        logger.debug(
            "Call #%d: +%d tokens (total: %d/%d)",
            num_calls,
            total,
            cumulative_tokens,
            self.max_tokens,
        )

    def telemetry_cursor(self) -> int:
        """Return a stable cursor for later per-sample telemetry slicing."""
        with self._lock:
            return len(self._call_records)

    def telemetry_since(self, cursor: int) -> list[dict[str, Any]]:
        """Return detached sanitized records written at or after ``cursor``."""
        if isinstance(cursor, bool) or not isinstance(cursor, int):
            raise TypeError("telemetry cursor must be an integer")
        if cursor < 0:
            raise ValueError("telemetry cursor must be nonnegative")
        with self._lock:
            if cursor > len(self._call_records):
                raise ValueError("telemetry cursor is beyond the current record count")
            return deepcopy(self._call_records[cursor:])

    @staticmethod
    def _field(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    @staticmethod
    def _token_count(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 0
        return max(0, int(value))

    @classmethod
    def _optional_token_count(cls, value: Any) -> int | None:
        if value is None:
            return None
        return cls._token_count(value)

    @staticmethod
    def _cost(value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 0.0
        return max(0.0, float(value))

    @staticmethod
    def _content_is_empty(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        try:
            return len(value) == 0
        except TypeError:
            return False

    @staticmethod
    def _nonempty_metadata(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a nonempty string")
        return value.strip()

    @staticmethod
    def _safe_finish_reason(value: Any) -> str | None:
        if value is None:
            return None
        if (
            not isinstance(value, str)
            or not 1 <= len(value) <= 64
            or not value.isascii()
            or any(not (character.isalnum() or character in "_.:-") for character in value)
        ):
            raise ValueError("provider finish_reason is not a safe bounded token")
        return value

    def has_budget(self) -> bool:
        """Check if we're within token budget."""
        return self.stats.total_tokens < self.max_tokens

    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.stats.total_tokens)

    def summary(self) -> str:
        return (
            f"Calls: {self.stats.num_calls}, "
            f"Tokens: {self.stats.total_tokens:,}/{self.max_tokens:,}, "
            f"Cost: ${self.stats.estimated_cost_usd:.4f}"
        )
