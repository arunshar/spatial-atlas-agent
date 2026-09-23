"""Regression tests for the leak-hint registry's failure handling.

The bug these exist for: `match_leak`'s except clause logged `entry.name`, but `LeakHint` is a
frozen dataclass whose fields are `slug`, `title`, `hint`, `detect`. There is no `name`. So a
detector that raised would trigger an `AttributeError` inside the very handler whose comment reads
"A malformed detector must never bring the pipeline down", converting a contained warning into an
uncaught exception.

It was unreachable in practice only because the sole registered detector is a pure string match that
cannot raise. That is a property of today's registry, not of the code, so it would have become
reachable the moment anyone added a detector doing real work. These tests pin the contract instead
of relying on that accident.
"""

from __future__ import annotations

import dataclasses

from src.mlebench.strategies import leaks
from src.mlebench.strategies.leaks import LeakHint, match_leak


def _exploding_hint() -> LeakHint:
    def _detect(description: str, file_listing: str) -> bool:
        raise RuntimeError("detector blew up")

    return LeakHint(
        slug="exploding-detector",
        title="Exploding detector",
        hint="never used",
        detect=_detect,
    )


def test_leakhint_has_no_name_field() -> None:
    """Pin the dataclass shape, since the bug was an attribute that never existed."""
    fields = {f.name for f in dataclasses.fields(LeakHint)}
    assert fields == {"slug", "title", "hint", "detect"}
    assert "name" not in fields


def test_raising_detector_is_contained_not_propagated(monkeypatch, caplog) -> None:
    """The whole point of the except clause: a bad detector must not escape."""
    monkeypatch.setattr(leaks, "_REGISTRY", (_exploding_hint(),))

    with caplog.at_level("WARNING"):
        result = match_leak("any description", "any listing")

    # Contained, and reported as "no match" rather than crashing the pipeline.
    assert result is None
    # getMessage() applies the args once; `record.message % record.args` double-formats.
    messages = [record.getMessage() for record in caplog.records]
    assert any("exploding-detector" in message for message in messages), messages
    # And the slug, not a nonexistent `name`, is what identifies it.
    assert any("Leak detector exploding-detector failed" in m for m in messages), messages


def test_a_raising_detector_does_not_block_a_later_match(monkeypatch) -> None:
    """First-hit-wins must survive an earlier detector raising.

    Without containment the first exception aborts the scan, so a genuinely matching later entry is
    never reached. This is the behavioural consequence of the bug, distinct from the crash itself.
    """
    good = LeakHint(
        slug="good-detector",
        title="Good detector",
        hint="the real hint",
        detect=lambda description, file_listing: True,
    )
    monkeypatch.setattr(leaks, "_REGISTRY", (_exploding_hint(), good))

    assert match_leak("d", "f") is good


def test_no_match_returns_none() -> None:
    """Baseline: the real registry does not fire on unrelated input."""
    assert match_leak("unrelated tabular competition", "train.csv test.csv") is None
