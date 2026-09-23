"""Regression tests for optional remote-code model provenance."""

import re
from pathlib import Path


def test_florence_remote_code_uses_immutable_revision():
    source = (
        Path(__file__).parents[1].joinpath("src/fieldwork/detector.py").read_text(encoding="utf-8")
    )

    match = re.search(r'model_revision = "([0-9a-f]{40})"', source)
    assert match is not None
    assert source.count("revision=model_revision") == 2
    assert source.count("trust_remote_code=True") == 2
