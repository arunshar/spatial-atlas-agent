"""Pins pypdf text extraction so a dependency bump cannot change it silently.

This exists because of a real miss. On 2026-09-11 the nightly audit went red on three pypdf
advisories against the pinned 6.15.0, and the obvious fix was to move the lock to 6.16.1. The
whole suite stayed green across that bump, which looked like evidence the upgrade was safe. It
was not evidence of anything. The only pypdf test at the time monkeypatched PdfReader with a
raiser, so it never reached real pypdf, and the coverage gate is scoped to fieldwork.perception,
so vision.py was not even measured.

The upgrade did change behavior. pypdf 6.16.0 altered how a kerning gap in a TJ array becomes a
space, so words that 6.15.0 separated are now run together: "image I of" became "imageI of",
"question q" became "questionq". That text is not parsed by anything. handler.py collects it into
file_contexts and hands it to spatial.build_scene(), so it reaches the model as reasoning input
and any degradation is invisible at runtime.

The behavior change is accepted rather than fixed, because 6.16.0 is the mandatory floor for
PYSEC-2026-3913 and there is no patched version that keeps the old spacing. What must not happen
again is shipping such a change without noticing it.

The fixture is page 4 of paper/spatial_atlas.pdf, carried as its own file so that rebuilding the
paper cannot break this test. It is kept because it uses an embedded subset font. A PDF using a
standard font such as Helvetica extracts identically on 6.15.0 and 6.16.1, so it would pass on
both versions and pin nothing.
"""

import hashlib
from pathlib import Path

from fieldwork.vision import VisionPipeline

FIXTURE = Path(__file__).parent / "fixtures" / "kerned_subset_font.pdf"

# Golden recorded under pypdf 6.16.1. A failure here is not automatically a bug. It means the
# extracted text moved, so diff the output against this hash's provenance and decide whether the
# new text is acceptable before updating the constant.
EXPECTED_SHA256 = "351d2ec9b2f375f67ef4a945fd8d6b02637b3b1b17fa00d88b377891b3a4ef19"

# The spacing behavior 6.16.0 introduced, asserted by name. When the golden above breaks, these
# say which part of the change moved, which a bare hash mismatch cannot.
GLUED_ON_6_16 = ("imageI", "questionq", "answera", "constraintsC")


def _extract() -> str:
    return VisionPipeline(llm=None)._process_pdf(FIXTURE.name, FIXTURE.read_bytes())


def test_pdf_extraction_matches_recorded_golden():
    """The exact bytes vision.py hands to the model must not drift unnoticed."""
    text = _extract()
    digest = hashlib.sha256(text.encode()).hexdigest()
    assert digest == EXPECTED_SHA256, (
        "pypdf text extraction changed. Compare the new output against the recorded golden and "
        "decide whether it is acceptable before updating EXPECTED_SHA256."
    )


def test_pdf_extraction_still_loses_word_boundaries():
    """Documents the accepted 6.16.x regression so a future fix is noticed too.

    If pypdf ever restores the old spacing this test fails, which is the signal to re-record the
    golden and delete this test rather than a reason to pin the old version.
    """
    text = _extract()
    still_glued = [token for token in GLUED_ON_6_16 if token in text]
    assert still_glued == list(GLUED_ON_6_16), (
        f"Word-boundary behavior changed. Still glued: {still_glued}. "
        f"Expected all of {list(GLUED_ON_6_16)}."
    )


def test_extraction_failure_is_not_silently_empty():
    """A corrupt PDF must produce the error marker, never an empty string.

    Guards the branch that matters operationally: vision.py catches Exception, so a parse failure
    degrades a whole document. That path must stay distinguishable from a genuinely empty PDF.
    """
    result = VisionPipeline(llm=None)._process_pdf("corrupt.pdf", b"%PDF-1.4 not actually a pdf")
    assert result == "[PDF: corrupt.pdf] Error: extraction failed"
