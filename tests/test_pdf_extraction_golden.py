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

On 2026-10-22 the nightly audit forced another bump, through 6.21.2. That release restores the
pre-6.16.0 spacing on this fixture, so the glued tokens are gone and the golden is re-recorded.
The behavior change is accepted because 6.21.1 was then the mandatory floor for the five
advisories that landed after 6.19.0, and there is no patched version that keeps the intermediate
glued spacing.
What must not happen again is shipping such a change without noticing it.

The fixture is page 4 of an earlier build of paper/spatial_atlas.pdf, carried as its own file so that rebuilding the
paper cannot break this test. It is kept because it uses an embedded subset font. A PDF using a
standard font such as Helvetica extracts identically across these versions, so it would pass on
all of them and pin nothing.
"""

import hashlib
from pathlib import Path

from fieldwork.vision import VisionPipeline

FIXTURE = Path(__file__).parent / "fixtures" / "kerned_subset_font.pdf"

# The golden was recorded under pypdf 6.21.2. The 2026-11-05 bump to 6.21.4 left it unchanged,
# so it was not re-recorded. A failure here is not automatically a bug. It means the extracted
# text moved, so diff the output against this hash's provenance and decide whether the new text
# is acceptable before updating the constant.
EXPECTED_SHA256 = "2e14cb3cf84480bd9138a697a9ee3c94c5b4d418e550546a3a4446877ec5735a"

# Spacing restored in 6.21.x relative to the accepted 6.16.x regression. Asserted by name so a
# future bump that glues these tokens again is a clear failure rather than a bare hash mismatch.
SPACED_ON_6_21 = ("image I", "question q", "answer a", "constraints C")
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


def test_pdf_extraction_keeps_word_boundaries():
    """Documents the restored 6.21.x spacing so a future glue regression is noticed too.

    If pypdf ever reintroduces the 6.16.x glued tokens this test fails, which is the signal to
    re-record the golden and decide whether the new text is acceptable rather than a reason to
    pin an older version.
    """
    text = _extract()
    missing_spaced = [token for token in SPACED_ON_6_21 if token not in text]
    still_glued = [token for token in GLUED_ON_6_16 if token in text]
    assert missing_spaced == [] and still_glued == [], (
        f"Word-boundary behavior changed. Missing spaced: {missing_spaced}. "
        f"Still glued: {still_glued}."
    )


def test_extraction_failure_is_not_silently_empty():
    """A corrupt PDF must produce the error marker, never an empty string.

    Guards the branch that matters operationally: vision.py catches Exception, so a parse failure
    degrades a whole document. That path must stay distinguishable from a genuinely empty PDF.
    """
    result = VisionPipeline(llm=None)._process_pdf("corrupt.pdf", b"%PDF-1.4 not actually a pdf")
    assert result == "[PDF: corrupt.pdf] Error: extraction failed"
