import ipaddress
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCS = (
    "README.md",
    "paper/spatial_atlas.md",
    "paper/spatial_atlas.tex",
)

# Retired result titles are assembled at run time, so a plain grep of the repository for the
# retired wording finds no match inside this guard file.
RETIRED_TITLES = tuple(
    "".join(parts)
    for parts in (
        ("FieldWorkArena ", "Accuracy"),
        ("MLE-Bench ", "Performance"),
        ("82% ", "valid"),
        ("82\\% ", "valid"),
        ("32% ", "medal"),
        ("32\\% ", "medal"),
        ("Self-Healing ", "ML Pipeline"),
    )
)


def _markdown_section(document: str, heading: str) -> str:
    """Return one second-level Markdown section without executing embedded commands."""
    marker = f"## {heading}"
    start = document.index(marker)
    end = document.find("\n## ", start + len(marker))
    return document[start:] if end == -1 else document[start:end]


@pytest.mark.unit
def test_ci_does_not_pass_repository_secrets_to_test_container():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()

    assert "toJson(secrets)" not in workflow
    assert "SECRETS_JSON" not in workflow
    assert "--env-file .env" not in workflow


@pytest.mark.unit
def test_non_loopback_launch_docs_require_bearer_token():
    readme = (ROOT / "README.md").read_text()
    docker_section = _markdown_section(readme, "Docker")

    assert "--host 0.0.0.0" in docker_section
    assert "ATLAS_BEARER_TOKEN" in docker_section
    assert "at least 32 characters" in docker_section
    assert "Authorization: Bearer <ATLAS_BEARER_TOKEN>" in docker_section
    assert "ATLAS_ALLOW_UNAUTHENTICATED_PUBLIC" not in docker_section


@pytest.mark.unit
def test_sample_env_documents_bearer_token_without_a_secret():
    sample_env = (ROOT / "sample.env").read_text()
    assignments = [
        line
        for line in sample_env.splitlines()
        if line.startswith("ATLAS_BEARER_TOKEN=")
    ]

    assert assignments == ["ATLAS_BEARER_TOKEN="]
    assert "non-loopback" in sample_env
    assert "at least 32 characters" in sample_env
    assert "Never commit the real value" in sample_env


@pytest.mark.unit
def test_hf_space_instructions_require_both_secrets():
    deploy_script = (ROOT / "deploy_to_hf.sh").read_text()
    preamble = deploy_script.split("set -e", 1)[0]
    space_readme = deploy_script.split("<< 'SPACE_README'", 1)[1].split(
        "\nSPACE_README", 1
    )[0]
    terminal_reminder = deploy_script.split('echo "=== Done! ==="', 1)[1]

    for instructions in (preamble, space_readme, terminal_reminder):
        normalized = " ".join(instructions.replace("#", " ").split())
        assert "OPENAI_API_KEY" in normalized
        assert "ATLAS_BEARER_TOKEN" in normalized
        assert "at least 32 characters" in normalized
    assert "Authorization: Bearer <ATLAS_BEARER_TOKEN>" in space_readme
    assert "ATLAS_ALLOW_UNAUTHENTICATED_PUBLIC" not in deploy_script


@pytest.mark.unit
def test_generated_junit_report_is_ignored():
    ignored_entries = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "junit.xml" in ignored_entries


@pytest.mark.unit
@pytest.mark.parametrize("paper_path", ["paper/spatial_atlas.md", "paper/spatial_atlas.tex"])
def test_paper_marks_fieldworkarena_as_unevaluated(paper_path):
    paper = (ROOT / paper_path).read_text()

    assert "FieldWorkArena remained gated and inaccessible" in paper
    for unsupported_claim in (
        "Full System (SSG + EG + F2)",
        "45,200",
        "21--24 percentage point improvement",
        "achieving an 82% " + "valid submission rate",
        "achieving an 82\\% " + "valid submission rate",
    ):
        assert unsupported_claim not in paper


@pytest.mark.unit
@pytest.mark.parametrize("doc_path", PUBLIC_DOCS)
def test_public_docs_match_fail_closed_mle_runtime(doc_path):
    document = (ROOT / doc_path).read_text()
    normalized = document.replace("\\_", "_").lower()

    assert "\u2014" not in document
    assert "at most 3 total attempts" in normalized or "3 total attempts" in normalized
    assert "600-second" in normalized or "600s" in normalized
    assert "dummy" in normalized and "disabled" in normalized
    # TeX sources write the thousands separator as "{,}", so strip that form as well.
    assert "150000" in normalized.replace("_", "").replace("{,}", "").replace(",", "")
    assert "public a2a" in normalized
    assert "concurrency-safe" in normalized
    assert "heuristic" in normalized
    assert "one a2a execution" in normalized or "per-execution" in normalized
    assert "estimate" in normalized and "prompt" in normalized
    assert "maximum completion" in normalized
    assert "not exact tokenizer" in normalized
    assert "frozen benchmark" in normalized
    assert "observational" in normalized or "only records usage" in normalized
    assert "artifact" in normalized
    assert "oversubscribe" in normalized or "exceed" in normalized

    assert "atlas_enable_mlebench_code_execution" in normalized
    assert "atlas_trusted_isolated_worker" in normalized
    assert "atlas_allow_dummy_submission" in normalized
    assert "atlas_bearer_token" in normalized
    assert "atlas_allow_unauthenticated_public" in normalized
    assert "32" in normalized
    assert "normal non-loopback" in normalized
    assert "loopback" in normalized and "may omit" in normalized
    assert "test-only" in normalized
    assert "64 mib" in normalized
    assert "503" in normalized
    assert "4" in normalized and ("concurrency" in normalized or "active requests" in normalized)

    for stale_claim in (
        "sandboxed subprocess",
        "safe sandbox",
        "default: 300 seconds",
        "300 seconds per attempt",
        "total of 4 attempts",
        "1 initial + 3",
    ):
        assert stale_claim not in normalized


@pytest.mark.unit
@pytest.mark.parametrize(
    "source_path",
    ["paper/spatial_atlas.md", "paper/spatial_atlas.tex", "poster/spatial_atlas_poster.tex"],
)
def test_retired_result_titles_stay_out_of_public_sources(source_path):
    text = (ROOT / source_path).read_text()

    for title in RETIRED_TITLES:
        assert title not in text


# Directories that a local checkout can hold but the public repository never ships.
_LOCAL_ONLY_DIRS = frozenset(
    {
        ".git",
        ".venv",
        ".venv-test",
        "__pycache__",
        ".pytest_cache",
        ".hypothesis",
        ".ruff_cache",
        ".mypy_cache",
        "htmlcov",
        "sessions",
    }
)
_PHONE_PATTERN = re.compile(r"\+?1?[ .-]?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}")
_IPV4_PATTERN = re.compile(r"(?<![\d.])\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?![\d.])")
# Built from integers so that this guard file holds no dotted private address of its own.
_RFC1918_NETWORKS = (
    ipaddress.IPv4Network((10 << 24, 8)),
    ipaddress.IPv4Network(((172 << 24) | (16 << 16), 12)),
    ipaddress.IPv4Network(((192 << 24) | (168 << 16), 16)),
)
# The retired source repository is private, so any link to it breaks for public readers. The
# pattern is written with escaped dots, so this file never matches itself.
_PRIVATE_REPO_PATTERN = re.compile(
    r"github\.com/arunshar/spatial-atlas(?!-agent)|raw\.githubusercontent\.com/arunshar/spatial-atlas/"
)
# Assembled at run time so that this guard file does not contain the scheme it forbids.
_TEL_SCHEME = "tel" + ":"


def _shipped_text_files():
    """Yield (path, text) for every UTF-8 file in the repository except uv.lock."""
    for directory, subdirectories, filenames in os.walk(ROOT):
        subdirectories[:] = sorted(d for d in subdirectories if d not in _LOCAL_ONLY_DIRS)
        for filename in sorted(filenames):
            if filename in {"uv.lock", ".coverage"}:
                continue
            path = Path(directory) / filename
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            yield path.relative_to(ROOT), text


def _visible_text(relative: Path, text: str) -> str:
    """Return the text a reader sees. SVG path coordinates look like phone numbers, so an SVG
    contributes only the content of its text elements."""
    if relative.suffix == ".svg":
        return "\n".join(re.findall(r"<text[^>]*>([^<]*)</text>", text))
    return text


@pytest.mark.unit
def test_shipped_text_has_no_phone_number_or_tel_link():
    offenders = []
    for relative, text in _shipped_text_files():
        visible = _visible_text(relative, text)
        if _TEL_SCHEME in visible:
            offenders.append(f"{relative}: {_TEL_SCHEME} link")
        if _PHONE_PATTERN.search(visible):
            offenders.append(f"{relative}: phone-number pattern")

    assert offenders == []


@pytest.mark.unit
def test_shipped_text_has_no_private_network_address():
    offenders = []
    for relative, text in _shipped_text_files():
        for match in _IPV4_PATTERN.finditer(text):
            try:
                address = ipaddress.ip_address(match.group())
            except ValueError:
                continue
            if any(address in network for network in _RFC1918_NETWORKS):
                offenders.append(f"{relative}: RFC 1918 address")

    assert offenders == []


@pytest.mark.unit
def test_shipped_text_does_not_link_the_private_source_repository():
    offenders = [
        str(relative)
        for relative, text in _shipped_text_files()
        if _PRIVATE_REPO_PATTERN.search(text)
    ]

    assert offenders == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "pdf_path", ["paper/spatial_atlas.pdf", "poster/spatial_atlas_poster.pdf"]
)
def test_shipped_pdfs_have_no_phone_link_or_private_repository_link(pdf_path):
    from pypdf import PdfReader

    reader = PdfReader(ROOT / pdf_path)
    uris = []
    for page in reader.pages:
        for annotation in page.get("/Annots") or []:
            action = annotation.get_object().get("/A")
            if action is not None and "/URI" in action:
                uris.append(str(action["/URI"]))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    assert not [uri for uri in uris if uri.lower().startswith(_TEL_SCHEME)]
    assert not [uri for uri in uris if _PRIVATE_REPO_PATTERN.search(uri)]
    assert "Phone:" not in text
    assert not _PRIVATE_REPO_PATTERN.search(text)

    # A rebuilt PDF can carry a local path, a cluster name, a job ID, or a private digest in its
    # page text or its document metadata, so both get the same scan as the shipped text files.
    metadata = reader.metadata or {}
    metadata_text = "\n".join(f"{key} {value}" for key, value in metadata.items())
    for scanned in (text, metadata_text):
        assert not [p.pattern[:12] for p in _CREDENTIAL_PATTERNS if p.search(scanned)]
        assert not _has_cluster_word(scanned)
        assert not _HEX64_PATTERN.search(scanned)
    assert "/PTEX.FileName" not in metadata


# File names that must never sit in the repository tree. The cluster batch-script suffix is
# assembled at run time, so this guard file does not match its own cluster-word scan.
_NEVER_SHIP_FILE_PATTERNS = (
    "build_p5_assets.py",
    "build_p3_table.py",
    "*HANDOFF*.md",
    "*." + "sb" + "atch",
    "qspatial_pilot_common.sh",
    "*.pem",
    "*.key",
    "*.tfstate*",
    "*.tfvars",
    "*.tfplan",
)
# The deploy script refuses these names too. A local checkout may legitimately hold .env or a
# log file, so only the deploy check covers them.
_NEVER_SHIP_NAME_PATTERNS = (
    "*.pem",
    "*.key",
    "*.p12",
    "id_rsa*",
    ".netrc",
    "secrets.*",
    "credentials.*",
    "*.tfstate*",
    "*.tfvars",
    "*.tfplan",
    "*." + "sb" + "atch",
    "*HANDOFF*",
    "qspatial_pilot_common.sh",
    "build_p5_assets.py",
    "build_p3_table.py",
    "*.log",
    "sessions",
    ".claude",
    ".env",
    ".env.*",
)


@pytest.mark.unit
def test_deploy_script_never_copies_env_files():
    deploy_script = (ROOT / "deploy_to_hf.sh").read_text()
    forbidden_loop = next(
        line for line in deploy_script.splitlines() if line.startswith("for forbidden in ")
    )
    forbidden_names = forbidden_loop.removeprefix("for forbidden in ").split(";", 1)[0].split()

    # The script copies only committed files, and a second check refuses any local-only name.
    assert "git -C \"$REPO_DIR\" archive --format=tar HEAD" in deploy_script
    assert "rsync" not in deploy_script
    assert ".env" in forbidden_names
    for pattern in _NEVER_SHIP_NAME_PATTERNS:
        assert f"-name '{pattern}'" in deploy_script, pattern
    # eval_bench.py and the figures keep terms that the Space's MIT card does not cover.
    for excluded in ("eval_bench.py", "assets", "tests", "paper", "poster"):
        assert f"':(exclude){excluded}'" in deploy_script
    # A Space URL with embedded credentials would be printed and stored, so it is refused.
    assert "*@*)" in deploy_script


@pytest.mark.unit
def test_repository_holds_no_never_ship_file_names():
    from fnmatch import fnmatch

    offenders = []
    for directory, subdirectories, filenames in os.walk(ROOT):
        subdirectories[:] = sorted(d for d in subdirectories if d not in _LOCAL_ONLY_DIRS)
        for name in filenames:
            if any(fnmatch(name, pattern) for pattern in _NEVER_SHIP_FILE_PATTERNS):
                offenders.append(str((Path(directory) / name).relative_to(ROOT)))

    assert offenders == []


def _github_slug(heading: str) -> str:
    """Approximate GitHub's heading anchor: lowercase, drop punctuation, spaces to hyphens."""
    slug = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return slug.replace(" ", "-")


def _strip_fenced_code(markdown: str) -> str:
    return re.sub(r"^```.*?^```", "", markdown, flags=re.MULTILINE | re.DOTALL)


def _anchors(markdown: str) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for heading in re.findall(r"^#{1,6} (.+)$", _strip_fenced_code(markdown), flags=re.MULTILINE):
        slug = _github_slug(heading)
        seen = counts.get(slug, 0)
        anchors.add(slug if seen == 0 else f"{slug}-{seen}")
        counts[slug] = seen + 1
    return anchors


_LINKED_DOCS = ("README.md", "docs/ARCHITECTURE.md", "docs/EXPLAINED.md", "docs/GUIDE.md")


def _exists_with_exact_case(resolved: Path) -> bool:
    """Path.exists() ignores letter case on the default macOS volume, and GitHub does not. This
    check compares each path part with the real directory listing, so a local run fails the same
    way the Linux CI job would."""
    try:
        parts = resolved.relative_to(ROOT).parts
    except ValueError:
        return resolved.exists()
    current = ROOT
    for part in parts:
        try:
            if part not in os.listdir(current):
                return False
        except OSError:
            return False
        current = current / part
    return True


@pytest.mark.unit
@pytest.mark.parametrize("doc_path", _LINKED_DOCS)
def test_relative_links_and_anchors_resolve(doc_path):
    document_path = ROOT / doc_path
    body = _strip_fenced_code(document_path.read_text())
    targets = re.findall(r"\]\(([^)\s]+)\)", body) + re.findall(r'(?:href|src)="([^"]+)"', body)
    broken = []
    for target in targets:
        if re.match(r"^[a-z][a-z0-9+.-]*:", target):
            continue
        path_part, _, anchor = target.partition("#")
        resolved = (document_path.parent / path_part).resolve() if path_part else document_path
        if not _exists_with_exact_case(resolved):
            broken.append(target)
            continue
        if anchor and resolved.suffix == ".md" and anchor not in _anchors(resolved.read_text()):
            broken.append(target)

    assert broken == []


# Credential and cluster patterns are assembled from parts, so this guard file never matches
# its own source. Each pattern names a secret shape or a private compute identifier.
_CREDENTIAL_PATTERNS = (
    re.compile("(" + "AK" + "IA|" + "AS" + "IA)[A-Z0-9]{16}"),
    re.compile("BEGIN [A-Z ]*" + "PRIVATE" + " KEY"),
    re.compile(r"\b(" + "sk|" + "hf|" + "ghp" + r")[_-][A-Za-z0-9]{20,}"),
    re.compile("/" + "Users" + "/"),
    re.compile("/" + "home/(?!agent/)"),
    re.compile("/" + "scratch"),
    re.compile(r"job[^\n]{0,20}\d{7,}", re.IGNORECASE),
)
# This guard checks only the generic scheduler words. Site-specific words belong in a local
# pre-publish scan that stays outside the repository.
_CLUSTER_WORDS = re.compile(r"\b(" + "sl" + "urm|" + "sb" + "atch" + r")\b", re.IGNORECASE)


def _has_cluster_word(text: str) -> bool:
    return bool(_CLUSTER_WORDS.search(text))


# .gitignore and the deploy script's leak check name the cluster files that must never ship,
# so they may mention them.
_CLUSTER_WORD_EXEMPT = frozenset({".gitignore", "deploy_to_hf.sh"})
_HEX64_PATTERN = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{64}(?![0-9a-fA-F])")
# Only these files may hold a 64-hex digest: the base-image pin and the golden PDF fixture hash.
_HEX64_ALLOWED = {"Dockerfile": 1, "tests/test_pdf_extraction_golden.py": 1}


@pytest.mark.unit
def test_shipped_text_has_no_credential_or_cluster_identifier():
    offenders = []
    hex64_counts: dict[str, int] = {}
    for relative, text in _shipped_text_files():
        visible = _visible_text(relative, text)
        for pattern in _CREDENTIAL_PATTERNS:
            if pattern.search(visible):
                offenders.append(f"{relative}: {pattern.pattern[:12]}")
        if str(relative) not in _CLUSTER_WORD_EXEMPT and _has_cluster_word(visible):
            offenders.append(f"{relative}: cluster identifier")
        count = len(_HEX64_PATTERN.findall(visible))
        if count:
            hex64_counts[str(relative)] = count

    assert offenders == []
    assert hex64_counts == _HEX64_ALLOWED


# A README excerpt names its source file and line range in its first line. The excerpt must
# match those exact source lines, so a later source edit cannot leave the citation stale.
_EXCERPT_HEADER = re.compile(r"^# (\S+), lines (\d+)-(\d+)$")


@pytest.mark.unit
def test_readme_code_excerpts_match_their_cited_lines():
    readme = (ROOT / "README.md").read_text()
    checked = 0
    mismatches = []
    for fence in re.findall(r"^```[^\n]*\n(.*?)^```", readme, flags=re.MULTILINE | re.DOTALL):
        lines = fence.split("\n")[:-1]
        header = _EXCERPT_HEADER.match(lines[0]) if lines else None
        if header is None:
            continue
        path, first, last = header.group(1), int(header.group(2)), int(header.group(3))
        source = (ROOT / path).read_text().split("\n")[first - 1 : last]
        checked += 1
        if lines[1:] != source:
            mismatches.append(f"{path}, lines {first}-{last}")

    assert checked > 0
    assert mismatches == []
