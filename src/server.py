"""
Spatial Atlas: spatial-aware research agent (A2A server)

Compute-grounded reasoning agent for AgentX-AgentBeats Phase 2 Sprint 2.
Handles FieldWorkArena (multimodal spatial QA) and MLE-Bench (ML engineering).

Its argument parsing and A2A server setup started from the RDI Foundation's AgentBeats agent
template. NOTICE gives the details.
"""

import argparse
import asyncio
import hmac
import logging
import os

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from dotenv import load_dotenv
from starlette.requests import Request
from starlette.responses import HTMLResponse, PlainTextResponse
from starlette.routing import Route

from config import Config
from executor import Executor

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("spatial-atlas")


class RequestSizeLimitMiddleware:
    """Validate and buffer bounded requests before A2A parsing starts."""

    def __init__(self, app, max_bytes: int):
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                response = PlainTextResponse("Invalid Content-Length", status_code=400)
                await response(scope, receive, send)
                return
            if declared > self.max_bytes:
                response = PlainTextResponse("Request body too large", status_code=413)
                await response(scope, receive, send)
                return

        received = 0
        chunks: list[bytes] = []
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                return
            if message.get("type") != "http.request":
                continue
            body = message.get("body", b"")
            received += len(body)
            if received > self.max_bytes:
                response = PlainTextResponse("Request body too large", status_code=413)
                await response(scope, receive, send)
                return
            chunks.append(body)
            if not message.get("more_body", False):
                break

        payload = b"".join(chunks)
        replayed = False

        async def replay_receive():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": payload, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)


class BearerAuthMiddleware:
    """Require a configured bearer token for non-read-only public requests."""

    def __init__(self, app, token: str | None):
        normalized = (token or "").strip()
        if normalized and len(normalized) < 32:
            raise ValueError("ATLAS_BEARER_TOKEN must contain at least 32 characters")
        self.app = app
        self.token = normalized

    async def __call__(self, scope, receive, send):
        if (
            self.token
            and scope.get("type") == "http"
            and scope.get("method", "GET").upper() not in {"GET", "HEAD", "OPTIONS"}
        ):
            headers = {key.lower(): value for key, value in scope.get("headers", [])}
            # Compare bytes, because hmac.compare_digest raises TypeError on non-ASCII str.
            supplied = headers.get(b"authorization", b"")
            if not hmac.compare_digest(supplied, b"Bearer " + self.token.encode("utf-8")):
                response = PlainTextResponse("Unauthorized", status_code=401)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


class ConcurrencyLimitMiddleware:
    """Reject excess concurrent requests instead of building an unbounded queue."""

    def __init__(self, app, maximum: int):
        if maximum <= 0:
            raise ValueError("maximum must be positive")
        self.app = app
        self.maximum = maximum
        self._active = 0
        self._lock = asyncio.Lock()

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        # Read-only requests do no model work, so they must not consume a slot.
        # A QR code on a poster produces exactly this shape: a burst of GET traffic
        # to the landing page and the agent card. Counting those against the limit
        # meant the fifth simultaneous visitor received HTTP 503 rather than a page,
        # which is the one failure mode a conference demo cannot afford. Mirrors the
        # exemption BearerAuthMiddleware already applies to the same method set.
        if scope.get("method", "GET").upper() in {"GET", "HEAD", "OPTIONS"}:
            await self.app(scope, receive, send)
            return
        async with self._lock:
            if self._active >= self.maximum:
                response = PlainTextResponse("Server is busy", status_code=503)
                await response(scope, receive, send)
                return
            self._active += 1
        try:
            await self.app(scope, receive, send)
        finally:
            async with self._lock:
                self._active -= 1


def _request_limit_bytes() -> int:
    """Read the public HTTP body limit from the environment."""
    raw = os.environ.get("ATLAS_MAX_REQUEST_BYTES", str(64 * 1024 * 1024))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("ATLAS_MAX_REQUEST_BYTES must be an integer") from exc
    if value <= 0:
        raise ValueError("ATLAS_MAX_REQUEST_BYTES must be positive")
    return value


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _validate_execution_security(bearer_token: str) -> None:
    if _env_enabled("ATLAS_ENABLE_MLEBENCH_CODE_EXECUTION") and (
        not _env_enabled("ATLAS_TRUSTED_ISOLATED_WORKER") or not bearer_token
    ):
        raise RuntimeError(
            "Enabled MLE code execution requires isolated-worker attestation and ATLAS_BEARER_TOKEN"
        )


def _validate_public_binding(host: str, bearer_token: str) -> None:
    loopback_hosts = {"127.0.0.1", "::1", "localhost"}
    if (
        host not in loopback_hosts
        and not bearer_token
        and not _env_enabled("ATLAS_ALLOW_UNAUTHENTICATED_PUBLIC")
    ):
        raise RuntimeError(
            "A non-loopback server requires ATLAS_BEARER_TOKEN or the explicit "
            "ATLAS_ALLOW_UNAUTHENTICATED_PUBLIC=true test-only override"
        )


def _build_protected_app(app, bearer_token: str):
    """Apply admission control before authentication and bounded body buffering."""
    size_limited_app = RequestSizeLimitMiddleware(app, _request_limit_bytes())
    authenticated_app = BearerAuthMiddleware(size_limited_app, bearer_token)
    return ConcurrencyLimitMiddleware(
        authenticated_app,
        _positive_int_env("ATLAS_MAX_CONCURRENT_REQUESTS", 4),
    )


def _resolve_public_url(card_url_arg: str | None, host: str, port: int) -> str:
    """
    Pick the URL to advertise in the AgentCard.

    Priority (highest first):
      1. --card-url CLI arg (explicit override)
      2. PUBLIC_URL env var (operator-set, any deploy target)
      3. SPACE_HOST env var (auto-set by Hugging Face Spaces, e.g.
         '<owner>-<space>.hf.space') which we turn into https://...
      4. Fallback: http://{host}:{port}/ for local dev

    Motivation: when the Space boots via the Dockerfile entrypoint, no
    --card-url is passed, so the old fallback published the internal bind
    address (http://0.0.0.0:9019/) in the agent card. Green agents following
    that URL got connection refused.
    """
    if card_url_arg:
        return card_url_arg

    public_url = os.environ.get("PUBLIC_URL")
    if public_url:
        return public_url if public_url.endswith("/") else public_url + "/"

    space_host = os.environ.get("SPACE_HOST")
    if space_host:
        return f"https://{space_host}/"

    return f"http://{host}:{port}/"


def main():
    parser = argparse.ArgumentParser(description="Run Spatial Atlas spatial-aware research agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    bearer_token = os.environ.get("ATLAS_BEARER_TOKEN", "").strip()
    _validate_execution_security(bearer_token)
    _validate_public_binding(args.host, bearer_token)

    public_url = _resolve_public_url(args.card_url, args.host, args.port)

    # Validate + log the resolved model tier map BEFORE any request can
    # arrive. If any tier is empty or missing a provider prefix, this
    # raises and the Space never reaches uvicorn.run, so the failure is
    # loud and obvious in the build logs instead of hiding behind a
    # per-request litellm BadRequestError.
    Config().log_resolved_tiers()

    skills = [
        AgentSkill(
            id="fieldwork-research",
            name="Multimodal Field Research",
            description=(
                "Analyzes factory, warehouse, and retail environments from images, "
                "videos, PDFs, and documents. Spatial reasoning with structured scene "
                "graphs, safety inspection, and formatted reporting."
            ),
            tags=["spatial", "multimodal", "vision", "fieldwork", "research"],
            examples=["Analyze warehouse layout for safety violations"],
        ),
        AgentSkill(
            id="ml-engineering",
            name="ML Engineering",
            description=(
                "Generates a Kaggle-style ML pipeline and a submission file. "
                "Generated-code execution is disabled by default and needs an isolated worker."
            ),
            tags=["ml", "kaggle", "data-science", "code-generation"],
            examples=["Train a model for the spaceship-titanic competition"],
        ),
    ]

    agent_card = AgentCard(
        name="Spatial Atlas",
        description=(
            "Spatial-aware research agent built on compute-grounded reasoning (CGR). "
            "Code computes selected facts over explicit spatial representations for "
            "field work analysis. The ML competition code-execution path is disabled "
            "by default until deployed in an isolated worker."
        ),
        url=public_url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=skills,
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )

    async def landing_page(request: Request) -> HTMLResponse:
        skills_html = "".join(
            f"<li><strong>{s.name}</strong>: {s.description}</li>" for s in skills
        )
        html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spatial Atlas</title>
<style>
  :root {{ --purple: #7c3aed; --indigo: #4f46e5; --green: #22c55e; --bg: #fafafa; --card: #fff; --text: #1a1a1a; --muted: #64748b; --border: #e2e8f0; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 900px; margin: 0 auto; padding: 2rem 1.5rem; color: var(--text); background: var(--bg); line-height: 1.6; }}
  h1 {{ font-size: 2rem; margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1.35rem; margin: 2rem 0 0.75rem; color: var(--purple); border-bottom: 2px solid var(--border); padding-bottom: 0.3rem; }}
  h3 {{ font-size: 1.1rem; margin: 1.25rem 0 0.5rem; }}
  p {{ margin-bottom: 0.75rem; }}
  a {{ color: var(--indigo); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .hero {{ background: linear-gradient(135deg, var(--purple), var(--indigo)); color: #fff; padding: 2rem; border-radius: 12px; margin-bottom: 2rem; }}
  .hero h1 {{ color: #fff; font-size: 2.2rem; }}
  .hero p {{ color: #e0e0ff; margin-bottom: 0.5rem; }}
  .hero a {{ color: #c4b5fd; }}
  .badge {{ display: inline-block; background: var(--green); color: #fff; padding: 2px 10px; border-radius: 4px; font-size: 0.85rem; font-weight: 600; }}
  .badges {{ display: flex; gap: 0.5rem; margin: 0.75rem 0; flex-wrap: wrap; }}
  .badges a {{ background: var(--border); color: var(--text); padding: 3px 10px; border-radius: 4px; font-size: 0.8rem; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem; margin-bottom: 1rem; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
  @media (max-width: 640px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  .grid .card h3 {{ margin-top: 0; color: var(--purple); }}
  table {{ width: 100%; border-collapse: collapse; margin: 0.75rem 0; font-size: 0.9rem; }}
  th, td {{ padding: 0.5rem 0.75rem; border: 1px solid var(--border); text-align: left; }}
  th {{ background: #f1f5f9; font-weight: 600; }}
  tr:nth-child(even) {{ background: #f8fafc; }}
  pre {{ background: #1e293b; color: #e2e8f0; padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem; line-height: 1.5; margin: 0.75rem 0; }}
  code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 3px; font-size: 0.88rem; }}
  pre code {{ background: none; padding: 0; }}
  ul {{ padding-left: 1.4rem; margin-bottom: 0.75rem; }}
  li {{ margin-bottom: 0.3rem; }}
  .endpoint-list {{ list-style: none; padding: 0; }}
  .endpoint-list li {{ background: var(--card); border: 1px solid var(--border); padding: 0.6rem 1rem; border-radius: 6px; margin-bottom: 0.5rem; }}
  .footer {{ margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid var(--border); color: var(--muted); font-size: 0.85rem; text-align: center; }}
</style>
</head><body>

<div class="hero">
  <h1>Spatial Atlas</h1>
  <p><span class="badge">v{agent_card.version}</span> &nbsp; Spatial-aware research agent built on compute-grounded reasoning</p>
  <p>AgentX-AgentBeats Phase 2, Sprint 2 &middot; Research Agent Track</p>
  <div class="badges">
    <a href="https://github.com/arunshar/spatial-atlas-agent">GitHub</a>
    <a href="/.well-known/agent-card.json">Agent Card</a>
    <a href="https://google.github.io/A2A/">A2A Protocol</a>
  </div>
</div>

<p><strong>Spatial Atlas</strong> implements <em>compute-grounded reasoning</em> (CGR): compute what can be computed deterministically, then let LLMs reason only about what must be generated. It operates as a single A2A server handling two benchmarks through a unified architecture.</p>

<h2>Benchmarks</h2>
<table>
  <tr><th>Benchmark</th><th>What</th><th>Input</th><th>Output</th></tr>
  <tr><td><strong>FieldWorkArena</strong></td><td>Multimodal spatial QA (factory, warehouse, retail)</td><td>Text + images, PDFs, videos</td><td>Formatted answer</td></tr>
  <tr><td><strong>MLE-Bench</strong></td><td>Kaggle-style ML competition tasks</td><td>Instructions + competition data</td><td>submission.csv</td></tr>
</table>

<h2>Skills</h2>
<ul>{skills_html}</ul>

<h2>Architecture</h2>
<pre><code>+--------------------------------------------------+
|            A2A Protocol Server                    |
+--------------------------------------------------+
                     |
              +------v------+
              |   Domain    |
              | Classifier  |
              +------+------+
              /              \\
   (goal format)          (tar.gz)
        /                      \\
+------v------+        +-------v------+
| FieldWork-  |        |  MLE-Bench   |
| Arena       |        |  Handler     |
| Handler     |        |              |
+------+------+        +-------+------+
       |                       |
+------v------+        +-------v------+
| Spatial     |        | Fail-Closed  |
| Scene Graph |        | ML Pipeline  |
| Engine      |        |              |
+------+------+        +-------+------+
       \\                      /
        \\                    /
   +-----v--------------------v-----+
   | Shared Infrastructure          |
   | LiteLLM | 4 model tiers  |     |
   | Cost Tracking                  |
   +---------------+----------------+
                   |
   +---------------v----------------+
   | Self-grade, at most 1 refine   |
   +--------------------------------+</code></pre>

<h2>Key Innovations</h2>
<div class="grid">
  <div class="card">
    <h3>1. Spatial Scene Graphs</h3>
    <p>Extract entities from vision descriptions, build a queryable graph with typed relations, then compute distances and constraint violations by arithmetic rather than by generation, and feed the computed facts to the LLM.</p>
    <p><strong>Status:</strong> implemented. In this path the positions are model-estimated, so the arithmetic is exact over estimated coordinates. No score is reported.</p>
  </div>
  <div class="card">
    <h3>2. Self-Graded Refinement</h3>
    <p>Reflection is gated on a confidence score attached to each candidate answer, so a second pass is spent only where the first answer looks weak.</p>
    <p><strong>Status:</strong> implemented as a prompting heuristic. The score is self-reported and has not been demonstrated to be calibrated. Confidence-based model routing is designed but not yet wired, and accuracy and cost effects are not yet claimed.</p>
  </div>
  <div class="card">
    <h3>3. Fail-Closed ML Pipeline</h3>
    <p>Strategy-aware code generation with error detection, diagnosis, and a bounded number of repair attempts. Covers tabular, NLP, vision, time series, and general strategies.</p>
    <p><strong>Safety:</strong> generated-code execution is disabled by default and requires execution opt-in and isolated-worker attestation. These controls are defense in depth, not a complete security sandbox.</p>
  </div>
  <div class="card">
    <h3>4. Score-Driven Refinement</h3>
    <p>Parses validation scores from pipeline output, uses the configured strong model to propose targeted improvements, and keeps whichever submission scores higher.</p>
    <p><strong>Status:</strong> implemented; no improvement rate is claimed without sealed artifacts.</p>
  </div>
  <div class="card">
    <h3>5. Leak Audit Registry</h3>
    <p>Adds code-generation guidance for checking four leakage patterns: ID overlap, row fingerprinting, temporal ordering, and byte hashing. It is prompt guidance, not a standalone detector.</p>
  </div>
  <div class="card">
    <h3>6. Named Model Tiers</h3>
    <p><strong>Fast</strong>: GPT-4.1-mini (answer self-grade, metric object phrases). <strong>Standard</strong>: GPT-4.1 (MLE data analysis). <strong>Strong</strong>: GPT-4.1 (scene extraction, answer, refinement, MLE code generation). <strong>Vision</strong>: GPT-4.1 (image and video descriptions). Each tier is configurable, and handler choice uses fixed string rules with no model call.</p>
  </div>
</div>

<h2>Evaluation Status</h2>
<div class="card">
  <p><strong>FieldWorkArena:</strong> no result is reported because the benchmark data were gated and inaccessible.</p>
  <p><strong>QSpatial:</strong> one private label-free smoke ran four paths with protected labels sealed. No score or paired comparison was computed, so no result is reported.</p>
  <p><strong>MLE-Bench:</strong> no aggregate performance, cost, or latency result is reported without a sealed end-to-end run artifact.</p>
</div>

<h2>Endpoints</h2>
<ul class="endpoint-list">
  <li><strong>GET</strong> <a href="/.well-known/agent-card.json"><code>/.well-known/agent-card.json</code></a>: Agent card (identity, skills, capabilities)</li>
  <li><strong>POST</strong> <code>/</code>: A2A JSON-RPC task submission</li>
</ul>

<h2>Quick Start</h2>
<pre><code>git clone https://github.com/arunshar/spatial-atlas-agent.git
cd spatial-atlas-agent
cp sample.env .env   # add your OPENAI_API_KEY
uv run src/server.py --host 127.0.0.1 --port 9019
curl http://localhost:9019/.well-known/agent-card.json</code></pre>

<div class="footer">
  <p><strong>Spatial Atlas</strong> &middot; Arun Sharma &middot; University of Minnesota, Twin Cities</p>
  <p>Built for Berkeley RDI AgentX-AgentBeats Competition</p>
  <p><a href="https://github.com/arunshar/spatial-atlas-agent">GitHub</a> &middot; <a href="https://github.com/arunshar/spatial-atlas-agent/blob/main/paper/spatial_atlas.pdf">Paper</a> &middot; <a href="https://github.com/arunshar/spatial-atlas-agent/blob/main/docs/GUIDE.md">Guide</a></p>
</div>

</body></html>"""
        return HTMLResponse(html)

    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    print("=" * 60)
    print("Spatial Atlas -- Spatial-Aware Research Agent")
    print("=" * 60)
    print(f"Server: http://{args.host}:{args.port}/")
    print(f"Agent Card: {agent_card.url.rstrip('/')}/.well-known/agent-card.json")
    print()
    print("Skills:")
    for skill in skills:
        print(f"  - {skill.name}: {skill.description[:80]}...")
    print("=" * 60)

    starlette_app = a2a_app.build()
    starlette_app.routes.insert(0, Route("/", landing_page, methods=["GET"]))
    protected_app = _build_protected_app(starlette_app, bearer_token)

    uvicorn.run(
        protected_app,
        host=args.host,
        port=args.port,
        timeout_keep_alive=300,
    )


if __name__ == "__main__":
    main()
