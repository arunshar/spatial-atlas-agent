"""
Spatial Atlas: local A2A smoke client.

Posts a synthetic FieldWorkArena-style question to a running Spatial Atlas
instance and asserts that it gets a non-empty formatted answer back. The client
sends no bearer token, so use it only against a loopback server that has no
token set.

Usage:
    # Smoke-test a local server (default target, text-only).
    uv run python eval_smoke.py

    # Smoke-test with an image (exercises the full vision pipeline).
    uv run python eval_smoke.py --image path/to/warehouse.jpg

    # Smoke-test a server at another address.
    uv run python eval_smoke.py --url http://127.0.0.1:9019/

Exit codes:
    0  AgentCard fetched, and the task completed with a usable text answer.
    1  Anything else (network error, bad card, a task that ended in any state
       other than completed, empty response, missing text).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import sys
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from a2a.client import A2AClient
from a2a.types import (
    AgentCard,
    FilePart,
    FileWithBytes,
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    TextPart,
)


DEFAULT_URL = "http://127.0.0.1:9019/"

# Text-only goal (no image). Exercises parser, classifier, reasoner, formatter.
SMOKE_GOAL_TEXT_ONLY = """# Question
How many fire extinguishers are visible in a typical warehouse loading dock with two workers present?

# Input Data

# Output Format
number
"""

# Goal used when --image is provided. The answer depends on actual image content.
SMOKE_GOAL_WITH_IMAGE = """# Question
Describe all safety-relevant objects visible in this image. Count each distinct type.

# Input Data
{image_name}

# Output Format
json
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Base URL of the Spatial Atlas server (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a local image (jpg/png) to attach. Exercises the full "
        "vision + spatial pipeline instead of the text-only path.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP request timeout in seconds (default: 120)",
    )
    return parser.parse_args()


def _is_internal_url(url: str) -> bool:
    """Return True for a loopback or wildcard bind address that other hosts cannot reach."""
    host = (urlsplit(url).hostname or "").lower()
    return host in {"0.0.0.0", "127.0.0.1", "::1", "localhost"}


async def _fetch_agent_card(base_url: str, timeout: float) -> AgentCard:
    """
    GET /.well-known/agent-card.json and parse it into an AgentCard.
    """
    card_url = base_url.rstrip("/") + "/.well-known/agent-card.json"
    async with httpx.AsyncClient(timeout=timeout) as http:
        resp = await http.get(card_url)
        resp.raise_for_status()
        data = resp.json()
    card = AgentCard.model_validate(data)
    print(f"[ok] AgentCard fetched: name={card.name!r}, url={card.url!r}")
    if not card.url.startswith(("http://", "https://")):
        print(f"[warn] AgentCard advertises non-URL: {card.url!r}", file=sys.stderr)
    if _is_internal_url(card.url) and not _is_internal_url(base_url):
        print(
            f"[warn] The agent card advertises an internal address: {card.url!r}. "
            "Set PUBLIC_URL when other agents must reach this server.",
            file=sys.stderr,
        )
    return card


async def _send_smoke_message(
    card: AgentCard,
    base_url: str,
    timeout: float,
    image_path: str | None = None,
) -> str:
    """
    Fire one send_message round-trip. Returns the first text block in the
    response. Raises on transport or protocol error.

    If image_path is set, sends an image-attached goal that exercises the
    full vision + spatial scene graph pipeline. Otherwise sends a text-only
    goal that validates the parser/reasoner/formatter path.
    """
    parts: list[Part] = []

    if image_path:
        img = Path(image_path)
        if not img.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        raw = img.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        mime = mimetypes.guess_type(img.name)[0] or "image/jpeg"
        parts.append(Part(root=TextPart(
            text=SMOKE_GOAL_WITH_IMAGE.format(image_name=img.name)
        )))
        parts.append(Part(root=FilePart(
            file=FileWithBytes(
                bytes=b64,
                name=img.name,
                mime_type=mime,
            )
        )))
        label = f"image-attached ({img.name}, {len(raw)} bytes)"
    else:
        parts.append(Part(root=TextPart(text=SMOKE_GOAL_TEXT_ONLY)))
        label = "text-only (no files)"

    async with httpx.AsyncClient(timeout=timeout) as http:
        client = A2AClient(httpx_client=http, agent_card=card, url=base_url)

        message = Message(
            role=Role.user,
            parts=parts,
            message_id=str(uuid.uuid4()),
        )
        params = MessageSendParams(message=message)
        request = SendMessageRequest(id=str(uuid.uuid4()), params=params)

        print(f"[..] Sending synthetic FieldWorkArena task ({label})...")
        response = await client.send_message(request=request)

    # SendMessageResponse wraps a success or error root. Extract text from
    # either the returned Task.status.message or the direct Message payload
    # that non-task responses carry.
    payload = response.model_dump(mode="json", exclude_none=True)
    # The root can be SendMessageSuccessResponse or a JSON-RPC error.
    result = payload.get("result")
    if not result:
        # Error path: surface whatever the server told us.
        raise RuntimeError(f"send_message returned error payload: {json.dumps(payload)[:1000]}")

    return _final_answer(result)


def _final_answer(result: dict) -> str:
    """Return the answer text, or raise when the task failed or carried no text.

    A failed task still carries an artifact with a sanitized error message, so the
    task state decides success before the text does. A bare Message has no state.
    """
    text = _extract_text(result)
    status = result.get("status")
    state = status.get("state") if isinstance(status, dict) else None
    if state is not None and state != "completed":
        raise RuntimeError(f"task ended in state {state!r}: {text[:200]}")
    if not text:
        raise RuntimeError(
            f"send_message succeeded but returned no text. "
            f"Raw result: {json.dumps(result)[:1000]}"
        )
    return text


def _extract_text(result: dict) -> str:
    """
    Pull the first non-empty text field out of a Task or Message result.

    A2A lets the server return either a finalized Task (with artifacts,
    a status.message, and/or history) or a bare Message. Spatial Atlas
    writes its final answer to `artifacts[*].parts[*].text` (under the
    artifact name 'Analysis' for fieldwork and 'Submission' for mlebench),
    so that shape is checked FIRST. Status-message / history fall back
    cover agents that use different conventions.
    """
    # Shape 0 (primary for Spatial Atlas): artifacts.
    artifacts = result.get("artifacts")
    if isinstance(artifacts, list):
        for art in artifacts:
            if isinstance(art, dict):
                text = _text_from_parts(art.get("parts", []))
                if text:
                    return text

    # Shape 1: Task with status.message.parts
    status = result.get("status")
    if isinstance(status, dict):
        msg = status.get("message")
        if isinstance(msg, dict):
            text = _text_from_parts(msg.get("parts", []))
            if text:
                return text

    # Shape 2: Task with history (last message wins; prefer non-working
    # agent messages so we don't return a progress ticker).
    history = result.get("history")
    if isinstance(history, list):
        for msg in reversed(history):
            if isinstance(msg, dict):
                text = _text_from_parts(msg.get("parts", []))
                if text:
                    return text

    # Shape 3: direct Message
    text = _text_from_parts(result.get("parts", []))
    if text:
        return text

    return ""


def _text_from_parts(parts: list) -> str:
    """Concatenate TextPart.text across a message's parts list."""
    chunks = []
    for part in parts or []:
        if not isinstance(part, dict):
            continue
        root = part.get("root", part)  # protocol sometimes wraps
        if root.get("kind") == "text" and root.get("text"):
            chunks.append(root["text"])
    return "\n".join(chunks).strip()


async def _main_async() -> int:
    args = _parse_args()
    print(f"[..] Target: {args.url}")
    try:
        card = await _fetch_agent_card(args.url, args.timeout)
    except Exception as e:
        print(f"[FAIL] Could not fetch agent card: {e}", file=sys.stderr)
        return 1

    try:
        answer = await _send_smoke_message(
            card, args.url, args.timeout, image_path=args.image
        )
    except Exception as e:
        print(f"[FAIL] send_message round-trip failed: {e}", file=sys.stderr)
        return 1

    print(f"[ok] Got response ({len(answer)} chars):")
    print("-" * 60)
    print(answer[:2000])
    print("-" * 60)
    return 0


def main() -> int:
    return asyncio.run(_main_async())


if __name__ == "__main__":
    raise SystemExit(main())
