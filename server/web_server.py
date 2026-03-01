"""
web_server.py — FastAPI WebSocket server for the browser-based editor.

This server provides the same compilation and AI feedback pipeline as the
LSP server, but exposed over WebSocket instead of LSP/JSON-RPC. This allows
the project to include a custom browser-based editor (built with Monaco
Editor) that works without any VS Code installation — useful for demos,
classroom use, and students who prefer a lightweight setup.

Protocol
--------
The client (web editor) sends JSON messages over WebSocket:

    Request (compile):
    {
        "type":   "compile",
        "source": "<full Haskell source text>",
        "uri":    "web://editor"   (optional, defaults to "web://editor")
    }

    Response (diagnostics):
    {
        "type": "diagnostics",
        "uri":  "web://editor",
        "diagnostics": [
            {
                "severity":    "error" | "warning" | "info",
                "message":     "<raw GHC message>",
                "startLine":   1,
                "startCol":    1,
                "endLine":     1,
                "endCol":      5,
                "explanation": "<AI explanation or null>",
                "hint":        "<AI hint or null>",
                "category":    "TYPE_ERROR" | ...
            },
            ...
        ]
    }

    Error response:
    {
        "type":    "error",
        "message": "<error description>"
    }

REST endpoints
--------------
    GET /health    — Health check (returns {"status": "ok"})
    GET /context   — Dump the current AI context for a URI (debug)

CORS
----
CORS is enabled for all origins during development. In a production
deployment this should be tightened to the specific frontend origin.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from server.ghc.bridge import GHCBridge
from server.ghc.models import GHCDiagnostic
from server.ai.engine import AIFeedbackEngine
from server.ai.context import ContextManager

logger = logging.getLogger(__name__)

# ── App setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Haskell AI Language Server — Web API",
    description="WebSocket endpoint for browser-based Haskell editor with AI feedback",
    version="0.1.0",
)

# Allow the web editor to connect from any origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared server state — one instance per process, shared across all connections
_bridge  = GHCBridge()
_context = ContextManager()
_engine  = AIFeedbackEngine(context_manager=_context)


# ── REST endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint. Returns 200 OK when the server is running."""
    return {"status": "ok", "server": "haskell-ai-lsp"}


@app.get("/context")
async def get_context(uri: str = "web://editor") -> dict[str, Any]:
    """
    Return the AI context summary for a given URI.

    This is a debug endpoint useful during development to inspect the
    experience level tracker and see how it adapts over a session.
    """
    ctx = _context.get_or_create(uri)
    return ctx.summary()


# ── WebSocket endpoint ─────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    Main WebSocket endpoint — handles compile requests from the web editor.

    One WebSocket connection per editor session. The connection stays open
    for the duration of the editing session. Each message is a JSON object
    with a "type" field.
    """
    await websocket.accept()
    logger.info("WebSocket connection opened from %s", websocket.client)

    try:
        while True:
            raw = await websocket.receive_text()
            await _handle_message(websocket, raw)
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc, exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


async def _handle_message(websocket: WebSocket, raw: str) -> None:
    """
    Dispatch a single WebSocket message.

    Parses the JSON, validates the type field, and calls the appropriate
    handler. Sends an error response for malformed or unknown messages.
    """
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError as exc:
        await websocket.send_json({"type": "error", "message": f"Invalid JSON: {exc}"})
        return

    msg_type = msg.get("type")

    if msg_type == "compile":
        await _handle_compile(websocket, msg)
    elif msg_type == "reset":
        _handle_reset(msg)
    else:
        await websocket.send_json({
            "type": "error",
            "message": f"Unknown message type: {msg_type!r}",
        })


async def _handle_compile(websocket: WebSocket, msg: dict) -> None:
    """
    Handle a compile request.

    Runs GHC, enriches diagnostics with AI, and sends back a diagnostics
    response message. Sends an error message if compilation itself fails
    (as opposed to GHC reporting errors in the code — those come back as
    normal diagnostics).
    """
    source = msg.get("source", "")
    uri    = msg.get("uri", "web://editor")

    if not source.strip():
        await websocket.send_json({
            "type":        "diagnostics",
            "uri":         uri,
            "diagnostics": [],
        })
        return

    logger.debug("Compiling for uri=%s (%d chars)", uri, len(source))

    try:
        result = await _bridge.compile(source, uri)
    except Exception as exc:
        logger.error("GHCBridge error: %s", exc)
        await websocket.send_json({
            "type":    "error",
            "message": f"Compilation failed: {exc}",
        })
        return

    # Enrich with AI (best-effort — errors fall back to raw diagnostic)
    enriched = await _engine.enrich_all(result.diagnostics[:5], source, uri)
    all_diags = enriched + result.diagnostics[5:]

    await websocket.send_json({
        "type":        "diagnostics",
        "uri":         uri,
        "diagnostics": [_diagnostic_to_dict(d) for d in all_diags],
    })


def _handle_reset(msg: dict) -> None:
    """Reset the AI context for a given URI (e.g. when the user clears the editor)."""
    uri = msg.get("uri", "web://editor")
    _context.reset(uri)
    logger.info("AI context reset for %s", uri)


# ── Serialisation ──────────────────────────────────────────────────────────

def _diagnostic_to_dict(d: GHCDiagnostic) -> dict[str, Any]:
    """
    Serialise a GHCDiagnostic to a JSON-compatible dict for the web client.

    All fields use camelCase to match JavaScript conventions in the
    web editor frontend.
    """
    return {
        "severity":    d.severity.name.lower(),   # "error", "warning", "info", "hint"
        "message":     d.message,
        "category":    d.category.value,
        "startLine":   d.span.start_line if d.span else 1,
        "startCol":    d.span.start_col  if d.span else 1,
        "endLine":     d.span.end_line   if d.span else 1,
        "endCol":      d.span.end_col    if d.span else 1,
        "explanation": d.ai_explanation,
        "hint":        d.ai_hint,
        "errorCode":   d.error_code,
    }