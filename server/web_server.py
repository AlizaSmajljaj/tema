"""
web_server.py — FastAPI WebSocket server for the Haskell AI Tutor web editor.
"""
from __future__ import annotations
import asyncio, json, logging, os
from pathlib import Path
from typing import Optional

import requests as req_lib
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from server.ghc.bridge import GHCBridge
from server.ghc.models import GHCDiagnostic
from server.ai.engine import AIFeedbackEngine
from server.ai.context import ContextManager

load_dotenv()
logger = logging.getLogger(__name__)

app = FastAPI(title="Haskell AI Tutor")

_bridge  = GHCBridge()
_context = ContextManager()
_engine  = AIFeedbackEngine(context_manager=_context)

MAX_AI_ENRICHMENTS = 5


@app.get("/")
async def serve_editor():
    candidates = [
        Path(__file__).parent.parent / "web_editor.html",
        Path(__file__).parent.parent / "web-editor" / "index.html",
        Path("web_editor.html"),
    ]
    for path in candidates:
        if path.exists():
            return FileResponse(str(path))
    return JSONResponse({"error": "web_editor.html not found."}, status_code=404)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat")
async def chat_proxy(request: Request):
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return JSONResponse({"content": "GROQ_API_KEY not configured on server."})
    try:
        body     = await request.json()
        messages = body.get("messages", [])
        resp = req_lib.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": os.environ.get("GROQ_MODEL","llama-3.1-8b-instant"), "max_tokens": 250, "messages": messages},
            timeout=15,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return JSONResponse({"content": content})
    except Exception as exc:
        logger.warning("chat_proxy error: %s", exc)
        return JSONResponse({"content": f"AI unavailable: {exc}"})


@app.get("/context")
async def get_context():
    return JSONResponse({"sessions": len(_context)})


def _diagnostic_to_dict(d: GHCDiagnostic) -> dict:
    return {
        "severity":    d.severity.name.lower(),
        "startLine":   d.span.start_line if d.span else 0,
        "startCol":    d.span.start_col  if d.span else 0,
        "endLine":     d.span.end_line   if d.span else 0,
        "endCol":      d.span.end_col    if d.span else 0,
        "message":     d.message,
        "category":    str(d.category),
        "explanation": d.ai_explanation,
        "hint":        d.ai_hint,
        "scaffold":    getattr(d, "ai_scaffold", ""),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type":"error","message":"Invalid JSON"}))
                continue

            if msg.get("type") != "compile":
                continue

            source = msg.get("source", "")
            uri    = msg.get("uri", "web://editor")

            if not source.strip():
                await websocket.send_text(json.dumps({"type":"diagnostics","diagnostics":[]}))
                continue

            try:
                result    = await _bridge.compile(source, uri)
                diags     = result.diagnostics
                to_enrich = diags[:MAX_AI_ENRICHMENTS]
                rest      = diags[MAX_AI_ENRICHMENTS:]
                enriched  = await _engine.enrich_all(to_enrich, source, uri)
                all_diags = enriched + rest
                payload   = json.dumps({
                    "type":        "diagnostics",
                    "diagnostics": [_diagnostic_to_dict(d) for d in all_diags],
                })
                await websocket.send_text(payload)
                logger.info("Compiled %s: %d diagnostics (%d AI-enriched)", uri, len(all_diags), len(enriched))
            except Exception as exc:
                logger.error("Compile error for %s: %s", uri, exc)
                await websocket.send_text(json.dumps({"type":"error","message":str(exc)}))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")


def start(port: int = 8765):
    logger.info("Starting Haskell AI web server on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")