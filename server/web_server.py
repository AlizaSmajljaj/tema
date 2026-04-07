"""
web_server.py — FastAPI WebSocket server for the Haskell AI Tutor web editor.

Endpoints:
  GET  /             — serves web_editor.html
  GET  /health       — health check
  POST /api/chat     — proxies Groq API calls (keeps API key off browser)
  POST /api/progress — tracks which problems a session has solved
  GET  /api/next     — suggests next problem based on session progress
  WS   /ws           — WebSocket compile + run endpoint
"""

from __future__ import annotations
import asyncio, json, logging, os, subprocess, tempfile
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

# Session progress: session_id -> set of solved problem ids
# Simple in-memory store (resets on server restart, good enough for exam prep)
_session_progress: dict[str, set] = {}


# ── Serve the web editor ───────────────────────────────────────────────────

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


# ── Health check ───────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "sessions": len(_context)}


# ── Groq API proxy ─────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat_proxy(request: Request):
    """
    Proxy AI calls — tries Groq first, falls back to Ollama.
    This means the tool works even without internet if Ollama is running.
    """
    body     = await request.json()
    messages = body.get("messages", [])

    # ── Try Groq first ─────────────────────────────────────────────────
    api_key = os.environ.get("GROQ_API_KEY", "")
    if api_key:
        try:
            resp = req_lib.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model":      os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
                    "max_tokens": 250,
                    "messages":   messages,
                },
                timeout=15,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            logger.debug("chat_proxy: used Groq")
            return JSONResponse({"content": content, "provider": "groq"})
        except Exception as exc:
            logger.warning("Groq failed (%s), trying Ollama fallback", exc)

    # ── Fall back to Ollama ────────────────────────────────────────────
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2")
    ollama_url   = os.environ.get("OLLAMA_URL",   "http://localhost:11434")
    try:
        resp = req_lib.post(
            f"{ollama_url}/api/chat",
            json={
                "model":    ollama_model,
                "messages": messages,
                "stream":   False,
                "options":  {"num_predict": 250},
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        logger.info("chat_proxy: used Ollama fallback (model=%s)", ollama_model)
        return JSONResponse({"content": content, "provider": "ollama"})
    except Exception as exc2:
        logger.warning("Ollama also failed: %s", exc2)

    return JSONResponse({"content": "AI unavailable — check your GROQ_API_KEY or start Ollama."})


# ── Progress tracking ──────────────────────────────────────────────────────

@app.post("/api/progress")
async def mark_solved(request: Request):
    """Mark a problem as solved for a session."""
    body       = await request.json()
    session_id = body.get("session_id", "default")
    problem_id = body.get("problem_id", "")
    if session_id not in _session_progress:
        _session_progress[session_id] = set()
    _session_progress[session_id].add(problem_id)
    return JSONResponse({
        "solved": list(_session_progress[session_id]),
        "count":  len(_session_progress[session_id]),
    })


@app.get("/api/progress/{session_id}")
async def get_progress(session_id: str):
    """Get progress for a session."""
    solved = list(_session_progress.get(session_id, set()))
    return JSONResponse({"solved": solved, "count": len(solved)})


# ── Run code and capture output ────────────────────────────────────────────

@app.post("/api/run")
async def run_code(request: Request):
    """
    Compile and RUN Haskell code, returning stdout.
    Used by the 'Check output' feature.
    Only runs if there are no compile errors first.
    """
    body   = await request.json()
    source = body.get("source", "")
    if not source.strip():
        return JSONResponse({"output": "", "error": "Empty source"})

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "Main.hs")
            exe_path = os.path.join(tmpdir, "Main")
            with open(src_path, "w", encoding="utf-8") as f:
                f.write(source)

            # Compile with full code generation (not -fno-code)
            compile_result = subprocess.run(
                ["ghc", "-o", exe_path, src_path],
                capture_output=True, text=True, timeout=30,
                cwd=tmpdir,
            )
            if compile_result.returncode != 0:
                return JSONResponse({
                    "output": "",
                    "error":  "Compilation failed — fix errors first.",
                })

            # Run the executable
            run_result = subprocess.run(
                [exe_path],
                capture_output=True, text=True, timeout=10,
                cwd=tmpdir,
            )
            return JSONResponse({
                "output": run_result.stdout,
                "error":  run_result.stderr if run_result.returncode != 0 else "",
            })
    except subprocess.TimeoutExpired:
        return JSONResponse({"output": "", "error": "Execution timed out (10s limit)."})
    except FileNotFoundError:
        return JSONResponse({"output": "", "error": "GHC not found. Is it installed?"})
    except Exception as exc:
        return JSONResponse({"output": "", "error": str(exc)})


# ── Diagnostic conversion ──────────────────────────────────────────────────

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


# ── WebSocket compile endpoint ─────────────────────────────────────────────

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
                    "clean":       len(all_diags) == 0,
                })
                await websocket.send_text(payload)
                logger.info("Compiled %s: %d diagnostics (%d AI-enriched)",
                            uri, len(all_diags), len(enriched))
            except Exception as exc:
                logger.error("Compile error for %s: %s", uri, exc)
                await websocket.send_text(json.dumps({"type":"error","message":str(exc)}))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")


# ── Entry point ────────────────────────────────────────────────────────────

def start(port: int = 8765):
    logger.info("Starting Haskell AI web server on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")