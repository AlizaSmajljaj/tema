"""
web_server.py — FastAPI WebSocket server with user authentication and persistence.

Endpoints:
  GET  /                      — serves web_editor.html
  GET  /health                — health check
  POST /api/register          — create account (username only)
  POST /api/login             — login by username, returns token
  GET  /api/me                — get current user info (requires token)
  GET  /api/progress          — get full progress for logged-in user
  POST /api/progress          — mark problem solved
  POST /api/save-code         — save current code for a problem
  GET  /api/load/{problem_id} — load saved code + conversation for a problem
  POST /api/chat              — proxy Groq/Ollama AI calls + save conversation
  POST /api/run               — compile and run Haskell code
  WS   /ws                    — WebSocket compile endpoint
"""

from __future__ import annotations
import asyncio, json, logging, os, subprocess, tempfile
from pathlib import Path

import requests as req_lib
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Header
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from server.ghc.bridge import GHCBridge
from server.ghc.models import GHCDiagnostic
from server.ai.engine import AIFeedbackEngine
from server.ai.context import ContextManager
from server.database import (
    get_user_by_token, register_user, get_user_by_username,
    save_code, mark_problem_solved, get_user_progress,
    get_saved_code, save_message, get_conversation,
    clear_conversation, increment_category, get_user_stats,
)

load_dotenv()
logger = logging.getLogger(__name__)

app = FastAPI(title="Haskell AI Tutor")

_bridge  = GHCBridge()
_context = ContextManager()
_engine  = AIFeedbackEngine(context_manager=_context)

MAX_AI_ENRICHMENTS = 5


# ── Auth helper ────────────────────────────────────────────────────────────

def _get_user(authorization: str = "") -> dict | None:
    """Extract and validate bearer token from Authorization header."""
    if not authorization.startswith("Bearer "):
        return None
    token = authorization[7:].strip()
    return get_user_by_token(token) if token else None


# ── Serve HTML ─────────────────────────────────────────────────────────────

@app.get("/")
async def serve_editor():
    for path in [
        Path(__file__).parent.parent / "web_editor.html",
        Path("web_editor.html"),
    ]:
        if path.exists():
            return FileResponse(str(path))
    return JSONResponse({"error": "web_editor.html not found."}, status_code=404)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/test")
async def test_everything():
    """
    Quick diagnostic endpoint — visit http://localhost:8765/api/test
    to verify all components are working.
    """
    import os
    results = {}

    # Check Groq API key
    api_key = os.environ.get("GROQ_API_KEY", "")
    results["groq_key_present"]    = bool(api_key)
    results["groq_key_looks_valid"] = api_key.startswith("gsk_") if api_key else False

    # Check Ollama
    try:
        r = req_lib.get("http://localhost:11434/api/tags", timeout=2)
        results["ollama_running"]   = r.status_code == 200
        models = [m["name"] for m in r.json().get("models", [])]
        results["ollama_models"]    = models
    except Exception:
        results["ollama_running"]   = False
        results["ollama_models"]    = []

    # Check GHC
    try:
        import subprocess
        r = subprocess.run(["ghc", "--version"], capture_output=True, text=True, timeout=5)
        results["ghc_found"]    = r.returncode == 0
        results["ghc_version"]  = r.stdout.strip()
    except Exception:
        results["ghc_found"]   = False
        results["ghc_version"] = "not found"

    # Check database
    try:
        from server.database import get_conn
        with get_conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        results["database_ok"]    = True
        results["user_count"]     = count
    except Exception as e:
        results["database_ok"]   = False
        results["database_error"] = str(e)

    # Overall status
    results["all_ok"] = (
        results["ghc_found"] and
        results["database_ok"] and
        (results["groq_key_looks_valid"] or results["ollama_running"])
    )

    return results


# ── Auth endpoints ─────────────────────────────────────────────────────────

@app.post("/api/register")
async def register(request: Request):
    """
    Register a new user with just a username.
    Returns a token the client stores in localStorage.
    """
    body     = await request.json()
    username = body.get("username", "").strip()
    if not username or len(username) < 2:
        return JSONResponse({"error": "Username must be at least 2 characters."}, status_code=400)
    if len(username) > 30:
        return JSONResponse({"error": "Username too long (max 30 chars)."}, status_code=400)

    user = register_user(username)
    if not user:
        return JSONResponse({"error": "Username already taken."}, status_code=409)

    logger.info("New user registered: %s", username)
    return JSONResponse({"token": user["token"], "username": user["username"], "id": user["id"]})


@app.post("/api/login")
async def login(request: Request):
    """Login by username — returns the same token as registration."""
    body     = await request.json()
    username = body.get("username", "").strip()
    user     = get_user_by_username(username)
    if not user:
        return JSONResponse({"error": "Username not found. Register first."}, status_code=404)
    return JSONResponse({"token": user["token"], "username": user["username"], "id": user["id"]})


@app.get("/api/me")
async def me(authorization: str = Header(default="")):
    """Get current user info and stats."""
    user = _get_user(authorization)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    stats = get_user_stats(user["id"])
    return JSONResponse({"username": user["username"], "id": user["id"], **stats})


# ── Progress endpoints ─────────────────────────────────────────────────────

@app.get("/api/progress")
async def get_progress(authorization: str = Header(default="")):
    """Get full problem progress for logged-in user."""
    user = _get_user(authorization)
    if not user:
        return JSONResponse({"solved": [], "in_progress": {}, "total_solved": 0})
    return JSONResponse(get_user_progress(user["id"]))


@app.post("/api/progress")
async def update_progress(request: Request, authorization: str = Header(default="")):
    """Mark a problem as solved."""
    body       = await request.json()
    problem_id = body.get("problem_id", "")
    user       = _get_user(authorization)

    if user:
        mark_problem_solved(user["id"], problem_id)
        logger.info("User %s solved problem %s", user["username"], problem_id)

    return JSONResponse({"ok": True})


@app.post("/api/save-code")
async def save_code_endpoint(request: Request, authorization: str = Header(default="")):
    """Auto-save current code for a problem."""
    body       = await request.json()
    problem_id = body.get("problem_id", "")
    code       = body.get("code", "")
    user       = _get_user(authorization)

    if user and problem_id and code:
        save_code(user["id"], problem_id, code)

    return JSONResponse({"ok": True})


@app.get("/api/load/{problem_id}")
async def load_problem(problem_id: str, authorization: str = Header(default="")):
    """Load saved code and conversation history for a problem."""
    user = _get_user(authorization)
    if not user:
        return JSONResponse({"code": "", "conversation": []})

    code         = get_saved_code(user["id"], problem_id)
    conversation = get_conversation(user["id"], problem_id, context="error")
    think_convo  = get_conversation(user["id"], problem_id, context="think")

    return JSONResponse({
        "code":         code,
        "conversation": conversation,
        "think":        think_convo,
    })


# ── AI chat proxy ──────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat_proxy(request: Request, authorization: str = Header(default="")):
    """
    Proxy AI calls — tries Groq first, falls back to Ollama.
    Saves conversation to database if user is logged in.
    """
    body       = await request.json()
    messages   = body.get("messages", [])
    problem_id = body.get("problem_id", "")
    context    = body.get("context", "error")   # "error" or "think"
    user       = _get_user(authorization)

    content  = None
    provider = None

    # ── Try Groq ───────────────────────────────────────────────────────
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
            content  = resp.json()["choices"][0]["message"]["content"]
            provider = "groq"
        except Exception as exc:
            logger.warning("Groq failed (%s), trying Ollama", exc)

    # ── Fall back to Ollama ────────────────────────────────────────────
    if content is None:
        ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2")
        ollama_url   = os.environ.get("OLLAMA_URL",   "http://localhost:11434")
        try:
            resp = req_lib.post(
                f"{ollama_url}/api/chat",
                json={
                    "model":   ollama_model,
                    "messages": messages,
                    "stream":  False,
                    "options": {"num_predict": 250},
                },
                timeout=30,
            )
            resp.raise_for_status()
            content  = resp.json()["message"]["content"]
            provider = "ollama"
            logger.info("Used Ollama fallback (model=%s)", ollama_model)
        except Exception as exc2:
            logger.warning("Ollama also failed: %s", exc2)

    if content is None:
        return JSONResponse({"content": "AI unavailable — check GROQ_API_KEY or start Ollama."})

    # ── Save conversation to database ──────────────────────────────────
    if user and problem_id:
        # Save the last user message and this AI response
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if user_msgs:
            save_message(user["id"], problem_id, "user", user_msgs[-1]["content"], context)
        save_message(user["id"], problem_id, "assistant", content, context)

    return JSONResponse({"content": content, "provider": provider})


# ── Run code ───────────────────────────────────────────────────────────────

@app.post("/api/run")
async def run_code(request: Request, authorization: str = Header(default="")):
    """Compile and run Haskell code, return stdout."""
    import shutil as _shutil
    body   = await request.json()
    source = body.get("source", "")
    if not source.strip():
        return JSONResponse({"output": "", "error": "Empty source"})

    # Resolve GHC the same way the bridge does — respects GHC_PATH env var
    ghc_path = os.environ.get("GHC_PATH") or _shutil.which("ghc")
    if not ghc_path:
        return JSONResponse({"output": "", "error": "GHC not found. Install GHC or set GHC_PATH in .env"})

    tmpdir = tempfile.mkdtemp(prefix="haskell-run-")
    try:
        src_path = os.path.join(tmpdir, "Main.hs")
        exe_path = os.path.join(tmpdir, "Main")
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(source)

        # Full compilation (with code generation, unlike -fno-code used for diagnostics)
        compile_result = subprocess.run(
            [ghc_path, "-o", exe_path, src_path],
            capture_output=True, text=True, timeout=30, cwd=tmpdir,
        )
        if compile_result.returncode != 0:
            # Return the actual GHC error so the student can see what's wrong
            err_text = compile_result.stderr or compile_result.stdout or "Compilation failed."
            return JSONResponse({"output": "", "error": err_text.strip()})

        # Execute the binary
        run_result = subprocess.run(
            [exe_path],
            capture_output=True, text=True, timeout=10, cwd=tmpdir,
        )
        output = run_result.stdout
        # Combine stdout + stderr so runtime errors (like pattern match failure) are visible
        if run_result.returncode != 0:
            error_out = run_result.stderr.strip()
            return JSONResponse({"output": output, "error": error_out})
        return JSONResponse({"output": output, "error": ""})

    except subprocess.TimeoutExpired:
        return JSONResponse({"output": "", "error": "Timed out after 10 seconds — check for infinite loops."})
    except FileNotFoundError:
        return JSONResponse({"output": "", "error": f"Could not execute: {exe_path}. Compilation may have failed silently."})
    except Exception as exc:
        return JSONResponse({"output": "", "error": str(exc)})
    finally:
        # Clean up after execution (not before, unlike TemporaryDirectory context manager)
        _shutil.rmtree(tmpdir, ignore_errors=True)


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


# ── WebSocket ──────────────────────────────────────────────────────────────

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
                await websocket.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                continue

            if msg.get("type") != "compile":
                continue

            source = msg.get("source", "")
            uri    = msg.get("uri", "web://editor")

            if not source.strip():
                await websocket.send_text(json.dumps({"type": "diagnostics", "diagnostics": [], "clean": True}))
                continue

            try:
                result    = await _bridge.compile(source, uri)
                diags     = result.diagnostics
                to_enrich = diags[:MAX_AI_ENRICHMENTS]
                rest      = diags[MAX_AI_ENRICHMENTS:]
                enriched  = await _engine.enrich_all(to_enrich, source, uri)
                all_diags = enriched + rest
                await websocket.send_text(json.dumps({
                    "type":        "diagnostics",
                    "diagnostics": [_diagnostic_to_dict(d) for d in all_diags],
                    "clean":       len(all_diags) == 0,
                }))
                logger.info("Compiled %s: %d diagnostics", uri, len(all_diags))
            except Exception as exc:
                logger.error("Compile error: %s", exc)
                await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")


def start(port: int = 8765):
    logger.info("Starting Haskell AI web server on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")