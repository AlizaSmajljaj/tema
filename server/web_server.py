"""
web_server.py — FastAPI WebSocket server with user authentication and persistence.
Updated for Railway compatibility: SQLite unixepoch fix, Port handling, and Groq Proxy fixes.
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

# Load .env for local development; Railway uses system env vars
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
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization[7:].strip()
    return get_user_by_token(token) if token else None


# ── Serve HTML ─────────────────────────────────────────────────────────────

@app.get("/")
async def serve_editor():
    # Try multiple common paths for Railway deployment
    possible_paths = [
        Path(__file__).parent.parent / "web_editor.html",
        Path("web_editor.html"),
        Path("/app/web_editor.html")
    ]
    for path in possible_paths:
        if path.exists():
            return FileResponse(str(path))
    return JSONResponse({"error": "web_editor.html not found."}, status_code=404)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/test")
async def test_everything():
    """Diagnostic endpoint to verify components are working on Railway."""
    results = {}

    # Check Groq API key
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    results["groq_key_present"] = bool(api_key)
    results["groq_key_looks_valid"] = api_key.startswith("gsk_") if api_key else False
    results["groq_model"] = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

    # Check database
    try:
        from server.database import get_conn
        with get_conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        results["database_ok"] = True
        results["user_count"] = count
    except Exception as e:
        results["database_ok"] = False
        results["database_error"] = str(e)

    # Check GHC
    try:
        r = subprocess.run(["ghc", "--version"], capture_output=True, text=True, timeout=5)
        results["ghc_found"] = r.returncode == 0
        results["ghc_version"] = r.stdout.strip()
    except Exception:
        results["ghc_found"] = False

    return results


# ── Auth endpoints ─────────────────────────────────────────────────────────

@app.post("/api/register")
async def register(request: Request):
    body     = await request.json()
    username = body.get("username", "").strip()
    if not username or len(username) < 2:
        return JSONResponse({"error": "Username must be at least 2 characters."}, status_code=400)

    user = register_user(username)
    if not user:
        return JSONResponse({"error": "Username already taken."}, status_code=409)

    logger.info("New user registered: %s", username)
    return JSONResponse({"token": user["token"], "username": user["username"], "id": user["id"]})


@app.post("/api/login")
async def login(request: Request):
    body     = await request.json()
    username = body.get("username", "").strip()
    user     = get_user_by_username(username)
    if not user:
        return JSONResponse({"error": "Username not found."}, status_code=404)
    return JSONResponse({"token": user["token"], "username": user["username"], "id": user["id"]})


@app.get("/api/me")
async def me(authorization: str = Header(default="")):
    user = _get_user(authorization)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    stats = get_user_stats(user["id"])
    return JSONResponse({"username": user["username"], "id": user["id"], **stats})


# ── Progress endpoints ─────────────────────────────────────────────────────

@app.get("/api/progress")
async def get_progress(authorization: str = Header(default="")):
    user = _get_user(authorization)
    if not user:
        return JSONResponse({"solved": [], "in_progress": {}, "total_solved": 0})
    return JSONResponse(get_user_progress(user["id"]))


@app.post("/api/progress")
async def update_progress(request: Request, authorization: str = Header(default="")):
    body       = await request.json()
    problem_id = body.get("problem_id", "")
    user       = _get_user(authorization)
    if user:
        mark_problem_solved(user["id"], problem_id)
    return JSONResponse({"ok": True})


@app.post("/api/save-code")
async def save_code_endpoint(request: Request, authorization: str = Header(default="")):
    body       = await request.json()
    problem_id = body.get("problem_id", "")
    code       = body.get("code", "")
    user       = _get_user(authorization)
    if user and problem_id:
        save_code(user["id"], problem_id, code)
    return JSONResponse({"ok": True})


@app.get("/api/load/{problem_id}")
async def load_problem(problem_id: str, authorization: str = Header(default="")):
    user = _get_user(authorization)
    if not user:
        return JSONResponse({"code": "", "conversation": []})

    code         = get_saved_code(user["id"], problem_id)
    conversation = get_conversation(user["id"], problem_id, context="error")
    think_convo  = get_conversation(user["id"], problem_id, context="think")

    return JSONResponse({
        "code":         code,
        "conversation": conversation,
        "think":         think_convo,
    })


# ── AI chat proxy ──────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat_proxy(request: Request, authorization: str = Header(default="")):
    body       = await request.json()
    messages   = body.get("messages", [])
    problem_id = body.get("problem_id", "")
    context    = body.get("context", "error")
    user       = _get_user(authorization)

    content  = None
    provider = None

    # ── Try Groq ───────────────────────────────────────────────────────
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    model_name = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

    if api_key:
        try:
            logger.info("Attempting Groq request: %s", model_name)
            resp = req_lib.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model":      model_name,
                    "max_tokens": 500,
                    "messages":   messages,
                },
                timeout=15,
            )
            if resp.status_code != 200:
                logger.error("Groq Error: %s", resp.text)
            
            resp.raise_for_status()
            content  = resp.json()["choices"][0]["message"]["content"]
            provider = "groq"
        except Exception as exc:
            logger.warning("Groq failed, trying Ollama fallback: %s", exc)

    # ── Fall back to Ollama ────────────────────────────────────────────
    if content is None:
        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        try:
            resp = req_lib.post(
                f"{ollama_url}/api/chat",
                json={
                    "model":   os.environ.get("OLLAMA_MODEL", "llama3.2"),
                    "messages": messages,
                    "stream":  False,
                },
                timeout=20,
            )
            resp.raise_for_status()
            content  = resp.json()["message"]["content"]
            provider = "ollama"
        except Exception:
            pass

    if content is None:
        return JSONResponse({"content": "AI is currently offline. Check Groq API Key settings.", "error": True})

    # ── Save to DB ─────────────────────────────────────────────────────
    if user and problem_id:
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if user_msgs:
            save_message(user["id"], problem_id, "user", user_msgs[-1]["content"], context)
        save_message(user["id"], problem_id, "assistant", content, context)

    return JSONResponse({"content": content, "provider": provider})


# ── Run code ───────────────────────────────────────────────────────────────

@app.post("/api/run")
async def run_code(request: Request):
    body   = await request.json()
    source = body.get("source", "")
    if not source.strip():
        return JSONResponse({"output": "", "error": "No source code provided."})
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "Main.hs")
            exe_path = os.path.join(tmpdir, "Main")
            with open(src_path, "w", encoding="utf-8") as f:
                f.write(source)
            
            compile_res = subprocess.run(["ghc", "-o", exe_path, src_path], capture_output=True, text=True, timeout=30)
            if compile_res.returncode != 0:
                return JSONResponse({"output": "", "error": compile_res.stderr})
            
            run_res = subprocess.run([exe_path], capture_output=True, text=True, timeout=10)
            return JSONResponse({"output": run_res.stdout, "error": run_res.stderr})
    except Exception as e:
        return JSONResponse({"output": "", "error": str(e)})


# ── WebSocket ──────────────────────────────────────────────────────────────

def _diagnostic_to_dict(d: GHCDiagnostic) -> dict:
    return {
        "severity":  d.severity.name.lower(),
        "startLine": d.span.start_line if d.span else 0,
        "startCol":  d.span.start_col  if d.span else 0,
        "endLine":   d.span.end_line   if d.span else 0,
        "endCol":    d.span.end_col    if d.span else 0,
        "message":   d.message,
        "explanation": d.ai_explanation,
        "hint":      d.ai_hint,
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            if msg.get("type") == "compile":
                result    = await _bridge.compile(msg.get("source", ""), "web://editor")
                enriched  = await _engine.enrich_all(result.diagnostics[:MAX_AI_ENRICHMENTS], msg.get("source", ""), "web://editor")
                all_diags = enriched + result.diagnostics[MAX_AI_ENRICHMENTS:]
                await websocket.send_text(json.dumps({
                    "type": "diagnostics",
                    "diagnostics": [_diagnostic_to_dict(d) for d in all_diags],
                    "clean": len(all_diags) == 0
                }))
    except WebSocketDisconnect:
        pass

def start(port: int = 8765):
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")