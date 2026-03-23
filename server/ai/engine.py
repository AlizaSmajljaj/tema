"""
engine.py — AI feedback engine.

This module is the central orchestrator of the AI layer. It receives a
GHCDiagnostic, looks up the student's context, selects the right prompt,
calls the Groq API, parses the response, and returns an enriched diagnostic.

Groq provides free, fast inference for open-source models (Llama 3.1, Mixtral)
through an OpenAI-compatible REST API. We call it directly with the standard
`requests` library — no third-party AI SDK required. This keeps the dependency
tree minimal and makes the HTTP interaction completely transparent.

API shape used
--------------
POST https://api.groq.com/openai/v1/chat/completions
Headers: Authorization: Bearer <GROQ_API_KEY>
Body:    { "model": ..., "max_tokens": ..., "messages": [...] }
Response: { "choices": [{ "message": { "content": "..." } }] }

Public API
----------
    engine = AIFeedbackEngine()
    enriched = await engine.enrich(diagnostic, source, uri)

Error handling strategy
-----------------------
AI enrichment is best-effort. If the API call fails for any reason (network
error, rate limit, invalid key, timeout), the engine logs the error and
returns the original diagnostic unchanged. This ensures that a Groq API
outage never breaks the core language server — students still get raw GHC
diagnostics even without AI explanations.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import requests

from server.ghc.models import GHCDiagnostic, ErrorCategory
from server.ai.context import ContextManager, ExperienceLevel
from server.ai.prompts import build_system_prompt, build_user_prompt, parse_response

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────

_GROQ_URL    = "https://api.groq.com/openai/v1/chat/completions"
_DEFAULT_MODEL = "llama-3.1-8b-instant"   # fast, free, great for code errors
_MAX_TOKENS    = 300
_CONTEXT_LINES = 3
_TIMEOUT_SECS  = 10    # hard timeout per request — keeps editor snappy

_SKIP_CATEGORIES: frozenset[ErrorCategory] = frozenset()


# ── Thin HTTP wrapper ─────────────────────────────────────────────────────

class _GroqClient:
    """
    Minimal Groq API client using the standard `requests` library.

    We implement only the single endpoint we need rather than pulling in the
    full openai SDK. This keeps the dependency tree small and the HTTP
    interaction easy to audit and explain in the thesis.
    """

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self.model    = model

    def chat_complete(self, system: str, user: str) -> str:
        """
        Call Groq's chat completions endpoint and return the response text.

        Raises
        ------
        requests.HTTPError
            On 4xx/5xx responses (e.g. 401 invalid key, 429 rate limit).
        requests.ConnectionError
            On network failures.
        requests.Timeout
            If the server takes longer than _TIMEOUT_SECS to respond.
        """
        payload = {
            "model":      self.model,
            "max_tokens": _MAX_TOKENS,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        }
        resp = requests.post(
            _GROQ_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type":  "application/json",
            },
            data=json.dumps(payload),
            timeout=_TIMEOUT_SECS,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ── Engine ────────────────────────────────────────────────────────────────

class AIFeedbackEngine:
    """
    Enriches GHCDiagnostic objects with AI-generated explanations and hints.

    Uses the Groq API (free tier) via a direct HTTP call with `requests`.
    Groq runs Llama 3.1 / Mixtral with very low latency, making it well-suited
    for interactive editor feedback.

    Parameters
    ----------
    api_key:
        Groq API key. Defaults to the GROQ_API_KEY environment variable.
        Get a free key at https://console.groq.com/
    model:
        Groq model identifier. Defaults to GROQ_MODEL env var or
        llama-3.1-8b-instant.
    context_manager:
        Optional pre-built ContextManager. If None, a new one is created.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        context_manager: ContextManager | None = None,
    ) -> None:
        # If api_key is explicitly provided (even as an empty string), respect it.
        # Otherwise, fall back to the environment variable.
        self._api_key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY", "")
        self._model   = model or os.environ.get("GROQ_MODEL", _DEFAULT_MODEL)
        self._context = context_manager or ContextManager()
        self._client  = _GroqClient(api_key=self._api_key, model=self._model)
        logger.info("AIFeedbackEngine ready [provider=Groq, model=%s]", self._model)

    # ── Public API ────────────────────────────────────────────────────────

    async def enrich(
        self,
        diagnostic: GHCDiagnostic,
        source: str,
        uri: str,
    ) -> GHCDiagnostic:
        """
        Enrich a single GHCDiagnostic with an AI explanation and hint.

        The original diagnostic object is not mutated — a new GHCDiagnostic
        is returned with ai_explanation and ai_hint populated.

        If AI enrichment fails for any reason the original is returned.
        """
        if diagnostic.category in _SKIP_CATEGORIES:
            return diagnostic

        if not self._api_key:
            logger.warning("GROQ_API_KEY not set — skipping AI enrichment")
            return diagnostic

        try:
            return await self._enrich_inner(diagnostic, source, uri)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            if status == 401:
                logger.error("Groq auth failed (401) — check GROQ_API_KEY")
            elif status == 429:
                logger.warning("Groq rate limit hit (429) — returning raw diagnostic")
            else:
                logger.warning("Groq HTTP error %s — returning raw diagnostic", status)
            return diagnostic
        except (requests.ConnectionError, requests.Timeout) as exc:
            logger.warning("Groq connection error: %s", exc)
            return diagnostic
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error during AI enrichment: %s", exc, exc_info=True)
            return diagnostic

    async def enrich_all(
        self,
        diagnostics: list[GHCDiagnostic],
        source: str,
        uri: str,
    ) -> list[GHCDiagnostic]:
        """
        Enrich a list of diagnostics sequentially.

        Sequential (not concurrent) to avoid bursting the rate limit.
        GHC typically produces at most 5–10 diagnostics per compilation.
        """
        return [await self.enrich(d, source, uri) for d in diagnostics]

    @property
    def context_manager(self) -> ContextManager:
        return self._context

    # ── Internal ─────────────────────────────────────────────────────────

    async def _enrich_inner(
        self,
        diagnostic: GHCDiagnostic,
        source: str,
        uri: str,
    ) -> GHCDiagnostic:
        user_ctx = self._context.get_or_create(uri)
        level    = user_ctx.record_diagnostic(diagnostic.category, diagnostic.message)

        source_context = _extract_source_context(
            source,
            line=diagnostic.span.start_line if diagnostic.span else 1,
            radius=_CONTEXT_LINES,
        )

        system_prompt = build_system_prompt(diagnostic.category, level)
        user_prompt   = build_user_prompt(
            ghc_message=diagnostic.message,
            source_context=source_context,
            filename=_basename(uri),
            line=diagnostic.span.start_line if diagnostic.span else 0,
            col=diagnostic.span.start_col  if diagnostic.span else 0,
        )

        logger.debug(
            "Calling Groq [model=%s, category=%s, level=%s]",
            self._model, diagnostic.category.value, level.name,
        )

        raw_text = self._client.chat_complete(system=system_prompt, user=user_prompt)
        logger.debug("Groq response: %r", raw_text[:200])

        explanation, hint = parse_response(raw_text)

        return GHCDiagnostic(
            severity=diagnostic.severity,
            span=diagnostic.span,
            message=diagnostic.message,
            category=diagnostic.category,
            error_code=diagnostic.error_code,
            context_lines=diagnostic.context_lines,
            related=diagnostic.related,
            ai_explanation=explanation,
            ai_hint=hint,
        )


# ── Module helpers ────────────────────────────────────────────────────────

def _extract_source_context(source: str, line: int, radius: int) -> str:
    """
    Return up to `radius` lines above and below the given 1-indexed line,
    with line numbers prefixed and the error line marked.

    Example output (radius=2, error on line 5):
        3 |   foo x = x + 1
        4 |
        5 |   bar = foo True   ← error
        6 |
        7 |   baz = bar + 1
    """
    if not source:
        return ""
    lines  = source.splitlines()
    total  = len(lines)
    start  = max(0, line - 1 - radius)
    end    = min(total, line - 1 + radius + 1)
    parts  = []
    for i in range(start, end):
        lineno = i + 1
        marker = "  ← error" if lineno == line else ""
        parts.append(f"{lineno:4d} |  {lines[i]}{marker}")
    return "\n".join(parts)


def _basename(uri: str) -> str:
    """Extract filename from a file:// URI or plain path."""
    return Path(uri.removeprefix("file://")).name or uri