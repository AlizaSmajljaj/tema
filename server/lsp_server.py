"""
lsp_server.py — Language Server Protocol server.

This module implements the LSP server that sits between the code editor
(VS Code or any LSP-capable editor) and the rest of the system. It uses
the pygls library, which provides a complete JSON-RPC / LSP framework,
allowing us to focus entirely on the Haskell-specific logic rather than
protocol plumbing.

Architecture overview
---------------------

    Editor (VS Code)
         │  JSON-RPC over stdio
         ▼
    HaskellLanguageServer   ← this file
         │
         ├─ GHCBridge          (server/ghc/bridge.py)
         │    └─ GHCParser     (server/ghc/parser.py)
         │
         └─ AIFeedbackEngine   (server/ai/engine.py)
              ├─ ContextManager (server/ai/context.py)
              └─ Prompts        (server/ai/prompts.py)

LSP features implemented
------------------------
• textDocument/didOpen     — compile on open, publish diagnostics
• textDocument/didChange   — debounced compile on every keystroke
• textDocument/didSave     — compile immediately on save
• textDocument/didClose    — clear diagnostics, reset AI context
• textDocument/hover       — show AI explanation on hover over an error
• textDocument/codeAction  — offer "Explain this error" quick-fix
• workspace/didChangeConfiguration — reload GHC path / model at runtime

Debouncing
----------
Compiling on every keystroke would overwhelm GHC and the Groq API.
We debounce didChange events with a 600 ms delay. Only the most recent
change triggers a compilation — intermediate keystrokes are discarded.

Error-free files
----------------
When a file compiles without errors, we publish an empty diagnostics list.
This clears any stale red squiggles from the previous compilation.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from lsprotocol.types import (
    InitializeParams, InitializeResult, ServerCapabilities,
    TextDocumentSyncKind, TextDocumentSyncOptions,
    
    Diagnostic, DiagnosticSeverity, Position, Range,
    PublishDiagnosticsParams,

    Hover, HoverParams, MarkupContent, MarkupKind,

    CodeAction, CodeActionKind, CodeActionParams,
    Command,

    DocumentHighlight, DocumentHighlightKind, DocumentHighlightParams,

    DidOpenTextDocumentParams, DidChangeTextDocumentParams,
    DidSaveTextDocumentParams, DidCloseTextDocumentParams,

    TextEdit, WorkspaceEdit,

    TEXT_DOCUMENT_DID_OPEN, TEXT_DOCUMENT_DID_CHANGE,
    TEXT_DOCUMENT_DID_SAVE, TEXT_DOCUMENT_DID_CLOSE,
    TEXT_DOCUMENT_HOVER, TEXT_DOCUMENT_CODE_ACTION,
    TEXT_DOCUMENT_DOCUMENT_HIGHLIGHT,
    INITIALIZE,
)
from pygls.server import LanguageServer

from server.ghc.bridge import GHCBridge
from server.ghc.models import GHCDiagnostic, Severity
from server.ai.engine import AIFeedbackEngine
from server.ai.context import ContextManager

load_dotenv()

logger = logging.getLogger(__name__)


SERVER_NAME    = "haskell-ai-lsp"
SERVER_VERSION = "0.1.0"

DEBOUNCE_DELAY = 0.6

MAX_AI_ENRICHMENTS = 5



class HaskellLanguageServer(LanguageServer):
    """
    The main language server class.

    Extends pygls.LanguageServer with Haskell-specific state:
    - GHCBridge for running the compiler
    - AIFeedbackEngine for AI-powered explanations
    - A debounce task table to coalesce rapid document changes
    - A diagnostic cache for powering hover responses

    All LSP handlers are registered as methods on this class using
    pygls's @server.feature() decorator.
    """

    def __init__(self) -> None:
        super().__init__(SERVER_NAME, SERVER_VERSION)

        self._bridge  = GHCBridge()
        self._context = ContextManager()
        self._engine  = AIFeedbackEngine(context_manager=self._context)

        self._debounce_tasks: dict[str, asyncio.Task] = {}

        self._diag_cache: dict[str, list[GHCDiagnostic]] = {}

        logger.info("%s %s initialised", SERVER_NAME, SERVER_VERSION)


    def reconfigure(self, settings: dict) -> None:
        """
        Apply new settings from workspace/didChangeConfiguration.

        Supports:
          haskellAiLsp.ghcPath   — path to GHC executable
          haskellAiLsp.groqModel — Groq model to use for AI feedback
        """
        ghc_path = settings.get("ghcPath") or os.environ.get("GHC_PATH")
        if ghc_path:
            self._bridge = GHCBridge(ghc_path=ghc_path)
            logger.info("GHC path updated to %s", ghc_path)

        model = settings.get("groqModel") or os.environ.get("GROQ_MODEL")
        if model:
            self._engine = AIFeedbackEngine(
                model=model,
                context_manager=self._context,
            )
            logger.info("AI model updated to %s", model)


    async def compile_and_publish(self, uri: str, source: str) -> None:
        """
        Run the full pipeline for one document:
          1. Compile with GHC
          2. Enrich top-N diagnostics with AI
          3. Convert to LSP Diagnostic objects
          4. Publish to the editor via publishDiagnostics

        This method is always called from an async context (either directly
        from a handler or from a debounced task).
        """
        logger.info("Compiling %s", uri)

        try:
            result = await self._bridge.compile(source, uri)
        except Exception as exc:
            logger.error("GHCBridge error for %s: %s", uri, exc)
            return

        diags = result.diagnostics

        to_enrich = diags[:MAX_AI_ENRICHMENTS]
        rest      = diags[MAX_AI_ENRICHMENTS:]

        enriched = await self._engine.enrich_all(to_enrich, source, uri)
        all_diags = enriched + rest

        self._diag_cache[uri] = all_diags

        lsp_diags = [_to_lsp_diagnostic(d) for d in all_diags]
        self.publish_diagnostics(uri, lsp_diags)

        logger.info(
            "Published %d diagnostics for %s (%d AI-enriched)",
            len(lsp_diags), uri, len(enriched),
        )

    async def _debounced_compile(self, uri: str, source: str) -> None:
        """
        Wait DEBOUNCE_DELAY seconds, then compile.

        If a newer task cancels this one before the delay expires,
        asyncio.CancelledError is raised and swallowed silently.
        This is the standard asyncio debounce pattern.
        """
        try:
            await asyncio.sleep(DEBOUNCE_DELAY)
            await self.compile_and_publish(uri, source)
        except asyncio.CancelledError:
            pass 

    def _schedule_compile(self, uri: str, source: str) -> None:
        """
        Cancel any pending debounce task for this URI and schedule a new one.
        Called on every didChange event.
        """
        old = self._debounce_tasks.get(uri)
        if old and not old.done():
            old.cancel()

        loop = asyncio.get_event_loop()
        task = loop.create_task(self._debounced_compile(uri, source))
        self._debounce_tasks[uri] = task



server = HaskellLanguageServer()



@server.feature(TEXT_DOCUMENT_DID_OPEN)
async def did_open(ls: HaskellLanguageServer, params: DidOpenTextDocumentParams) -> None:
    """Compile immediately when a Haskell file is opened."""
    uri    = params.text_document.uri
    source = params.text_document.text
    logger.debug("didOpen: %s", uri)
    await ls.compile_and_publish(uri, source)


@server.feature(TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls: HaskellLanguageServer, params: DidChangeTextDocumentParams) -> None:
    """
    Debounced compile on every change event.

    pygls calls this synchronously so we use _schedule_compile rather
    than awaiting directly. The debounce task runs on the event loop.
    """
    uri    = params.text_document.uri
    source = params.content_changes[-1].text 
    logger.debug("didChange: %s (scheduling debounced compile)", uri)
    ls._schedule_compile(uri, source)


@server.feature(TEXT_DOCUMENT_DID_SAVE)
async def did_save(ls: HaskellLanguageServer, params: DidSaveTextDocumentParams) -> None:
    """
    Compile immediately on save, bypassing the debounce.

    The editor sends the saved text in params.text if the server negotiated
    includeText=True during initialization. We fall back to the cache if not.
    """
    uri = params.text_document.uri
    logger.debug("didSave: %s", uri)

    old = ls._debounce_tasks.get(uri)
    if old and not old.done():
        old.cancel()

    source = getattr(params, "text", None)
    if source is None:
        doc = ls.workspace.get_document(uri)
        source = doc.source if doc else ""

    if source:
        await ls.compile_and_publish(uri, source)


@server.feature(TEXT_DOCUMENT_DID_CLOSE)
def did_close(ls: HaskellLanguageServer, params: DidCloseTextDocumentParams) -> None:
    """
    Clean up state when a document is closed.

    - Cancel any pending debounce task
    - Clear the diagnostic cache
    - Reset the AI context for this document (experience level resets)
    - Publish empty diagnostics to clear squiggles
    """
    uri = params.text_document.uri
    logger.debug("didClose: %s", uri)

    old = ls._debounce_tasks.pop(uri, None)
    if old and not old.done():
        old.cancel()

    ls._diag_cache.pop(uri, None)
    ls._context.reset(uri)
    ls.publish_diagnostics(uri, [])


@server.feature(TEXT_DOCUMENT_HOVER)
def hover(ls: HaskellLanguageServer, params: HoverParams) -> Optional[Hover]:
    """
    Return an AI explanation when the user hovers over a diagnostic location.

    We look up the cached diagnostics for the document and find the first
    one whose source span contains the hover position. If that diagnostic
    has an AI explanation, we return it as a markdown hover card.

    Returns None if there is no diagnostic at the cursor position, which
    tells the editor to show nothing.
    """
    uri      = params.text_document.uri
    position = params.position
    cached   = ls._diag_cache.get(uri, [])

    for diag in cached:
        if not diag.span:
            continue
        if _position_in_span(position, diag):
            content = _format_hover(diag)
            if content:
                return Hover(
                    contents=MarkupContent(kind=MarkupKind.Markdown, value=content),
                    range=_span_to_range(diag),
                )

    return None


@server.feature(TEXT_DOCUMENT_CODE_ACTION)
def code_action(
    ls: HaskellLanguageServer, params: CodeActionParams
) -> list[CodeAction]:
    """
    Return code actions for diagnostics at the cursor position.

    Currently offers one action per diagnostic: "Explain this error (AI)",
    which is a client-side command that triggers a hover popup. This gives
    students a discoverable entry point to the AI explanations even if they
    don't know to hover.
    """
    uri     = params.text_document.uri
    range_  = params.range
    cached  = ls._diag_cache.get(uri, [])
    actions = []

    for diag in cached:
        if not diag.span:
            continue
        if not _range_overlaps(range_, diag):
            continue
        if not (diag.ai_explanation or diag.ai_hint):
            continue

        preview = (diag.ai_explanation or "")[:60]
        if len(diag.ai_explanation or "") > 60:
            preview += "…"

        actions.append(
            CodeAction(
                title=f"💡 AI: {preview}" if preview else "💡 Explain this error (AI)",
                kind=CodeActionKind.QuickFix,
                diagnostics=[_to_lsp_diagnostic(diag)],
                command=Command(
                    title="Show AI explanation",
                    command="haskell-ai-lsp.showExplanation",
                    arguments=[uri, diag.span.start_line, diag.ai_explanation, diag.ai_hint],
                ),
            )
        )

    return actions


@server.feature(TEXT_DOCUMENT_DOCUMENT_HIGHLIGHT)
def document_highlight(
    ls: HaskellLanguageServer, params: DocumentHighlightParams
) -> list[DocumentHighlight]:
    """
    Highlight the exact error token in the editor when the cursor is on it.

    When the student places their cursor on or near a diagnostic, this handler
    returns a DocumentHighlight that tells the editor to visually mark the
    precise source range that GHC flagged. This makes it immediately obvious
    which token caused the error — particularly useful for type errors where
    the problem token may be a single variable or operator in a long expression.

    Uses DocumentHighlightKind.Read (yellow/blue tint depending on theme)
    rather than Write, since the student is looking at the problematic
    expression, not writing to it.
    """
    uri      = params.text_document.uri
    position = params.position
    cached   = ls._diag_cache.get(uri, [])

    highlights = []
    for diag in cached:
        if not _position_in_span(position, diag):
            continue
        span_range = _span_to_range(diag)
        if span_range:
            highlights.append(
                DocumentHighlight(
                    range=span_range,
                    kind=DocumentHighlightKind.Read,
                )
            )

    return highlights



def _to_lsp_diagnostic(d: GHCDiagnostic) -> Diagnostic:
    """
    Convert a GHCDiagnostic to an LSP Diagnostic.

    The AI explanation and hint are embedded in the diagnostic message
    so they appear in the editor's Problems panel even without hover.
    The hover handler provides a richer formatted version.
    """
    parts = [d.message]
    if d.ai_explanation:
        parts.append(f"\n💡 {d.ai_explanation}")
    if d.ai_hint:
        parts.append(f"🔑 {d.ai_hint}")
    message = "\n".join(parts)

    severity_map = {
        Severity.ERROR:   DiagnosticSeverity.Error,
        Severity.WARNING: DiagnosticSeverity.Warning,
        Severity.INFO:    DiagnosticSeverity.Information,
        Severity.HINT:    DiagnosticSeverity.Hint,
    }
    lsp_severity = severity_map.get(d.severity, DiagnosticSeverity.Error)

    if d.span:
        lsp_range = Range(
            start=Position(line=d.span.start_line - 1, character=d.span.start_col - 1),
            end=Position(
                line=d.span.end_line - 1,
                character=max(d.span.end_col - 1, d.span.start_col),
            ),
        )
    else:
        lsp_range = Range(
            start=Position(line=0, character=0),
            end=Position(line=0, character=1),
        )

    return Diagnostic(
        range=lsp_range,
        message=message,
        severity=lsp_severity,
        source="ghc",
        code=d.error_code,
    )


def _format_hover(d: GHCDiagnostic) -> str:
    """
    Build a markdown hover card from a GHCDiagnostic.

    Structure:
        ### GHC Error  (or Warning / Info)
        ```
        <raw GHC message>
        ```
        **What this means:**
        <AI explanation>

        **Hint:**
        <AI hint>
    """
    lines: list[str] = []

    kind = d.severity.name.capitalize()
    lines.append(f"### GHC {kind}")
    lines.append("")

    lines.append("```")
    lines.append(d.message.strip())
    lines.append("```")
    lines.append("")

    if d.ai_explanation:
        lines.append("**🤔 Let's think about this:**")
        lines.append("")
        lines.append(d.ai_explanation)
        lines.append("")

    if d.ai_hint:
        lines.append("**💭 Something to consider:**")
        lines.append("")
        lines.append(d.ai_hint)
        lines.append("")

    if not d.ai_explanation and not d.ai_hint:
        return ""  
    return "\n".join(lines)



def _position_in_span(position: Position, d: GHCDiagnostic) -> bool:
    """
    Return True if the LSP cursor position falls within the GHC diagnostic span.

    LSP positions are 0-indexed; GHC spans are 1-indexed.
    We convert the LSP position to 1-indexed before comparing.
    """
    if not d.span:
        return False
    line = position.line + 1
    col  = position.character + 1
    if line < d.span.start_line or line > d.span.end_line:
        return False
    if line == d.span.start_line and col < d.span.start_col:
        return False
    if line == d.span.end_line and col > d.span.end_col:
        return False
    return True


def _range_overlaps(lsp_range: Range, d: GHCDiagnostic) -> bool:
    """Return True if the LSP range overlaps the diagnostic's span."""
    if not d.span:
        return False
    r_start_line = lsp_range.start.line + 1
    r_end_line   = lsp_range.end.line   + 1
    return not (r_end_line < d.span.start_line or r_start_line > d.span.end_line)


def _span_to_range(d: GHCDiagnostic) -> Optional[Range]:
    """Convert a GHCDiagnostic's SourceSpan to an LSP Range."""
    if not d.span:
        return None
    return Range(
        start=Position(line=d.span.start_line - 1, character=d.span.start_col - 1),
        end=Position(line=d.span.end_line - 1,   character=d.span.end_col - 1),
    )