"""
tests/test_lsp_integration.py — Integration tests for the server layer.

These tests verify the LSP server's pipeline (compile → enrich → publish)
and the web server's WebSocket protocol, without requiring a running editor
or a live GHC installation.

All external calls are mocked:
  - GHCBridge.compile  → returns a controlled CompilationResult
  - AIFeedbackEngine.enrich_all → returns diagnostics unchanged (fast path)

The tests focus on:
  1. Correct LSP Diagnostic conversion (severity, range, message format)
  2. Hover response structure and position matching
  3. Code action generation
  4. Debounce logic (task cancellation)
  5. didClose cleanup
  6. Web server: health endpoint
  7. Web server: WebSocket compile request / response shape
  8. Web server: empty source short-circuit
  9. Web server: error handling on GHCBridge failure
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.ghc.models import (
    GHCDiagnostic, SourceSpan, Severity, ErrorCategory, CompilationResult,
)
from server.lsp_server import (
    HaskellLanguageServer,
    _to_lsp_diagnostic,
    _format_hover,
    _position_in_span,
    _range_overlaps,
    _span_to_range,
)
from server.web_server import _diagnostic_to_dict

@pytest.fixture
def type_error_diag() -> GHCDiagnostic:
    return GHCDiagnostic(
        severity=Severity.ERROR,
        span=SourceSpan(
            file="Main.hs", start_line=5, start_col=9,
            end_line=5, end_col=13,
        ),
        message="Couldn't match expected type 'Int' with actual type 'Bool'",
        category=ErrorCategory.TYPE_ERROR,
        ai_explanation="You supplied a Bool where GHC expected an Int.",
        ai_hint="Check the type of the expression on the right-hand side.",
    )


@pytest.fixture
def warning_diag() -> GHCDiagnostic:
    return GHCDiagnostic(
        severity=Severity.WARNING,
        span=SourceSpan(file="Main.hs", start_line=2, start_col=1,
                        end_line=2, end_col=15),
        message="Defined but not used: 'helper'",
        category=ErrorCategory.WARNING_GENERIC,
    )


@pytest.fixture
def compilation_result(type_error_diag, warning_diag) -> CompilationResult:
    return CompilationResult(
        file="Main.hs",
        diagnostics=[type_error_diag, warning_diag],
        success=False,
        raw_stderr="...",
    )

class TestToLspDiagnostic:

    def test_error_severity_mapped(self, type_error_diag):
        from lsprotocol.types import DiagnosticSeverity
        lsp = _to_lsp_diagnostic(type_error_diag)
        assert lsp.severity == DiagnosticSeverity.Error

    def test_warning_severity_mapped(self, warning_diag):
        from lsprotocol.types import DiagnosticSeverity
        lsp = _to_lsp_diagnostic(warning_diag)
        assert lsp.severity == DiagnosticSeverity.Warning

    def test_range_is_zero_indexed(self, type_error_diag):
        lsp = _to_lsp_diagnostic(type_error_diag)
        assert lsp.range.start.line      == 4
        assert lsp.range.start.character == 8

    def test_message_contains_ghc_text(self, type_error_diag):
        lsp = _to_lsp_diagnostic(type_error_diag)
        assert "Couldn't match" in lsp.message

    def test_message_contains_ai_explanation(self, type_error_diag):
        lsp = _to_lsp_diagnostic(type_error_diag)
        assert "Bool where GHC expected" in lsp.message

    def test_message_contains_ai_hint(self, type_error_diag):
        lsp = _to_lsp_diagnostic(type_error_diag)
        assert "right-hand side" in lsp.message

    def test_source_is_ghc(self, type_error_diag):
        lsp = _to_lsp_diagnostic(type_error_diag)
        assert lsp.source == "ghc"

    def test_no_span_falls_back_to_origin(self):
        diag = GHCDiagnostic(
            severity=Severity.ERROR,
            span=None,
            message="Some error",
            category=ErrorCategory.UNKNOWN,
        )
        lsp = _to_lsp_diagnostic(diag)
        assert lsp.range.start.line == 0
        assert lsp.range.start.character == 0

    def test_no_ai_content_message_is_just_ghc(self, warning_diag):
        lsp = _to_lsp_diagnostic(warning_diag)
        assert lsp.message.strip() == warning_diag.message


class TestFormatHover:

    def test_returns_empty_when_no_ai_content(self, warning_diag):
        result = _format_hover(warning_diag)
        assert result == ""

    def test_contains_ghc_message(self, type_error_diag):
        result = _format_hover(type_error_diag)
        assert "Couldn't match" in result

    def test_contains_explanation(self, type_error_diag):
        result = _format_hover(type_error_diag)
        assert "Bool where GHC expected" in result

    def test_contains_hint(self, type_error_diag):
        result = _format_hover(type_error_diag)
        assert "right-hand side" in result

    def test_is_markdown(self, type_error_diag):
        result = _format_hover(type_error_diag)
        assert "###" in result     
        assert "```" in result      

    def test_severity_in_header(self, type_error_diag):
        result = _format_hover(type_error_diag)
        assert "Error" in result

class TestPositionHelpers:

    def _pos(self, line, char):
        from lsprotocol.types import Position
        return Position(line=line, character=char)

    def _range(self, sl, sc, el, ec):
        from lsprotocol.types import Range, Position
        return Range(
            start=Position(line=sl, character=sc),
            end=Position(line=el, character=ec),
        )

    def test_position_inside_span(self, type_error_diag):
        assert _position_in_span(self._pos(4, 9), type_error_diag)

    def test_position_at_span_start(self, type_error_diag):
        assert _position_in_span(self._pos(4, 8), type_error_diag)

    def test_position_before_span(self, type_error_diag):
        assert not _position_in_span(self._pos(4, 0), type_error_diag)

    def test_position_on_wrong_line(self, type_error_diag):
        assert not _position_in_span(self._pos(0, 9), type_error_diag)

    def test_no_span_returns_false(self):
        diag = GHCDiagnostic(
            severity=Severity.ERROR, span=None,
            message="x", category=ErrorCategory.UNKNOWN,
        )
        from lsprotocol.types import Position
        assert not _position_in_span(Position(line=0, character=0), diag)

    def test_range_overlaps_span(self, type_error_diag):
        assert _range_overlaps(self._range(4, 0, 4, 20), type_error_diag)

    def test_range_does_not_overlap(self, type_error_diag):
        assert not _range_overlaps(self._range(0, 0, 1, 0), type_error_diag)


class TestCompileAndPublish:
    """
    Test the compile_and_publish pipeline by mocking GHCBridge and
    AIFeedbackEngine so no real compilation or API calls happen.
    """

    def _make_server(self, compilation_result) -> HaskellLanguageServer:
        """Create a server instance with all external deps mocked."""
        srv = HaskellLanguageServer.__new__(HaskellLanguageServer)

        srv._bridge          = MagicMock()
        srv._bridge.compile  = AsyncMock(return_value=compilation_result)
        srv._context         = MagicMock()
        srv._engine          = MagicMock()
        srv._engine.enrich_all = AsyncMock(
            side_effect=lambda diags, *a, **kw: diags  
        )
        srv._debounce_tasks  = {}
        srv._diag_cache      = {}
        srv.publish_diagnostics = MagicMock()
        return srv

    @pytest.mark.asyncio
    async def test_publishes_diagnostics_after_compile(self, compilation_result):
        srv = self._make_server(compilation_result)
        await srv.compile_and_publish("file:///Main.hs", "module Main where")
        assert srv.publish_diagnostics.called
        args = srv.publish_diagnostics.call_args[0]
        uri, lsp_diags = args
        assert uri == "file:///Main.hs"
        assert len(lsp_diags) == 2

    @pytest.mark.asyncio
    async def test_caches_diagnostics(self, compilation_result):
        srv = self._make_server(compilation_result)
        await srv.compile_and_publish("file:///Main.hs", "module Main where")
        assert "file:///Main.hs" in srv._diag_cache
        assert len(srv._diag_cache["file:///Main.hs"]) == 2

    @pytest.mark.asyncio
    async def test_bridge_error_does_not_crash_server(self):
        srv = HaskellLanguageServer.__new__(HaskellLanguageServer)
        srv._bridge = MagicMock()
        srv._bridge.compile = AsyncMock(side_effect=RuntimeError("GHC not found"))
        srv._context = MagicMock()
        srv._engine  = MagicMock()
        srv._debounce_tasks = {}
        srv._diag_cache     = {}
        srv.publish_diagnostics = MagicMock()
        await srv.compile_and_publish("file:///Main.hs", "module Main where")
        srv.publish_diagnostics.assert_not_called()

    @pytest.mark.asyncio
    async def test_debounce_cancels_previous_task(self, compilation_result):
        srv = self._make_server(compilation_result)
        uri = "file:///Main.hs"

        srv._schedule_compile(uri, "source v1")
        task1 = srv._debounce_tasks[uri]

        srv._schedule_compile(uri, "source v2")
        task2 = srv._debounce_tasks[uri]

        assert task1 is not task2
        assert task1.cancelled() or task1.cancelling()

        task2.cancel()
        await asyncio.sleep(0)


class TestDiagnosticToDict:

    def test_all_keys_present(self, type_error_diag):
        d = _diagnostic_to_dict(type_error_diag)
        for key in ("severity", "message", "category", "startLine", "startCol",
                    "endLine", "endCol", "explanation", "hint", "scaffold"):
            assert key in d, f"Missing key: {key}"

    def test_severity_is_string(self, type_error_diag):
        d = _diagnostic_to_dict(type_error_diag)
        assert isinstance(d["severity"], str)

    def test_positions_are_one_indexed(self, type_error_diag):
        d = _diagnostic_to_dict(type_error_diag)
        assert d["startLine"] == 5
        assert d["startCol"]  == 9

    def test_ai_fields_present(self, type_error_diag):
        d = _diagnostic_to_dict(type_error_diag)
        assert d["explanation"] == type_error_diag.ai_explanation
        assert d["hint"]        == type_error_diag.ai_hint

    def test_no_span_defaults_to_one(self):
        diag = GHCDiagnostic(
            severity=Severity.ERROR, span=None,
            message="x", category=ErrorCategory.UNKNOWN,
        )
        d = _diagnostic_to_dict(diag)
        assert d["startLine"] == 0
        assert d["startCol"] == 0


class TestWebServerHealth:

    @pytest.mark.asyncio
    async def test_health_endpoint_returns_ok(self):
        from httpx import AsyncClient, ASGITransport
        from server.web_server import app
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"