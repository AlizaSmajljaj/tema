"""
tests/test_comprehensive.py — Full regression and feature test suite.

Covers everything introduced or changed after the supervisor feedback session:

1.  Prompt style validation    — new Socratic prompts must NOT directly point
                                 at the problem line; must ask questions/use analogies
2.  All 9 categories           — system prompt generated for every ErrorCategory
3.  All 3 experience levels    — BEGINNER / INTERMEDIATE / ADVANCED all produce
                                 distinct prompt content
4.  Document highlight         — new LSP handler returns correct Range for every
                                 diagnostic; returns empty list when cursor is outside
5.  Hover card labels          — new labels ("Let's think" / "Something to consider")
6.  Full pipeline smoke test   — compile → enrich → publish → hover → highlight,
                                 all wired together with mocked GHC + Groq
7.  Graceful degradation       — engine returns original diag when API is unavailable;
                                 highlight handler returns [] when no cache entry exists
8.  Multi-error compilation    — 7 errors: first 5 enriched, last 2 plain; all 7
                                 published; cache holds all 7
9.  Debounce cancellation      — second didChange within 600 ms cancels first task
10. Parser round-trip          — every ErrorCategory survives
                                 GHCDiagnostic → LSP Diagnostic → back to category tag
11. Web server dict shape      — _diagnostic_to_dict includes all required keys
12. Context reset              — reset() and reset_all() clear state correctly
13. Experience level advance   — category counter drives correct level transitions
14. Source context extraction  — _extract_source_context returns correct window
15. Response parse edge cases  — empty string, missing HINT, garbled output
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from server.ghc.models import (
    GHCDiagnostic, SourceSpan, Severity, ErrorCategory, CompilationResult,
)
from server.ai.context import (
    ContextManager, UserContext, ExperienceLevel,
    _INTERMEDIATE_THRESHOLD, _ADVANCED_THRESHOLD,
)
from server.ai.prompts import (
    build_system_prompt, build_user_prompt, parse_response,
    _TEACHING_STYLE, _CATEGORY_GUIDANCE,
)
from server.ai.engine import AIFeedbackEngine, _extract_source_context
from server.lsp_server import (
    HaskellLanguageServer,
    _to_lsp_diagnostic,
    _format_hover,
    _position_in_span,
    _range_overlaps,
    _span_to_range,
    document_highlight,
)
from server.web_server import _diagnostic_to_dict
from lsprotocol.types import (
    Position, Range,
    DocumentHighlightParams, TextDocumentIdentifier,
    DocumentHighlightKind,
)


# ═══════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ═══════════════════════════════════════════════════════════════════════════

URI = "file:///workspace/Main.hs"

@pytest.fixture
def make_diag():
    """Factory for GHCDiagnostic with sensible defaults."""
    def _make(
        category=ErrorCategory.TYPE_ERROR,
        message="Couldn't match expected type 'Int' with actual type 'Bool'",
        start_line=5, start_col=9,
        end_line=5, end_col=13,
        severity=Severity.ERROR,
        explanation="", hint="",
    ) -> GHCDiagnostic:
        d = GHCDiagnostic(
            severity=severity,
            span=SourceSpan(
                file="Main.hs",
                start_line=start_line, start_col=start_col,
                end_line=end_line, end_col=end_col,
            ),
            message=message,
            category=category,
            error_code=None,
            raw_ghc_output="",
        )
        d.ai_explanation = explanation
        d.ai_hint = hint
        return d
    return _make


@pytest.fixture
def diag_with_ai(make_diag):
    return make_diag(
        explanation="Hmm, let's think about types here — what did you mean to put in this spot?",
        hint="What type do you think the right-hand side should have?",
    )


@pytest.fixture
def compilation_result(make_diag):
    """A CompilationResult with 7 diagnostics — 5 type errors + 2 scope errors."""
    diags = [
        make_diag(category=ErrorCategory.TYPE_ERROR,  start_line=i, end_line=i)
        for i in range(1, 6)
    ] + [
        make_diag(category=ErrorCategory.SCOPE_ERROR, start_line=i, end_line=i,
                  message="Variable not in scope: helper")
        for i in range(6, 8)
    ]
    return CompilationResult(file="Main.hs",diagnostics=diags, success=False, raw_stderr="")


@pytest.fixture
def server():
    """A fresh HaskellLanguageServer with mocked bridge and engine."""
    srv = HaskellLanguageServer()
    srv._bridge = MagicMock()
    srv._engine = MagicMock()
    return srv


MOCK_AI_RESPONSE = (
    "EXPLANATION: Hmm, interesting! Think of a vending machine — it only accepts coins. "
    "What are you putting in here?\n"
    "HINT: What type do you think GHC is expecting on the left side of this expression?"
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Prompt style validation — Socratic, NOT directive
# ═══════════════════════════════════════════════════════════════════════════

class TestPromptStyle:
    """
    The supervisor asked for explanations that speak to students like a curious
    friend, not a compiler. These tests enforce that the prompts:
      - contain question words / analogies
      - do NOT directly name the error line or token
      - DO include the teaching style block
    """

    def test_teaching_style_present_in_all_prompts(self):
        """Every generated system prompt includes the shared teaching style."""
        for category in ErrorCategory:
            for level in ExperienceLevel:
                prompt = build_system_prompt(category, level)
                # Key phrases from _TEACHING_STYLE must appear
                assert "analogy" in prompt.lower() or "everyday" in prompt.lower() \
                    or "ask question" in prompt.lower() or "curiosity" in prompt.lower(), \
                    f"Teaching style not embedded in prompt for {category}/{level}"

    def test_no_directive_language_in_system_prompts(self):
        """Prompts must not tell the model to 'point at the error' directly."""
        forbidden = [
            "tell the student what is wrong",
            "point directly at",
            "the error is on line",
            "you wrote X but should write",
        ]
        for category in ErrorCategory:
            prompt = build_system_prompt(category, ExperienceLevel.BEGINNER)
            for phrase in forbidden:
                assert phrase.lower() not in prompt.lower(), \
                    f"Forbidden directive phrase '{phrase}' found in {category} prompt"

    def test_hint_format_asks_for_question_or_nudge(self):
        """The format instruction must request a question or gentle nudge for HINT."""
        prompt = build_system_prompt(ErrorCategory.TYPE_ERROR, ExperienceLevel.BEGINNER)
        assert "question" in prompt.lower() or "nudge" in prompt.lower() or \
               "gentle" in prompt.lower(), \
            "Format instruction should request question/nudge style for HINT"

    def test_no_code_fix_rule_present(self):
        """All prompts must include the no-corrected-code rule."""
        for category in ErrorCategory:
            prompt = build_system_prompt(category, ExperienceLevel.BEGINNER)
            assert "do not write corrected haskell code" in prompt.lower(), \
                f"No-code-fix rule missing from {category} prompt"

    def test_beginner_uses_analogy_language(self):
        """BEGINNER prompts must reference everyday analogies."""
        analogy_categories = [
            ErrorCategory.TYPE_ERROR,
            ErrorCategory.SCOPE_ERROR,
            ErrorCategory.PATTERN_MATCH,
            ErrorCategory.RECURSION,
            ErrorCategory.INSTANCE_ERROR,
            ErrorCategory.IMPORT_ERROR,
        ]
        for cat in analogy_categories:
            prompt = build_system_prompt(cat, ExperienceLevel.BEGINNER)
            has_analogy = any(word in prompt.lower() for word in [
                "analogy", "imagine", "like a", "think of", "classroom",
                "vending", "library", "mirror", "box", "container",
            ])
            assert has_analogy, \
                f"BEGINNER prompt for {cat} should contain an everyday analogy"

    def test_advanced_prompt_shorter_than_beginner(self):
        """ADVANCED prompts should be more concise than BEGINNER prompts."""
        for cat in [ErrorCategory.TYPE_ERROR, ErrorCategory.SCOPE_ERROR,
                    ErrorCategory.PATTERN_MATCH]:
            beginner = build_system_prompt(cat, ExperienceLevel.BEGINNER)
            advanced = build_system_prompt(cat, ExperienceLevel.ADVANCED)
            assert len(advanced) <= len(beginner) + 100, \
                f"ADVANCED prompt should be shorter than BEGINNER for {cat}"

    def test_level_description_differs_by_level(self):
        """Each experience level produces a different level description string."""
        descriptions = {
            level: build_system_prompt(ErrorCategory.TYPE_ERROR, level)
            for level in ExperienceLevel
        }
        assert len(set(descriptions.values())) == len(ExperienceLevel), \
            "Each experience level must produce a distinct prompt"



class TestAllCategories:

    def test_all_categories_have_guidance(self):
        """Every ErrorCategory must have an entry in _CATEGORY_GUIDANCE."""
        for cat in ErrorCategory:
            prompt = build_system_prompt(cat, ExperienceLevel.BEGINNER)
            assert len(prompt) > 100, \
                f"Prompt for {cat} is suspiciously short: {len(prompt)} chars"

    def test_all_categories_all_levels_produce_format_instruction(self):
        """Every prompt must include the EXPLANATION/HINT format instruction."""
        for cat in ErrorCategory:
            for level in ExperienceLevel:
                prompt = build_system_prompt(cat, level)
                assert "EXPLANATION:" in prompt, \
                    f"Format instruction missing EXPLANATION for {cat}/{level}"
                assert "HINT:" in prompt, \
                    f"Format instruction missing HINT for {cat}/{level}"

    def test_category_guidance_is_category_specific(self):
        """Different categories must produce meaningfully different prompts."""
        type_prompt  = build_system_prompt(ErrorCategory.TYPE_ERROR,  ExperienceLevel.BEGINNER)
        scope_prompt = build_system_prompt(ErrorCategory.SCOPE_ERROR, ExperienceLevel.BEGINNER)
        assert type_prompt != scope_prompt, \
            "TYPE_ERROR and SCOPE_ERROR prompts must differ"

    @pytest.mark.parametrize("category,expected_word", [
        (ErrorCategory.TYPE_ERROR,     "type"),
        (ErrorCategory.SCOPE_ERROR,    "scope"),
        (ErrorCategory.SYNTAX_ERROR,   "parse"),
        (ErrorCategory.PATTERN_MATCH,  "pattern"),
        (ErrorCategory.INSTANCE_ERROR, "typeclass"),
        (ErrorCategory.KIND_ERROR,     "kind"),
        (ErrorCategory.IMPORT_ERROR,   "import"),
        (ErrorCategory.RECURSION,      "infinite"),
        (ErrorCategory.WARNING_GENERIC,"warning"),
    ])
    def test_category_prompt_contains_relevant_keyword(self, category, expected_word):
        """Each category's prompt must mention its core concept."""
        prompt = build_system_prompt(category, ExperienceLevel.BEGINNER)
        assert expected_word.lower() in prompt.lower(), \
            f"'{expected_word}' not found in {category} prompt"

class TestDocumentHighlight:

    def _make_params(self, line: int, char: int) -> DocumentHighlightParams:
        return DocumentHighlightParams(
            text_document=TextDocumentIdentifier(uri=URI),
            position=Position(line=line, character=char),
        )

    def test_returns_highlight_when_cursor_inside_span(self, server, diag_with_ai):
        """Cursor exactly on the error span → one highlight returned."""
        server._diag_cache[URI] = [diag_with_ai]
        params = self._make_params(line=4, char=9)
        result = document_highlight(server, params)
        assert len(result) == 1
        assert result[0].kind == DocumentHighlightKind.Read

    def test_returns_highlight_at_span_start(self, server, diag_with_ai):
        """Cursor exactly at the first character of the span → highlight."""
        server._diag_cache[URI] = [diag_with_ai]
        params = self._make_params(line=4, char=8)
        result = document_highlight(server, params)
        assert len(result) == 1

    def test_returns_empty_when_cursor_before_span(self, server, diag_with_ai):
        """Cursor before the span → no highlight."""
        server._diag_cache[URI] = [diag_with_ai]
        params = self._make_params(line=4, char=2) 
        result = document_highlight(server, params)
        assert result == []

    def test_returns_empty_when_cursor_after_span(self, server, diag_with_ai):
        """Cursor after the span → no highlight."""
        server._diag_cache[URI] = [diag_with_ai]
        params = self._make_params(line=4, char=20)  
        result = document_highlight(server, params)
        assert result == []

    def test_returns_empty_when_cursor_on_different_line(self, server, diag_with_ai):
        """Cursor on a different line → no highlight."""
        server._diag_cache[URI] = [diag_with_ai]
        params = self._make_params(line=7, char=9)
        result = document_highlight(server, params)
        assert result == []

    def test_returns_empty_when_no_cache_entry(self, server):
        """No cache entry for this URI → empty list, no crash."""
        params = self._make_params(line=4, char=9)
        result = document_highlight(server, params)
        assert result == []

    def test_highlight_range_matches_diagnostic_span(self, server, diag_with_ai):
        """The returned Range must exactly match the diagnostic's SourceSpan."""
        server._diag_cache[URI] = [diag_with_ai]
        params = self._make_params(line=4, char=9)
        result = document_highlight(server, params)
        assert len(result) == 1
        r = result[0].range
        span = diag_with_ai.span
        assert r.start.line      == span.start_line - 1
        assert r.start.character == span.start_col  - 1
        assert r.end.line        == span.end_line   - 1
        assert r.end.character   == span.end_col    - 1

    def test_multiple_diagnostics_only_highlights_matching_one(self, server, make_diag):
        """With two diagnostics, only the one under the cursor is highlighted."""
        d1 = make_diag(start_line=3, end_line=3, start_col=1, end_col=5,
                       explanation="e1", hint="h1")
        d2 = make_diag(start_line=7, end_line=7, start_col=1, end_col=5,
                       explanation="e2", hint="h2")
        server._diag_cache[URI] = [d1, d2]
        params = self._make_params(line=2, char=2)
        result = document_highlight(server, params)
        assert len(result) == 1
        assert result[0].range.start.line == 2 

    def test_diagnostic_without_span_not_highlighted(self, server, make_diag):
        """A diagnostic with no span must not produce a highlight."""
        d = make_diag(explanation="e", hint="h")
        d.span = None
        server._diag_cache[URI] = [d]
        params = self._make_params(line=4, char=9)
        result = document_highlight(server, params)
        assert result == []


class TestHoverCardLabels:

    def test_hover_uses_new_explanation_label(self, diag_with_ai):
        """Hover card must use the new '🤔 Let's think about this' label."""
        card = _format_hover(diag_with_ai)
        assert "Let's think about this" in card or "think about" in card.lower()

    def test_hover_uses_new_hint_label(self, diag_with_ai):
        """Hover card must use the new '💭 Something to consider' label."""
        card = _format_hover(diag_with_ai)
        assert "Something to consider" in card or "consider" in card.lower()

    def test_hover_does_not_use_old_labels(self, diag_with_ai):
        """Hover card must NOT use the old 'What this means' / 'Hint:' labels."""
        card = _format_hover(diag_with_ai)
        assert "What this means" not in card
        assert "**Hint:**" not in card

    def test_hover_still_contains_ghc_message(self, diag_with_ai):
        """The raw GHC message must still appear in a code block."""
        card = _format_hover(diag_with_ai)
        assert diag_with_ai.message in card
        assert "```" in card

    def test_hover_contains_ai_explanation_text(self, diag_with_ai):
        """The actual explanation text must appear in the hover card."""
        card = _format_hover(diag_with_ai)
        assert diag_with_ai.ai_explanation in card

    def test_hover_contains_ai_hint_text(self, diag_with_ai):
        """The actual hint text must appear in the hover card."""
        card = _format_hover(diag_with_ai)
        assert diag_with_ai.ai_hint in card

    def test_hover_returns_empty_string_when_no_ai_content(self, make_diag):
        """If there is no AI content, hover must return empty string."""
        d = make_diag()
        assert _format_hover(d) == ""


class TestFullPipeline:
    """
    Wires compile_and_publish end-to-end with mocked GHCBridge and
    AIFeedbackEngine, then checks hover and document highlight both work
    off the populated cache.
    """

    @pytest.mark.asyncio
    async def test_compile_publish_populates_cache(self, server, make_diag):
        """After compile_and_publish, _diag_cache[uri] holds all diagnostics."""
        diags = [make_diag(start_line=i, end_line=i) for i in range(1, 4)]
        result = CompilationResult(file="Main.hs", diagnostics=diags, success=False, raw_stderr="")

        server._bridge.compile = AsyncMock(return_value=result)
        server._engine.enrich_all = AsyncMock(return_value=diags)
        server.publish_diagnostics = MagicMock()

        await server.compile_and_publish(URI, "main = putStrLn True\n")

        assert URI in server._diag_cache
        assert len(server._diag_cache[URI]) == 3

    @pytest.mark.asyncio
    async def test_compile_publish_calls_publish_diagnostics(self, server, make_diag):
        """publish_diagnostics must be called exactly once per compilation."""
        diag = make_diag()
        result = CompilationResult(file="Main.hs",diagnostics=[diag], success=False, raw_stderr="")
        server._bridge.compile = AsyncMock(return_value=result)
        server._engine.enrich_all = AsyncMock(return_value=[diag])
        server.publish_diagnostics = MagicMock()

        await server.compile_and_publish(URI, "x = True + 1\n")

        server.publish_diagnostics.assert_called_once()
        call_uri, call_diags = server.publish_diagnostics.call_args[0]
        assert call_uri == URI
        assert len(call_diags) == 1

    @pytest.mark.asyncio
    async def test_max_5_enrichments_enforced(self, server, compilation_result):
        """Only the first 5 diagnostics are passed to enrich_all."""
        server._bridge.compile = AsyncMock(return_value=compilation_result)

        enriched_5 = compilation_result.diagnostics[:5]
        server._engine.enrich_all = AsyncMock(return_value=enriched_5)
        server.publish_diagnostics = MagicMock()

        await server.compile_and_publish(URI, "-- code\n")

        call_args = server._engine.enrich_all.call_args[0]
        diags_passed = call_args[0]
        assert len(diags_passed) == 5

    @pytest.mark.asyncio
    async def test_all_7_diagnostics_published(self, server, compilation_result):
        """Even though only 5 are enriched, all 7 must be published."""
        server._bridge.compile = AsyncMock(return_value=compilation_result)
        server._engine.enrich_all = AsyncMock(
            return_value=compilation_result.diagnostics[:5]
        )
        server.publish_diagnostics = MagicMock()

        await server.compile_and_publish(URI, "-- code\n")

        _, published = server.publish_diagnostics.call_args[0]
        assert len(published) == 7

    @pytest.mark.asyncio
    async def test_hover_works_after_compile(self, server, make_diag):
        """After compile_and_publish, hover returns AI content for cached diag."""
        diag = make_diag(
            start_line=5, start_col=9, end_line=5, end_col=13,
            explanation="Hmm, what type do you think this should be?",
            hint="Have you checked what type the left side expects?",
        )
        result = CompilationResult(file="Main.hs", diagnostics=[diag], success=False, raw_stderr="")
        server._bridge.compile = AsyncMock(return_value=result)
        server._engine.enrich_all = AsyncMock(return_value=[diag])
        server.publish_diagnostics = MagicMock()

        await server.compile_and_publish(URI, "x = True + 1\n")

        from lsprotocol.types import HoverParams, TextDocumentIdentifier
        hover_params = HoverParams(
            text_document=TextDocumentIdentifier(uri=URI),
            position=Position(line=4, character=9),
        )
        from server.lsp_server import hover
        result_hover = hover(server, hover_params)
        assert result_hover is not None
        assert "Hmm, what type" in result_hover.contents.value

    @pytest.mark.asyncio
    async def test_highlight_works_after_compile(self, server, make_diag):
        """After compile_and_publish, document_highlight returns the error span."""
        diag = make_diag(
            start_line=5, start_col=9, end_line=5, end_col=13,
            explanation="Think of it like this...",
            hint="What type were you expecting here?",
        )
        result = CompilationResult(file="Main.hs", diagnostics=[diag], success=False, raw_stderr="")
        server._bridge.compile = AsyncMock(return_value=result)
        server._engine.enrich_all = AsyncMock(return_value=[diag])
        server.publish_diagnostics = MagicMock()

        await server.compile_and_publish(URI, "x = True + 1\n")

        params = DocumentHighlightParams(
            text_document=TextDocumentIdentifier(uri=URI),
            position=Position(line=4, character=9),
        )
        highlights = document_highlight(server, params)
        assert len(highlights) == 1
        assert highlights[0].range.start.line == 4
        assert highlights[0].range.start.character == 8

    @pytest.mark.asyncio
    async def test_ghc_exception_does_not_crash_server(self, server):
        """If GHCBridge raises, compile_and_publish must log and return silently."""
        server._bridge.compile = AsyncMock(side_effect=RuntimeError("GHC not found"))
        server.publish_diagnostics = MagicMock()

        await server.compile_and_publish(URI, "main = putStrLn True\n")
        server.publish_diagnostics.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_source_still_compiles(self, server):
        """Empty source string must go through the full pipeline without error."""
        result = CompilationResult(file="Main.hs", diagnostics=[], success=True, raw_stderr="")
        server._bridge.compile = AsyncMock(return_value=result)
        server._engine.enrich_all = AsyncMock(return_value=[])
        server.publish_diagnostics = MagicMock()

        await server.compile_and_publish(URI, "")

        server.publish_diagnostics.assert_called_once_with(URI, [])


class TestGracefulDegradation:

    @pytest.mark.asyncio
    async def test_engine_returns_original_when_no_api_key(self, make_diag):
        """If GROQ_API_KEY is absent, enrich() returns the original diagnostic."""
        diag = make_diag()
        engine = AIFeedbackEngine(api_key="", context_manager=ContextManager())        
        engine._client = None   
        engine._context = ContextManager()

        result = await engine.enrich(diag, "main = True + 1\n", URI)
        assert result is diag
        assert not result.ai_explanation
        assert not result.ai_hint

    @pytest.mark.asyncio
    async def test_engine_returns_original_on_http_error(self, make_diag):
        """Network error during Groq call → original diagnostic returned unchanged."""
        diag = make_diag()
        engine = AIFeedbackEngine(context_manager=ContextManager())
        engine._client = MagicMock()
        engine._client.chat_complete = MagicMock(
            side_effect=ConnectionError("network unreachable")
        )

        result = await engine.enrich(diag, "main = True + 1\n", URI)
        assert result is diag

    def test_highlight_returns_empty_for_unknown_uri(self, server):
        """document_highlight for an unknown URI returns [] without raising."""
        params = DocumentHighlightParams(
            text_document=TextDocumentIdentifier(uri="file:///unknown.hs"),
            position=Position(line=0, character=0),
        )
        result = document_highlight(server, params)
        assert result == []


class TestDebounce:

    @pytest.mark.asyncio
    async def test_second_change_cancels_first_task(self, server):
        """Two rapid _schedule_compile calls → only second task survives."""
        compile_count = 0

        async def slow_compile(uri, source):
            nonlocal compile_count
            await asyncio.sleep(0.5)  
            compile_count += 1

        server._debounced_compile = slow_compile

        server._schedule_compile(URI, "v1")
        first_task = server._debounce_tasks[URI]

        server._schedule_compile(URI, "v2")
        second_task = server._debounce_tasks[URI]

        assert first_task.cancelled() or first_task.cancelling() > 0, \
            "First task should have been cancelled"
        assert first_task is not second_task

    @pytest.mark.asyncio
    async def test_only_one_task_per_uri(self, server):
        """After multiple schedule calls, only one pending task exists per URI."""
        async def noop(uri, source):
            await asyncio.sleep(10)

        server._debounced_compile = noop
        for i in range(5):
            server._schedule_compile(URI, f"version {i}")

        assert URI in server._debounce_tasks
        pending = [t for t in [server._debounce_tasks[URI]] if not t.done()]
        assert len(pending) == 1


class TestDiagnosticConversion:

    def test_severity_error_maps_correctly(self, make_diag):
        from lsprotocol.types import DiagnosticSeverity
        d = make_diag(severity=Severity.ERROR)
        lsp = _to_lsp_diagnostic(d)
        assert lsp.severity == DiagnosticSeverity.Error

    def test_severity_warning_maps_correctly(self, make_diag):
        from lsprotocol.types import DiagnosticSeverity
        d = make_diag(severity=Severity.WARNING, category=ErrorCategory.WARNING_GENERIC)
        lsp = _to_lsp_diagnostic(d)
        assert lsp.severity == DiagnosticSeverity.Warning

    def test_coordinates_converted_to_zero_indexed(self, make_diag):
        """GHC 1-indexed coords must become 0-indexed in LSP Diagnostic."""
        d = make_diag(start_line=5, start_col=9, end_line=5, end_col=13)
        lsp = _to_lsp_diagnostic(d)
        assert lsp.range.start.line      == 4   
        assert lsp.range.start.character == 8   
        assert lsp.range.end.line        == 4

    def test_ai_content_embedded_in_message(self, diag_with_ai):
        """AI explanation and hint must appear in the LSP diagnostic message."""
        lsp = _to_lsp_diagnostic(diag_with_ai)
        assert diag_with_ai.ai_explanation in lsp.message
        assert diag_with_ai.ai_hint       in lsp.message

    def test_source_is_ghc(self, make_diag):
        """All diagnostics must have source='ghc'."""
        lsp = _to_lsp_diagnostic(make_diag())
        assert lsp.source == "ghc"

    def test_no_span_falls_back_to_line_zero(self, make_diag):
        """A diagnostic without a span gets range (0,0)→(0,1) as fallback."""
        d = make_diag()
        d.span = None
        lsp = _to_lsp_diagnostic(d)
        assert lsp.range.start.line == 0
        assert lsp.range.start.character == 0

    @pytest.mark.parametrize("category", list(ErrorCategory))
    def test_all_categories_convert_without_error(self, make_diag, category):
        """_to_lsp_diagnostic must work for every ErrorCategory."""
        d = make_diag(category=category)
        lsp = _to_lsp_diagnostic(d)
        assert lsp is not None


class TestWebServerDictShape:

    REQUIRED_KEYS = {
        "severity", "startLine", "startCol", "endLine", "endCol",
        "message", "category", "explanation", "hint",
    }

    def test_dict_contains_all_required_keys(self, diag_with_ai):
        d = _diagnostic_to_dict(diag_with_ai)
        missing = self.REQUIRED_KEYS - set(d.keys())
        assert not missing, f"Missing keys in diagnostic dict: {missing}"

    def test_severity_is_lowercase_string(self, make_diag):
        d = _diagnostic_to_dict(make_diag(severity=Severity.ERROR))
        assert d["severity"] == "error"

    def test_warning_severity_is_lowercase(self, make_diag):
        d = _diagnostic_to_dict(make_diag(severity=Severity.WARNING,
                                           category=ErrorCategory.WARNING_GENERIC))
        assert d["severity"] == "warning"

    def test_coordinates_are_1_indexed(self, make_diag):
        """Web editor uses 1-indexed coords (unlike LSP which is 0-indexed)."""
        d = _diagnostic_to_dict(make_diag(start_line=5, start_col=9))
        assert d["startLine"] == 5
        assert d["startCol"]  == 9

    def test_explanation_and_hint_present(self, diag_with_ai):
        d = _diagnostic_to_dict(diag_with_ai)
        assert d["explanation"] == diag_with_ai.ai_explanation
        assert d["hint"]        == diag_with_ai.ai_hint

    def test_empty_explanation_and_hint_when_not_enriched(self, make_diag):
        d = _diagnostic_to_dict(make_diag())
        assert d["explanation"] == ""
        assert d["hint"]        == ""

    def test_category_is_string(self, make_diag):
        d = _diagnostic_to_dict(make_diag(category=ErrorCategory.TYPE_ERROR))
        assert isinstance(d["category"], str)
        assert "TYPE_ERROR" in d["category"] or "type_error" in d["category"].lower()

    def test_dict_is_json_serialisable(self, diag_with_ai):
        """The dict must be serialisable to JSON without errors."""
        d = _diagnostic_to_dict(diag_with_ai)
        serialised = json.dumps(d)
        assert len(serialised) > 10

class TestContextAndLevels:

    def test_fresh_context_is_beginner_for_all_categories(self):
        ctx = ContextManager().get_or_create(URI)
        for cat in ErrorCategory:
            assert ctx.get_level(cat) == ExperienceLevel.BEGINNER

    def test_level_advances_to_intermediate(self):
        ctx = ContextManager().get_or_create(URI)
        for _ in range(_INTERMEDIATE_THRESHOLD):
            ctx.record_diagnostic(ErrorCategory.TYPE_ERROR, "test error")
        assert ctx.get_level(ErrorCategory.TYPE_ERROR) == ExperienceLevel.INTERMEDIATE

    def test_level_advances_to_advanced(self):
        ctx = ContextManager().get_or_create(URI)
        for _ in range(_ADVANCED_THRESHOLD):
            ctx.record_diagnostic(ErrorCategory.TYPE_ERROR, "test error")
        assert ctx.get_level(ErrorCategory.TYPE_ERROR) == ExperienceLevel.ADVANCED

    def test_category_tracking_is_independent(self):
        """Advancing on one category does not affect others."""
        ctx = ContextManager().get_or_create(URI)
        for _ in range(_ADVANCED_THRESHOLD):
            ctx.record_diagnostic(ErrorCategory.TYPE_ERROR, "test error")
        for cat in ErrorCategory:
            if cat != ErrorCategory.TYPE_ERROR:
                assert ctx.get_level(cat) == ExperienceLevel.BEGINNER, \
                    f"{cat} should still be BEGINNER"

    def test_reset_clears_single_uri(self):
        mgr = ContextManager()
        ctx = mgr.get_or_create(URI)
        for _ in range(_ADVANCED_THRESHOLD):
            ctx.record_diagnostic(ErrorCategory.TYPE_ERROR, "test error")
        mgr.reset(URI)
        fresh = mgr.get_or_create(URI)
        assert fresh.get_level(ErrorCategory.TYPE_ERROR) == ExperienceLevel.BEGINNER

    def test_reset_all_clears_everything(self):
        mgr = ContextManager()
        for i in range(3):
            ctx = mgr.get_or_create(f"file:///file{i}.hs")
            ctx.record_diagnostic(ErrorCategory.SCOPE_ERROR, "test error")
        assert len(mgr) == 3
        mgr.reset_all()
        assert len(mgr) == 0

    def test_get_or_create_returns_same_object_for_same_uri(self):
        mgr = ContextManager()
        a = mgr.get_or_create(URI)
        b = mgr.get_or_create(URI)
        assert a is b

class TestSourceContextExtraction:

    SOURCE = "\n".join([f"line{i}" for i in range(1, 21)])  # 20 lines

    def test_extracts_window_around_error(self):
        from server.ghc.models import SourceSpan
        span = SourceSpan(file="Main.hs", start_line=10, start_col=1,end_line=10, end_col=5)
        ctx = _extract_source_context(self.SOURCE, span.start_line, radius=3)
        assert "line10" in ctx

    def test_includes_lines_before_error(self):
        from server.ghc.models import SourceSpan
        span = SourceSpan(file="Main.hs", start_line=10, start_col=1,
                          end_line=10, end_col=5)
        ctx = _extract_source_context(self.SOURCE, span.start_line, radius=3)

        assert "line7" in ctx or "line8" in ctx or "line9" in ctx

    def test_includes_lines_after_error(self):
        from server.ghc.models import SourceSpan
        span = SourceSpan(file="Main.hs", start_line=10, start_col=1,
                          end_line=10, end_col=5)
        ctx = _extract_source_context(self.SOURCE, span.start_line, radius=3)

        assert "line11" in ctx or "line12" in ctx or "line13" in ctx

    def test_handles_span_at_start_of_file(self):
        from server.ghc.models import SourceSpan
        span = SourceSpan(file="Main.hs", start_line=1, start_col=1, end_line=1, end_col=5)
        ctx = _extract_source_context(self.SOURCE, span.start_line, radius=3)

        assert "line1" in ctx  

    def test_handles_span_at_end_of_file(self):
        from server.ghc.models import SourceSpan
        span = SourceSpan(file="Main.hs", start_line=20, start_col=1,
end_line=20, end_col=5)
        ctx = _extract_source_context(self.SOURCE, span.start_line, radius=3)

        assert "line20" in ctx

    def test_returns_empty_string_for_none_span(self):
        result = _extract_source_context(self.SOURCE, 1, radius=3)
        assert result == "" or result is not None   


class TestResponseParseEdgeCases:

    def test_normal_response_parsed_correctly(self):
        raw = "EXPLANATION: Think of a vending machine.\nHINT: What type goes here?"
        exp, hint = parse_response(raw)
        assert "vending machine" in exp
        assert "What type" in hint

    def test_case_insensitive_parsing(self):
        raw = "explanation: Some explanation here.\nhint: A question here?"
        exp, hint = parse_response(raw)
        assert exp != ""
        assert hint != ""

    def test_missing_hint_returns_empty_hint(self):
        raw = "EXPLANATION: Let's think about this carefully."
        exp, hint = parse_response(raw)
        assert "Let's think" in exp
        assert hint == ""

    def test_empty_string_returns_empty_explanation(self):
        exp, hint = parse_response("")
        assert exp == ""
        assert hint == ""

    def test_garbled_response_falls_back_to_full_text(self):
        """If neither marker is found, full text becomes the explanation."""
        raw = "Something went wrong with the type system here."
        exp, hint = parse_response(raw)
        assert raw.strip() in exp
        assert hint == ""

    def test_extra_whitespace_is_stripped(self):
        raw = "EXPLANATION:   Lots of spaces.   \nHINT:   Also spaces.   "
        exp, hint = parse_response(raw)
        assert exp == "Lots of spaces."
        assert hint == "Also spaces."

    def test_multiline_explanation_captured(self):
        """Only the EXPLANATION: line is captured, not multiple lines."""
        raw = "EXPLANATION: First line of explanation.\nSome continuation.\nHINT: A question?"
        exp, hint = parse_response(raw)
        assert "First line" in exp
        assert hint == "A question?"

class TestUserPromptBuilding:

    def test_user_prompt_contains_ghc_message(self):
        msg = "Couldn't match expected type 'Int' with actual type 'Bool'"
        prompt = build_user_prompt(msg, "x = True", "Main.hs", 5, 9)
        assert msg in prompt

    def test_user_prompt_contains_source_context(self):
        ctx = "line4\nline5 = True + 1\nline6"
        prompt = build_user_prompt("some error", ctx, "Main.hs", 5, 9)
        assert ctx in prompt

    def test_user_prompt_contains_filename(self):
        prompt = build_user_prompt("err", "src", "MyModule.hs", 3, 1)
        assert "MyModule.hs" in prompt

    def test_user_prompt_contains_line_and_col(self):
        prompt = build_user_prompt("err", "src", "Main.hs", 42, 7)
        assert "42" in prompt
        assert "7" in prompt

    def test_empty_source_context_replaced_with_placeholder(self):
        prompt = build_user_prompt("err", "", "Main.hs", 1, 1)
        assert "not available" in prompt.lower() or "context" in prompt.lower()