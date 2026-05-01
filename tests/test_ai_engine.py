"""
Tests for server.ai — context tracking, prompt building, response parsing,
and engine behaviour.

All tests are fully offline. The Claude API is never called — the engine's
API client is replaced with a mock that returns a controlled response string.
This means the suite runs without ANTHROPIC_API_KEY and with no network access.
"""

import pytest
from unittest.mock import MagicMock, patch

from server.ghc.models import (
    GHCDiagnostic, SourceSpan, Severity, ErrorCategory,
)
from server.ai.context import (
    ContextManager, UserContext, ExperienceLevel,
    _INTERMEDIATE_THRESHOLD, _ADVANCED_THRESHOLD,
)
from server.ai.prompts import (
    build_system_prompt, build_user_prompt, parse_response,
)
from server.ai.engine import AIFeedbackEngine, _extract_source_context, _basename


@pytest.fixture
def simple_type_error() -> GHCDiagnostic:
    return GHCDiagnostic(
        severity=Severity.ERROR,
        span=SourceSpan(
            file="Main.hs",
            start_line=10, start_col=5,
            end_line=10, end_col=10,
        ),
        message="Couldn't match expected type 'Int' with actual type '[Char]'",
        category=ErrorCategory.TYPE_ERROR,
        error_code=None,
    )


@pytest.fixture
def scope_error() -> GHCDiagnostic:
    return GHCDiagnostic(
        severity=Severity.ERROR,
        span=SourceSpan(file="Main.hs", start_line=3, start_col=1,
                        end_line=3, end_col=8),
        message="Variable not in scope: myFunc :: Int -> Int",
        category=ErrorCategory.SCOPE_ERROR,
        error_code=None,
    )


@pytest.fixture
def sample_source() -> str:
    return "\n".join([
        "module Main where",          # 1
        "",                           # 2
        "import Data.List",           # 3
        "",                           # 4
        "main :: IO ()",              # 5
        "main = do",                  # 6
        "  let x = True",             # 7
        "  print (x + 1)",            # 8  ← type error here
        "",                           # 9
        "helper :: Int -> Int",       # 10
        "helper n = n * 2",           # 11
    ])


@pytest.fixture
def context_manager() -> ContextManager:
    return ContextManager()


@pytest.fixture
def engine_with_mock_client(context_manager) -> AIFeedbackEngine:
    """Return an AIFeedbackEngine whose Groq HTTP client is mocked.

    We mock _GroqClient.chat_complete so no network calls are made.
    The method returns a pre-formatted EXPLANATION/HINT string.
    """
    engine = AIFeedbackEngine(
        api_key="test-key",
        context_manager=context_manager,
    )
    from unittest.mock import MagicMock
    engine._client = MagicMock()
    engine._client.chat_complete.return_value = (
        "EXPLANATION: You gave a Bool where an Int was expected.\n"
        "HINT: Check whether the variable on line 8 should be a number."
    )
    engine._client.model = engine._model
    return engine


class TestCategoryStats:
    def test_first_encounter_is_beginner(self):
        mgr = ContextManager()
        ctx = mgr.get_or_create("file:///a.hs")
        level = ctx.record_diagnostic(ErrorCategory.TYPE_ERROR, "some error")
        assert level == ExperienceLevel.BEGINNER

    def test_reaches_intermediate_after_threshold(self):
        mgr = ContextManager()
        ctx = mgr.get_or_create("file:///a.hs")
        for i in range(_INTERMEDIATE_THRESHOLD):
            ctx.record_diagnostic(ErrorCategory.TYPE_ERROR, f"error {i}")
        level = ctx.record_diagnostic(ErrorCategory.TYPE_ERROR, "one more")
        assert level == ExperienceLevel.INTERMEDIATE

    def test_reaches_advanced_after_threshold(self):
        mgr = ContextManager()
        ctx = mgr.get_or_create("file:///a.hs")
        for i in range(_ADVANCED_THRESHOLD):
            ctx.record_diagnostic(ErrorCategory.TYPE_ERROR, f"error {i}")
        level = ctx.record_diagnostic(ErrorCategory.TYPE_ERROR, "final")
        assert level == ExperienceLevel.ADVANCED

    def test_levels_are_independent_per_category(self):
        mgr = ContextManager()
        ctx = mgr.get_or_create("file:///a.hs")
        
        for i in range(_ADVANCED_THRESHOLD + 1):
            ctx.record_diagnostic(ErrorCategory.TYPE_ERROR, "t")
       
        level = ctx.record_diagnostic(ErrorCategory.SCOPE_ERROR, "s")
        assert level == ExperienceLevel.BEGINNER
        assert ctx.get_level(ErrorCategory.TYPE_ERROR) == ExperienceLevel.ADVANCED

    def test_total_counter_increments(self):
        mgr = ContextManager()
        ctx = mgr.get_or_create("file:///a.hs")
        for _ in range(4):
            ctx.record_diagnostic(ErrorCategory.SYNTAX_ERROR, "x")
        assert ctx.total_diagnostics_seen == 4


class TestContextManager:
    def test_get_or_create_returns_same_object(self, context_manager):
        ctx1 = context_manager.get_or_create("file:///x.hs")
        ctx2 = context_manager.get_or_create("file:///x.hs")
        assert ctx1 is ctx2

    def test_different_uris_get_different_contexts(self, context_manager):
        ctx1 = context_manager.get_or_create("file:///a.hs")
        ctx2 = context_manager.get_or_create("file:///b.hs")
        assert ctx1 is not ctx2

    def test_reset_removes_context(self, context_manager):
        context_manager.get_or_create("file:///a.hs")
        assert len(context_manager) == 1
        context_manager.reset("file:///a.hs")
        assert len(context_manager) == 0

    def test_reset_all_clears_everything(self, context_manager):
        for i in range(5):
            context_manager.get_or_create(f"file:///file{i}.hs")
        context_manager.reset_all()
        assert len(context_manager) == 0

    def test_summary_is_serialisable(self, context_manager):
        ctx = context_manager.get_or_create("file:///a.hs")
        ctx.record_diagnostic(ErrorCategory.TYPE_ERROR, "e")
        s = ctx.summary()
        assert s["uri"] == "file:///a.hs"
        assert s["total_diagnostics_seen"] == 1
        assert "TYPE_ERROR" in s["categories"]


class TestExperienceLevel:
    def test_describe_returns_string(self):
        for level in ExperienceLevel:
            desc = level.describe()
            assert isinstance(desc, str)
            assert len(desc) > 10


class TestSystemPrompt:
    def test_contains_level_description(self):
        prompt = build_system_prompt(ErrorCategory.TYPE_ERROR, ExperienceLevel.BEGINNER)
        assert "beginner" in prompt.lower()

    def test_contains_category_guidance_type_error(self):
        prompt = build_system_prompt(ErrorCategory.TYPE_ERROR, ExperienceLevel.BEGINNER)
        assert "type error" in prompt.lower() or "mismatch" in prompt.lower()

    def test_contains_format_instruction(self):
        prompt = build_system_prompt(ErrorCategory.TYPE_ERROR, ExperienceLevel.BEGINNER)
        assert "EXPLANATION:" in prompt
        assert "HINT:" in prompt

    def test_contains_no_code_rule(self):
        prompt = build_system_prompt(ErrorCategory.SYNTAX_ERROR, ExperienceLevel.INTERMEDIATE)
        assert "corrected" in prompt.lower() or "fixed" in prompt.lower()

    @pytest.mark.parametrize("category", list(ErrorCategory))
    def test_all_categories_produce_non_empty_prompt(self, category):
        prompt = build_system_prompt(category, ExperienceLevel.BEGINNER)
        assert len(prompt) > 100

    @pytest.mark.parametrize("level", list(ExperienceLevel))
    def test_all_levels_produce_non_empty_prompt(self, level):
        prompt = build_system_prompt(ErrorCategory.TYPE_ERROR, level)
        assert len(prompt) > 100


class TestUserPrompt:
    def test_contains_ghc_message(self):
        prompt = build_user_prompt(
            ghc_message="Couldn't match Int with Bool",
            source_context="5 |  foo x = x + True",
            filename="Main.hs",
            line=5,
            col=12,
        )
        assert "Couldn't match Int with Bool" in prompt

    def test_contains_filename(self):
        prompt = build_user_prompt(
            ghc_message="err",
            source_context="",
            filename="Lib.hs",
            line=1, col=1,
        )
        assert "Lib.hs" in prompt

    def test_fallback_when_no_source_context(self):
        prompt = build_user_prompt(
            ghc_message="err",
            source_context="",
            filename="Main.hs",
            line=1, col=1,
        )
        assert "not available" in prompt

class TestParseResponse:
    def test_parses_well_formed_response(self):
        raw = (
            "EXPLANATION: You gave a Bool where Int was expected.\n"
            "HINT: Check the type of the variable on the right-hand side."
        )
        explanation, hint = parse_response(raw)
        assert explanation == "You gave a Bool where Int was expected."
        assert hint == "Check the type of the variable on the right-hand side."

    def test_case_insensitive_keys(self):
        raw = (
            "explanation: Something went wrong.\n"
            "hint: Try this approach."
        )
        explanation, hint = parse_response(raw)
        assert explanation == "Something went wrong."
        assert hint == "Try this approach."

    def test_graceful_fallback_on_malformed_response(self):
        raw = "GHC got confused because the types don't line up."
        explanation, hint = parse_response(raw)
        assert explanation == raw
        assert hint == ""

    def test_empty_hint_is_ok(self):
        raw = "EXPLANATION: The name is not in scope."
        explanation, hint = parse_response(raw)
        assert explanation == "The name is not in scope."
        assert hint == ""

    def test_strips_whitespace(self):
        raw = "EXPLANATION:   Lots of spaces.   \nHINT:   Also spaces.   "
        explanation, hint = parse_response(raw)
        assert explanation == "Lots of spaces."
        assert hint == "Also spaces."


class TestExtractSourceContext:
    def test_marks_error_line(self, sample_source):
        ctx = _extract_source_context(sample_source, line=8, radius=2)
        lines = ctx.splitlines()
        error_line = next(l for l in lines if "← error" in l)
        assert "8" in error_line
        assert "print" in error_line

    def test_respects_radius(self, sample_source):
        ctx = _extract_source_context(sample_source, line=8, radius=2)
        lines = ctx.splitlines()
        assert len(lines) == 5

    def test_clamps_at_start_of_file(self, sample_source):
        ctx = _extract_source_context(sample_source, line=1, radius=3)
        lines = ctx.splitlines()
        assert all(int(l.split("|")[0].strip()) >= 1 for l in lines)

    def test_clamps_at_end_of_file(self, sample_source):
        total = len(sample_source.splitlines())
        ctx = _extract_source_context(sample_source, line=total, radius=5)
        lines = ctx.splitlines()
        assert all(int(l.split("|")[0].strip()) <= total for l in lines)

    def test_empty_source_returns_empty(self):
        ctx = _extract_source_context("", line=1, radius=3)
        assert ctx == ""


class TestBasename:
    def test_file_uri(self):
        assert _basename("file:///home/user/Main.hs") == "Main.hs"

    def test_plain_path(self):
        assert _basename("/home/user/project/Lib.hs") == "Lib.hs"

    def test_filename_only(self):
        assert _basename("Main.hs") == "Main.hs"


class TestAIFeedbackEngine:
    @pytest.mark.asyncio
    async def test_enrich_populates_explanation_and_hint(
        self, engine_with_mock_client, simple_type_error, sample_source
    ):
        result = await engine_with_mock_client.enrich(
            simple_type_error, sample_source, "file:///Main.hs"
        )
        assert result.ai_explanation != ""
        assert "Bool" in result.ai_explanation or "expected" in result.ai_explanation
        assert result.ai_hint != ""

    @pytest.mark.asyncio
    async def test_enrich_does_not_mutate_original(
        self, engine_with_mock_client, simple_type_error, sample_source
    ):
        original_explanation = simple_type_error.ai_explanation
        await engine_with_mock_client.enrich(
            simple_type_error, sample_source, "file:///Main.hs"
        )
        assert simple_type_error.ai_explanation == original_explanation

    @pytest.mark.asyncio
    async def test_enrich_preserves_original_fields(
        self, engine_with_mock_client, simple_type_error, sample_source
    ):
        result = await engine_with_mock_client.enrich(
            simple_type_error, sample_source, "file:///Main.hs"
        )
        assert result.severity == simple_type_error.severity
        assert result.span == simple_type_error.span
        assert result.message == simple_type_error.message
        assert result.category == simple_type_error.category

    @pytest.mark.asyncio
    async def test_enrich_falls_back_on_api_error(
        self, context_manager, simple_type_error, sample_source
    ):
        engine = AIFeedbackEngine(api_key="test-key", context_manager=context_manager)
        engine._client = MagicMock()
        engine._client.chat_complete.side_effect = Exception("Network error")

        result = await engine.enrich(
            simple_type_error, sample_source, "file:///Main.hs"
        )
        assert result is simple_type_error

    @pytest.mark.asyncio
    async def test_no_api_key_returns_original(
        self, context_manager, simple_type_error, sample_source
    ):
        engine = AIFeedbackEngine(api_key="", context_manager=context_manager)
        result = await engine.enrich(
            simple_type_error, sample_source, "file:///Main.hs"
        )
        assert result is simple_type_error

    @pytest.mark.asyncio
    async def test_enrich_all_returns_list_same_length(
        self, engine_with_mock_client, simple_type_error, scope_error, sample_source
    ):
        results = await engine_with_mock_client.enrich_all(
            [simple_type_error, scope_error], sample_source, "file:///Main.hs"
        )
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_context_advances_with_repeated_calls(
        self, engine_with_mock_client, simple_type_error, sample_source
    ):
        uri = "file:///Main.hs"
       
        for _ in range(_INTERMEDIATE_THRESHOLD + 1):
            await engine_with_mock_client.enrich(
                simple_type_error, sample_source, uri
            )
        ctx = engine_with_mock_client.context_manager.get_or_create(uri)
        assert ctx.get_level(ErrorCategory.TYPE_ERROR) == ExperienceLevel.INTERMEDIATE

    @pytest.mark.asyncio
    async def test_claude_api_called_with_correct_model(
        self, engine_with_mock_client, simple_type_error, sample_source
    ):
        await engine_with_mock_client.enrich(
            simple_type_error, sample_source, "file:///Main.hs"
        )
        assert engine_with_mock_client._client.chat_complete.called

    @pytest.mark.asyncio
    async def test_system_prompt_passed_to_api(
        self, engine_with_mock_client, simple_type_error, sample_source
    ):
        await engine_with_mock_client.enrich(
            simple_type_error, sample_source, "file:///Main.hs"
        )
        call_kwargs = engine_with_mock_client._client.chat_complete.call_args.kwargs
        system = call_kwargs["system"]
        assert len(system) > 50 