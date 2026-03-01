"""
test_ghc_parser.py — Unit tests for the GHC output parser.

Tests are fully offline — they use static GHC stderr snapshots so no
GHC installation is required to run the test suite.
"""

import pytest
from server.ghc.parser import parse_ghc_output
from server.ghc.models import Severity, ErrorCategory


# ---------------------------------------------------------------------------
# Fixtures — realistic GHC stderr samples
# ---------------------------------------------------------------------------

TYPE_ERROR_OUTPUT = """\
/home/user/Test.hs:5:10: error:
    • Couldn't match expected type 'Int' with actual type '[Char]'
    • In the expression: "hello"
      In an equation for 'foo': foo = "hello"
"""

SCOPE_ERROR_OUTPUT = """\
/home/user/Test.hs:3:1: error:
    Variable not in scope: myFunc :: Int -> Int
"""

PATTERN_MATCH_WARNING = """\
/home/user/Test.hs:8:1: warning: [-Wincomplete-patterns]
    Pattern match(es) are non-exhaustive
    In an equation for 'describe':
        Patterns of type 'Bool' not matched: False
"""

PARSE_ERROR_OUTPUT = """\
/home/user/Test.hs:12:5: error:
    parse error (possibly incorrect indentation or mismatched brackets)
"""

MULTI_LINE_RANGE = """\
/home/user/Test.hs:(15,3)-(17,10): error:
    • Occurs check: cannot construct the infinite type: a ~ [a]
    • In the expression: x : x
"""

UNUSED_IMPORT_WARNING = """\
/home/user/Test.hs:1:1: warning: [-Wunused-imports]
    The import of 'Data.List' is redundant
      except perhaps to import instances from 'Data.List'
"""

MULTIPLE_DIAGNOSTICS = """\
/home/user/Test.hs:3:5: error:
    • Couldn't match type 'Int' with 'Bool'
    • In the expression: True + 1

/home/user/Test.hs:7:1: warning: [-Wunused-binds]
    Defined but not used: 'helper'
"""

# GHC >= 9.4 JSON format
JSON_DIAGNOSTIC_OUTPUT = """\
{"span":{"file":"/home/user/Test.hs","startLine":5,"startCol":10,"endLine":5,"endCol":17},"severity":"Error","message":"Couldn't match expected type 'Int' with actual type '[Char]'","code":83}
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLocationParsing:
    def test_point_location(self):
        result = parse_ghc_output(SCOPE_ERROR_OUTPUT, "/home/user/Test.hs", exit_code=1)
        diag = result.diagnostics[0]
        assert diag.span.start_line == 3
        assert diag.span.start_col == 1

    def test_col_range_location(self):
        result = parse_ghc_output(TYPE_ERROR_OUTPUT, "/home/user/Test.hs", exit_code=1)
        diag = result.diagnostics[0]
        assert diag.span.start_line == 5
        assert diag.span.start_col == 10

    def test_multiline_range_location(self):
        result = parse_ghc_output(MULTI_LINE_RANGE, "/home/user/Test.hs", exit_code=1)
        diag = result.diagnostics[0]
        assert diag.span.start_line == 15
        assert diag.span.start_col == 3
        assert diag.span.end_line == 17
        assert diag.span.end_col == 10

    def test_lsp_range_is_zero_indexed(self):
        result = parse_ghc_output(TYPE_ERROR_OUTPUT, "/home/user/Test.hs", exit_code=1)
        lsp_range = result.diagnostics[0].span.to_lsp_range()
        # GHC line 5, col 10 → LSP line 4, char 9
        assert lsp_range["start"]["line"] == 4
        assert lsp_range["start"]["character"] == 9


class TestSeverity:
    def test_error_severity(self):
        result = parse_ghc_output(TYPE_ERROR_OUTPUT, "/home/user/Test.hs", exit_code=1)
        assert result.diagnostics[0].severity == Severity.ERROR

    def test_warning_severity(self):
        result = parse_ghc_output(PATTERN_MATCH_WARNING, "/home/user/Test.hs", exit_code=0)
        assert result.diagnostics[0].severity == Severity.WARNING

    def test_success_flag_on_zero_exit(self):
        result = parse_ghc_output(PATTERN_MATCH_WARNING, "/home/user/Test.hs", exit_code=0)
        assert result.success is True

    def test_failure_flag_on_nonzero_exit(self):
        result = parse_ghc_output(TYPE_ERROR_OUTPUT, "/home/user/Test.hs", exit_code=1)
        assert result.success is False


class TestErrorCategory:
    def test_type_error_category(self):
        result = parse_ghc_output(TYPE_ERROR_OUTPUT, "/home/user/Test.hs", exit_code=1)
        assert result.diagnostics[0].category == ErrorCategory.TYPE_ERROR

    def test_scope_error_category(self):
        result = parse_ghc_output(SCOPE_ERROR_OUTPUT, "/home/user/Test.hs", exit_code=1)
        assert result.diagnostics[0].category == ErrorCategory.SCOPE_ERROR

    def test_pattern_match_category(self):
        result = parse_ghc_output(PATTERN_MATCH_WARNING, "/home/user/Test.hs", exit_code=0)
        assert result.diagnostics[0].category == ErrorCategory.PATTERN_MATCH

    def test_syntax_error_category(self):
        result = parse_ghc_output(PARSE_ERROR_OUTPUT, "/home/user/Test.hs", exit_code=1)
        assert result.diagnostics[0].category == ErrorCategory.SYNTAX_ERROR

    def test_recursion_occurs_check_category(self):
        result = parse_ghc_output(MULTI_LINE_RANGE, "/home/user/Test.hs", exit_code=1)
        assert result.diagnostics[0].category == ErrorCategory.RECURSION


class TestErrorCodes:
    def test_warning_code_extracted(self):
        result = parse_ghc_output(PATTERN_MATCH_WARNING, "/home/user/Test.hs", exit_code=0)
        assert result.diagnostics[0].error_code == "-Wincomplete-patterns"

    def test_unused_import_code(self):
        result = parse_ghc_output(UNUSED_IMPORT_WARNING, "/home/user/Test.hs", exit_code=0)
        assert result.diagnostics[0].error_code == "-Wunused-imports"


class TestMultipleDiagnostics:
    def test_count(self):
        result = parse_ghc_output(MULTIPLE_DIAGNOSTICS, "/home/user/Test.hs", exit_code=1)
        assert len(result.diagnostics) == 2

    def test_errors_and_warnings_split(self):
        result = parse_ghc_output(MULTIPLE_DIAGNOSTICS, "/home/user/Test.hs", exit_code=1)
        assert len(result.errors) == 1
        assert len(result.warnings) == 1

    def test_message_body_captured(self):
        result = parse_ghc_output(MULTIPLE_DIAGNOSTICS, "/home/user/Test.hs", exit_code=1)
        assert "Couldn't match type" in result.errors[0].message


class TestEmptyInput:
    def test_no_output_returns_empty(self):
        result = parse_ghc_output("", "/home/user/Test.hs", exit_code=0)
        assert result.diagnostics == []
        assert result.success is True


class TestJsonFormat:
    def test_json_diagnostic_parsed(self):
        result = parse_ghc_output(JSON_DIAGNOSTIC_OUTPUT, "/home/user/Test.hs", exit_code=1)
        assert len(result.diagnostics) == 1
        diag = result.diagnostics[0]
        assert diag.severity == Severity.ERROR
        assert diag.span.start_line == 5
        assert diag.span.start_col == 10
        assert "Couldn't match" in diag.message
        assert diag.error_code == "83"


class TestLspSerialization:
    def test_to_lsp_diagnostic_basic(self):
        result = parse_ghc_output(TYPE_ERROR_OUTPUT, "/home/user/Test.hs", exit_code=1)
        lsp_diag = result.diagnostics[0].to_lsp_diagnostic()
        assert lsp_diag["severity"] == Severity.ERROR.value
        assert lsp_diag["source"] == "ghc"
        assert "range" in lsp_diag
        assert "Couldn't match" in lsp_diag["message"]

    def test_to_lsp_diagnostic_with_ai_explanation(self):
        result = parse_ghc_output(TYPE_ERROR_OUTPUT, "/home/user/Test.hs", exit_code=1)
        diag = result.diagnostics[0]
        diag.ai_explanation = "You tried to use a String where an Int was expected."
        lsp_diag = diag.to_lsp_diagnostic()
        assert "You tried to use a String" in lsp_diag["message"]