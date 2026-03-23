"""
parser.py — Parses raw GHC stderr output into structured GHCDiagnostic objects.

GHC's error format has evolved over versions. We handle two formats:

FORMAT A — Classic (GHC < 9.4):
    /path/to/File.hs:10:5: error:
        • Could not deduce (Num String)
          from the context: ...
        • In the expression: x + "hello"

FORMAT B — JSON-style with -fdiagnostics-as-json (GHC >= 9.4, optional):
    {"span": {...}, "severity": "Error", "message": "..."}

We support both, preferring JSON when available.

GHC error categories are detected via regex matching on the message text —
this powers the AI engine's prompt selection.
"""

import re
import json
import logging
from typing import Optional

from .models import (
    GHCDiagnostic,
    CompilationResult,
    ErrorCategory,
    Severity,
    SourceSpan,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regex patterns for classic GHC output
# ---------------------------------------------------------------------------

# Matches the header line of a diagnostic, e.g.:
#   /abs/path/File.hs:12:4: error:
#   /abs/path/File.hs:12:4-9: warning: [-Wunused-binds]
#   /abs/path/File.hs:(12,4)-(13,8): error:

#_HEADER_RE = re.compile(
#    r"^(?P<file>.+\.hs)"
 #   r":(?P<loc>"
  #      r"(?:\(\d+,\d+\)-\(\d+,\d+\))"   # (line,col)-(line,col) range
   #     r"|(?:\d+:\d+-\d+)"               # line:col-col range
   #     r"|(?:\d+:\d+)"                   # line:col point
   # r")"
    #r":\s*(?P<severity>error|warning|note)\s*:?"
    #r"(?:\s*\[(?P<code>[^\]]+)\])?"        # optional [-Wflag] or [GHC-NNNN]
    #r"\s*$",
    #re.IGNORECASE,
#)
_HEADER_RE = re.compile(
    r"^(?P<file>.*?):(?P<loc>(\d+:\d+(-\d+)?|\(\d+,\d+\)-\(\d+,\d+\))):"
    r"\s*(?P<severity>error|warning|note):?"
    r"(\s+\[(?P<code>.*?)\])?",
    re.IGNORECASE
)
# Version line: "ghc: version 9.4.7"
_VERSION_RE = re.compile(r"The Glorious Glasgow Haskell Compilation System,\s+version\s+([\d.]+)")

# Matches GHC >= 9.4 JSON diagnostics flag indicator in stderr
_JSON_MARKER = re.compile(r'^\{"version":|^\{"span":')


# ---------------------------------------------------------------------------
# Category detection — regex matched against the error message body
# ---------------------------------------------------------------------------

_CATEGORY_PATTERNS: list[tuple[re.Pattern, ErrorCategory]] = [
    # Type errors
    (re.compile(r"Could not deduce|No instance for|Couldn't match"
                r"|type mismatch|expected type|actual type"
                r"|rigid type variable", re.I), ErrorCategory.TYPE_ERROR),
    # Kind errors
    (re.compile(r"Expected kind|kind mismatch|applied to too many", re.I),
     ErrorCategory.KIND_ERROR),
    # Scope / name errors
    (re.compile(r"Not in scope|Variable not in scope|"
                r"Perhaps you meant|not defined", re.I), ErrorCategory.SCOPE_ERROR),
    # Pattern match
    (re.compile(r"Non-exhaustive patterns|"
                r"Pattern match|overlap|redundant", re.I), ErrorCategory.PATTERN_MATCH),
    # Recursion / infinite type (occurs check)
    (re.compile(r"Occurs check|infinite type|"
                r"Cannot construct the infinite", re.I), ErrorCategory.RECURSION),
    # Instance / typeclass
    (re.compile(r"No instance|overlapping instances|"
                r"Could not find an instance", re.I), ErrorCategory.INSTANCE_ERROR),
    # Import
    (re.compile(r"Could not load module|not exported|"
                r"ambiguous module name", re.I), ErrorCategory.IMPORT_ERROR),
    # Syntax
    (re.compile(r"parse error|lexical error|"
                r"unexpected token|syntax error", re.I), ErrorCategory.SYNTAX_ERROR),
]


def _detect_category(message: str) -> ErrorCategory:
    """Return the best-matching ErrorCategory for a GHC message."""
    for pattern, category in _CATEGORY_PATTERNS:
        if pattern.search(message):
            return category
    return ErrorCategory.UNKNOWN


# ---------------------------------------------------------------------------
# Location parsing helpers
# ---------------------------------------------------------------------------

def _parse_location(file: str, loc_str: str) -> SourceSpan:
    """
    Parse GHC location string into a SourceSpan.

    Handles:
      "12:4"           → point (line 12, col 4)
      "12:4-9"         → same line range (line 12, col 4–9)
      "(12,4)-(13,8)"  → multi-line range
    """
    # (line,col)-(line,col)
    m = re.match(r"\((\d+),(\d+)\)-\((\d+),(\d+)\)", loc_str)
    if m:
        return SourceSpan(file, int(m[1]), int(m[2]), int(m[3]), int(m[4]))

    # line:col-col
    m = re.match(r"(\d+):(\d+)-(\d+)", loc_str)
    if m:
        return SourceSpan(file, int(m[1]), int(m[2]), int(m[1]), int(m[3]))

    # line:col (point)
    m = re.match(r"(\d+):(\d+)", loc_str)
    if m:
        return SourceSpan(file, int(m[1]), int(m[2]), int(m[1]), int(m[2]))

    # Fallback — file-level span
    logger.warning("Could not parse GHC location: %r", loc_str)
    return SourceSpan(file, 1, 1, 1, 1)


def _parse_severity(text: str) -> Severity:
    t = text.lower()
    if t == "error":
        return Severity.ERROR
    if t == "warning":
        return Severity.WARNING
    return Severity.INFO


# ---------------------------------------------------------------------------
# Classic text parser (state machine)
# ---------------------------------------------------------------------------

class _ClassicParser:
    """
    Parses GHC classic stderr using a line-by-line state machine.

    States:
      IDLE       — waiting for a diagnostic header line
      IN_DIAG    — accumulating body lines for the current diagnostic
    """

    def __init__(self, target_file: str):
        self.target_file = target_file
        self.diagnostics: list[GHCDiagnostic] = []
        self._current_header: Optional[re.Match] = None
        self._body_lines: list[str] = []
        self.ghc_version: Optional[str] = None

    def feed(self, line: str) -> None:
        # Check for version string
        vm = _VERSION_RE.search(line)
        if vm:
            self.ghc_version = vm.group(1)
            return

        m = _HEADER_RE.match(line)
        if m:
            self._flush()
            self._current_header = m
            self._body_lines = []
        elif self._current_header is not None:
            # Accumulate indented body lines; skip blank-ish separator lines
            if line.strip():
                self._body_lines.append(line.rstrip())

    def finish(self) -> None:
        self._flush()

    def _flush(self) -> None:
        """Emit a diagnostic from the current accumulated state."""
        if self._current_header is None:
            return

        m = self._current_header
        file_path = m.group("file")
        loc_str   = m.group("loc")
        sev_str   = m.group("severity")
        code      = m.group("code")

        # Only emit diagnostics for our target file (GHC sometimes prints
        # errors from imported modules; we surface those as notes).
        span = _parse_location(file_path, loc_str)
        severity = _parse_severity(sev_str)
        message = "\n".join(self._body_lines).strip()

        if not message:
            message = "(no message)"

        category = _detect_category(message)

        diag = GHCDiagnostic(
            severity=severity,
            span=span,
            message=message,
            category=category,
            error_code=code,
        )
        self.diagnostics.append(diag)
        self._current_header = None
        self._body_lines = []


# ---------------------------------------------------------------------------
# JSON parser (GHC >= 9.4 with -fdiagnostics-as-json)
# ---------------------------------------------------------------------------

def _parse_json_diagnostics(stderr: str, target_file: str) -> tuple[list[GHCDiagnostic], Optional[str]]:
    """
    Parse JSON-format GHC diagnostics. Returns (diagnostics, ghc_version).
    Each line of stderr may be a JSON object.
    """
    diagnostics: list[GHCDiagnostic] = []
    ghc_version: Optional[str] = None

    for line in stderr.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            vm = _VERSION_RE.search(line)
            if vm:
                ghc_version = vm.group(1)
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        span_obj = obj.get("span") or {}
        file_path = span_obj.get("file", target_file)
        sl = span_obj.get("startLine", 1)
        sc = span_obj.get("startCol", 1)
        el = span_obj.get("endLine", sl)
        ec = span_obj.get("endCol", sc)

        severity_str = obj.get("severity", "Error")
        message = obj.get("message", "")
        code = obj.get("code")

        span = SourceSpan(file_path, sl, sc, el, ec)
        severity = _parse_severity(severity_str)
        category = _detect_category(message)

        diag = GHCDiagnostic(
            severity=severity,
            span=span,
            message=message,
            category=category,
            error_code=str(code) if code else None,
        )
        diagnostics.append(diag)

    return diagnostics, ghc_version


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_ghc_output(
    stderr: str,
    target_file: str,
    exit_code: int,
) -> CompilationResult:
    """
    Parse GHC's stderr into a CompilationResult.

    Automatically detects whether to use JSON or classic parsing.

    Args:
        stderr:       Raw stderr string from GHC.
        target_file:  Absolute path to the compiled .hs file.
        exit_code:    GHC process exit code (0 = success).

    Returns:
        CompilationResult with populated diagnostics list.
    """
    result = CompilationResult(
        file=target_file,
        success=(exit_code == 0),
        raw_stderr=stderr,
    )

    if not stderr.strip():
        return result

    # Detect format: does any line look like a JSON diagnostic?
    use_json = any(_JSON_MARKER.match(l.strip()) for l in stderr.splitlines())

    if use_json:
        logger.debug("Using JSON diagnostic parser")
        diagnostics, version = _parse_json_diagnostics(stderr, target_file)
    else:
        logger.debug("Using classic text diagnostic parser")
        parser = _ClassicParser(target_file)
        for line in stderr.splitlines():
            parser.feed(line)
        parser.finish()
        diagnostics = parser.diagnostics
        version = parser.ghc_version

    result.diagnostics = diagnostics
    result.ghc_version = version

    logger.info(
        "Parsed %d diagnostics (%d errors, %d warnings) from GHC output",
        len(result.diagnostics),
        len(result.errors),
        len(result.warnings),
    )
    return result