"""
models.py — Structured data models for GHC compiler output.

These dataclasses are the shared language between the GHC bridge/parser
and the rest of the system (LSP server, AI engine).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Severity(Enum):
    """Maps to LSP DiagnosticSeverity."""
    ERROR = 1
    WARNING = 2
    INFO = 3
    HINT = 4


class ErrorCategory(Enum):
    """
    High-level category of the error — used by the AI engine to select
    appropriate prompt templates and explanation strategies.
    """
    TYPE_ERROR = "type_error"
    SYNTAX_ERROR = "syntax_error"
    SCOPE_ERROR = "scope_error"           # out-of-scope variable / unknown name
    PATTERN_MATCH = "pattern_match"       # non-exhaustive patterns, redundant match
    RECURSION = "recursion"               # infinite type, occurs check
    IMPORT_ERROR = "import_error"
    KIND_ERROR = "kind_error"
    INSTANCE_ERROR = "instance_error"     # missing / overlapping typeclass instances
    WARNING_GENERIC = "warning_generic"
    UNKNOWN = "unknown"


@dataclass
class SourceSpan:
    """
    A range in a source file.
    All values are 1-indexed (as GHC reports them).
    LSP uses 0-indexed — conversion happens in lsp_server.py.
    """
    file: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int

    def to_lsp_range(self) -> dict:
        """Convert to LSP Range object (0-indexed)."""
        return {
            "start": {"line": self.start_line - 1, "character": self.start_col - 1},
            "end":   {"line": self.end_line - 1,   "character": self.end_col - 1},
        }


@dataclass
class GHCDiagnostic:
    """
    A single diagnostic item produced by GHC.

    This is the primary data structure flowing through the system:
      GHC stderr → parser → GHCDiagnostic → AI engine + LSP server
    """
    severity: Severity
    span: SourceSpan
    message: str                          # raw GHC message (may be multiline)
    category: ErrorCategory = ErrorCategory.UNKNOWN
    error_code: Optional[str] = None      # e.g. "-Wunused-imports", "GHC-12345"
    context_lines: list[str] = field(default_factory=list)  # source lines near error
    related: list["GHCDiagnostic"] = field(default_factory=list)  # sub-notes
    raw_ghc_output: str = ""
    
    # Set by AI engine after processing
    ai_explanation: Optional[str] = None
    ai_hint: Optional[str] = None

    def __str__(self) -> str:
        loc = f"{self.span.file}:{self.span.start_line}:{self.span.start_col}"
        return f"[{self.severity.name}] {loc} — {self.message[:80]}"

    def to_lsp_diagnostic(self) -> dict:
        """
        Serialize to an LSP Diagnostic object.
        The AI explanation (if present) is appended to the message.
        """
        msg = self.message
        if self.ai_explanation:
            msg += f"\n\n💡 {self.ai_explanation}"
        if self.ai_hint:
            msg += f"\n\n🔍 Hint: {self.ai_hint}"

        diag = {
            "range": self.span.to_lsp_range(),
            "severity": self.severity.value,
            "source": "ghc",
            "message": msg,
        }
        if self.error_code:
            diag["code"] = self.error_code
        return diag


@dataclass
class CompilationResult:
    """
    The full result of compiling a Haskell file.
    Wraps a list of diagnostics plus metadata.
    """
    file: str
    diagnostics: list[GHCDiagnostic] = field(default_factory=list)
    success: bool = False                 # True only if GHC exit code == 0
    ghc_version: Optional[str] = None
    raw_stderr: str = ""                  # preserved for debugging

    @property
    def errors(self) -> list[GHCDiagnostic]:
        return [d for d in self.diagnostics if d.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[GHCDiagnostic]:
        return [d for d in self.diagnostics if d.severity == Severity.WARNING]

    def __repr__(self) -> str:
        return (
            f"CompilationResult(file={self.file!r}, "
            f"errors={len(self.errors)}, warnings={len(self.warnings)}, "
            f"success={self.success})"
        )