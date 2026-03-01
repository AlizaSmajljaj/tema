"""
server/ghc — GHC compiler bridge, output parser, and data models.

Public API:
    run_ghc(source, file_path) → CompilationResult   (async)
    get_ghc_version()          → Optional[str]        (async)
    parse_ghc_output(...)      → CompilationResult    (sync, for testing)

Data types:
    CompilationResult, GHCDiagnostic, SourceSpan, Severity, ErrorCategory
"""

from .bridge import run_ghc, get_ghc_version, clear_cache
from .parser import parse_ghc_output
from .models import (
    CompilationResult,
    GHCDiagnostic,
    SourceSpan,
    Severity,
    ErrorCategory,
)

__all__ = [
    "run_ghc",
    "get_ghc_version",
    "clear_cache",
    "parse_ghc_output",
    "CompilationResult",
    "GHCDiagnostic",
    "SourceSpan",
    "Severity",
    "ErrorCategory",
]