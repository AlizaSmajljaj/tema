"""
context.py — Tracks a student's error history across a session and computes
an adaptive experience level used by the AI engine to tune explanation depth.

Design
------
The context is keyed by file URI (one context object per open document).
For each ErrorCategory, we track how many times the student has encountered
that category. The experience level for a given category is then computed as:

    BEGINNER   → 0–1 encounters
    INTERMEDIATE → 2–4 encounters
    ADVANCED   → 5+ encounters

This means explanations start detailed and gradually become subtler as the
student demonstrates familiarity with a particular error type. The level is
per-category, not global — a student can be ADVANCED on type errors but
still BEGINNER on instance/typeclass errors.

The context is intentionally in-memory and session-scoped. It resets when
the language server restarts. Persistence across sessions is a future
improvement noted in the thesis limitations section.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

from server.ghc.models import ErrorCategory

logger = logging.getLogger(__name__)


class ExperienceLevel(Enum):
    """
    The student's apparent familiarity with a given error category.
    Used by the AI engine to select prompt depth and verbosity.
    """
    BEGINNER     = "beginner"      # First or second encounter → full explanation
    INTERMEDIATE = "intermediate"  # Some experience → explanation + focused hint
    ADVANCED     = "advanced"      # Repeated encounters → hint only, minimal hand-holding

    def describe(self) -> str:
        """Human-readable description for prompt injection."""
        return {
            ExperienceLevel.BEGINNER:     (
                "a complete beginner who has never seen this type of error before"
            ),
            ExperienceLevel.INTERMEDIATE: (
                "an intermediate student who has encountered this error a few times "
                "but still needs guidance"
            ),
            ExperienceLevel.ADVANCED:     (
                "a more experienced student who is familiar with this error category "
                "and benefits more from subtle hints than lengthy explanations"
            ),
        }[self]


_INTERMEDIATE_THRESHOLD = 2
_ADVANCED_THRESHOLD = 5


@dataclass
class CategoryStats:
    """Statistics for one error category within one file context."""
    encounters: int = 0
    last_message: str = ""         

    @property
    def level(self) -> ExperienceLevel:
        if self.encounters < _INTERMEDIATE_THRESHOLD:
            return ExperienceLevel.BEGINNER
        if self.encounters < _ADVANCED_THRESHOLD:
            return ExperienceLevel.INTERMEDIATE
        return ExperienceLevel.ADVANCED


@dataclass
class UserContext:
    """
    Tracks error history for a single open document (identified by URI).

    Attributes
    ----------
    uri:
        The document URI this context belongs to.
    category_stats:
        Maps ErrorCategory → CategoryStats. Populated as diagnostics arrive.
    total_diagnostics_seen:
        Running total of all diagnostics processed for this document.
    """
    uri: str
    category_stats: dict[ErrorCategory, CategoryStats] = field(
        default_factory=lambda: defaultdict(CategoryStats)
    )
    total_diagnostics_seen: int = 0

    def record_diagnostic(self, category: ErrorCategory, message: str) -> ExperienceLevel:
        """
        Record that the student has encountered a diagnostic of the given
        category, and return the experience level to use for the AI prompt.

        Parameters
        ----------
        category:
            The ErrorCategory of the incoming diagnostic.
        message:
            The raw GHC message (stored for debugging / future use).

        Returns
        -------
        ExperienceLevel
            The level computed *after* recording this encounter, so the first
            time a category is seen the student is still BEGINNER.
        """
        stats = self.category_stats[category]
        stats.encounters += 1
        stats.last_message = message
        self.total_diagnostics_seen += 1

        level = stats.level
        logger.debug(
            "Context update [%s] category=%s encounters=%d level=%s",
            self.uri, category.value, stats.encounters, level.name,
        )
        return level

    def get_level(self, category: ErrorCategory) -> ExperienceLevel:
        """Return the current experience level for a category without recording."""
        return self.category_stats[category].level

    def summary(self) -> dict:
        """Return a JSON-serialisable summary (useful for debugging endpoints)."""
        return {
            "uri": self.uri,
            "total_diagnostics_seen": self.total_diagnostics_seen,
            "categories": {
                cat.name: {
                    "encounters": stats.encounters,
                    "level": stats.level.name,
                }
                for cat, stats in self.category_stats.items()
            },
        }


class ContextManager:
    """
    Session-scoped store of UserContext objects, one per open document URI.

    The LSP server holds a single ContextManager instance and passes it to
    the AI engine on every diagnostic request.
    """

    def __init__(self) -> None:
        self._contexts: dict[str, UserContext] = {}

    def get_or_create(self, uri: str) -> UserContext:
        """Return the existing context for a URI, or create a fresh one."""
        if uri not in self._contexts:
            self._contexts[uri] = UserContext(uri=uri)
            logger.info("Created new UserContext for %s", uri)
        return self._contexts[uri]

    def reset(self, uri: str) -> None:
        """Clear the context for a URI (e.g. when the document is closed)."""
        if uri in self._contexts:
            del self._contexts[uri]
            logger.info("Reset UserContext for %s", uri)

    def reset_all(self) -> None:
        """Clear all contexts (e.g. on server restart)."""
        self._contexts.clear()

    def __len__(self) -> int:
        return len(self._contexts)