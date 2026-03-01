"""
server.ai — AI feedback engine for the Haskell language server.

Public exports
--------------
    AIFeedbackEngine   — main entry point: enriches GHCDiagnostics with
                         AI-generated explanations and hints
    ContextManager     — session-scoped store of per-document UserContext
    UserContext        — tracks error history for a single document
    ExperienceLevel    — BEGINNER / INTERMEDIATE / ADVANCED
"""

from server.ai.engine import AIFeedbackEngine
from server.ai.context import ContextManager, UserContext, ExperienceLevel
from server.ai.prompts import build_system_prompt, build_user_prompt, parse_response

__all__ = [
    "AIFeedbackEngine",
    "ContextManager",
    "UserContext",
    "ExperienceLevel",
    "build_system_prompt",
    "build_user_prompt",
    "parse_response",
]