"""
prompts.py — Prompt templates for the AI feedback engine.

Architecture
------------
Each GHC error category has a dedicated system prompt that establishes the
pedagogical context for that error type. A shared user-turn template then
injects the concrete diagnostic details (error message, source snippet,
experience level) at call time.

Design principles
-----------------
1. Category-specific system prompts: a type error requires different framing
   than a scope error or a pattern match warning. Generic prompts produce
   generic responses; specific prompts produce specific, actionable ones.

2. Experience-level adaptation: the system prompt instructs Claude to adjust
   its explanation depth based on the student's level (BEGINNER /
   INTERMEDIATE / ADVANCED). The level is injected dynamically from the
   UserContext at call time.

3. Constrained output format: every prompt requests a strict two-part
   response — EXPLANATION and HINT — separated by a sentinel line. This
   makes the response reliably parseable without JSON mode.

4. No code fixes: prompts explicitly instruct Claude not to write corrected
   code. The goal is understanding, not copy-paste fixes. This is a core
   pedagogical choice documented in the thesis.

5. Brevity: the system prompts enforce a word budget. Inline editor hovers
   have limited screen real estate; long explanations are counterproductive.
"""

from server.ghc.models import ErrorCategory
from server.ai.context import ExperienceLevel


# ── Shared constants ───────────────────────────────────────────────────────

_FORMAT_INSTRUCTION = """
Your response MUST follow this exact format — no other text, no markdown headers:

EXPLANATION: <one or two sentences explaining what the error means conceptually>
HINT: <one sentence suggesting how to think about fixing it, without writing code>
""".strip()

_NO_CODE_RULE = (
    "Do not write corrected Haskell code. "
    "Do not show the fixed version. "
    "Focus entirely on building the student's understanding."
)


# ── System prompt templates ────────────────────────────────────────────────

_BASE_SYSTEM = """
You are a patient and encouraging Haskell tutor embedded in a code editor.
Your job is to help students understand GHC compiler errors by explaining
them in plain, accessible language.

The student you are helping is {level_description}.

{category_guidance}

Rules:
- Keep EXPLANATION to 1–2 sentences maximum.
- Keep HINT to 1 sentence maximum.
- Use simple language; avoid type-theory jargon unless the student is advanced.
- Be encouraging, never condescending.
- {no_code_rule}

{format_instruction}
""".strip()


_CATEGORY_GUIDANCE: dict[ErrorCategory, str] = {

    ErrorCategory.TYPE_ERROR: """
You are explaining a Haskell type error. Type errors occur when GHC infers
that a value of one type is used in a context that expects a different type.
For beginners, focus on the mismatch itself: "you gave X but the function
expected Y". For advanced students, briefly note which constraint is unsatisfied
and where the inference chain breaks.
""".strip(),

    ErrorCategory.SCOPE_ERROR: """
You are explaining a scope error — a name that GHC cannot find. This usually
means a variable, function, or constructor is used before it is defined,
defined in a different module that hasn't been imported, or simply misspelled.
For beginners, explain the concept of scope. For advanced students, focus on
the likely specific cause (import, typo, or binding order).
""".strip(),

    ErrorCategory.SYNTAX_ERROR: """
You are explaining a Haskell parse error. GHC could not parse the source
file — the code violates Haskell's grammar or indentation rules. Haskell uses
significant indentation (like Python) and this is the most common source of
parse errors for new students. Focus on the likely structural issue without
reproducing the code.
""".strip(),

    ErrorCategory.PATTERN_MATCH: """
You are explaining a pattern match issue. This is either a non-exhaustive
pattern match warning (not all cases of a type are covered) or a redundant
pattern (a case can never be reached). For beginners, explain what pattern
matching is and why all cases must be handled. For advanced students, focus
on which constructor or case is missing or unreachable.
""".strip(),

    ErrorCategory.RECURSION: """
You are explaining a GHC occurs-check failure — an infinite type error.
This happens when GHC tries to unify a type variable with a type that
contains that same variable, producing an infinite type like `a ~ [a]`.
This is often caused by applying a function to itself, or by a recursive
data structure without a newtype wrapper. For beginners, use an analogy
(e.g., a box that contains itself). For advanced students, identify the
unification cycle directly.
""".strip(),

    ErrorCategory.INSTANCE_ERROR: """
You are explaining a missing or overlapping typeclass instance error.
GHC could not find an instance of a typeclass (like Show, Eq, or Num) for
a particular type. For beginners, explain what a typeclass is (a set of
operations a type must support) and why deriving or manual instances are
needed. For advanced students, focus on which constraint is unsatisfied and
in which context.
""".strip(),

    ErrorCategory.KIND_ERROR: """
You are explaining a kind error. Kinds are the "types of types" in Haskell's
type system. A kind error means a type constructor is applied to the wrong
number or kind of arguments. For beginners, use a simple analogy: just as
functions have types, type constructors have kinds, and mismatching them is
an error. For advanced students, state the expected and actual kinds directly.
""".strip(),

    ErrorCategory.IMPORT_ERROR: """
You are explaining a module import error. GHC cannot find, load, or resolve
the requested module. This could be a misspelled module name, a missing
package dependency, or a module that is not exposed by its package.
For beginners, explain how Haskell's module system works at a high level.
For advanced students, focus on the most likely cause given the error text.
""".strip(),

    ErrorCategory.WARNING_GENERIC: """
You are explaining a GHC warning. Warnings do not prevent compilation but
indicate potentially problematic code — unused bindings, incomplete patterns,
redundant imports, and so on. For beginners, explain why the warning exists
and what it protects against. For advanced students, briefly state the fix.
""".strip(),

    ErrorCategory.UNKNOWN: """
You are explaining a GHC diagnostic. You may not recognise the exact error
category, so focus on the literal meaning of the error message text, explain
any technical terms used, and suggest a general approach to investigating it.
""".strip(),
}


# ── User turn template ─────────────────────────────────────────────────────

_USER_TURN = """
Here is the GHC error the student encountered:

ERROR MESSAGE:
{ghc_message}

SOURCE CONTEXT (lines around the error):
{source_context}

FILE: {filename}
LINE: {line}, COLUMN: {col}

Please explain this error and provide a hint, following the format specified.
""".strip()


# ── Public API ─────────────────────────────────────────────────────────────

def build_system_prompt(
    category: ErrorCategory,
    level: ExperienceLevel,
) -> str:
    """
    Build the system prompt for a given error category and experience level.

    Parameters
    ----------
    category:
        The ErrorCategory of the diagnostic being explained.
    level:
        The student's current ExperienceLevel for this category.

    Returns
    -------
    str
        A fully-rendered system prompt ready to pass to the Claude API.
    """
    guidance = _CATEGORY_GUIDANCE.get(category, _CATEGORY_GUIDANCE[ErrorCategory.UNKNOWN])

    return _BASE_SYSTEM.format(
        level_description=level.describe(),
        category_guidance=guidance,
        no_code_rule=_NO_CODE_RULE,
        format_instruction=_FORMAT_INSTRUCTION,
    )


def build_user_prompt(
    ghc_message: str,
    source_context: str,
    filename: str,
    line: int,
    col: int,
) -> str:
    """
    Build the user-turn message containing the concrete diagnostic details.

    Parameters
    ----------
    ghc_message:
        The raw GHC error or warning message text.
    source_context:
        A short excerpt of the source file centred on the error location
        (typically ±3 lines). May be empty if the source is unavailable.
    filename:
        The basename of the file containing the error.
    line:
        The 1-indexed line number of the error.
    col:
        The 1-indexed column number of the error.

    Returns
    -------
    str
        A fully-rendered user-turn message.
    """
    if not source_context:
        source_context = "(source context not available)"

    return _USER_TURN.format(
        ghc_message=ghc_message,
        source_context=source_context,
        filename=filename,
        line=line,
        col=col,
    )


def parse_response(raw: str) -> tuple[str, str]:
    """
    Parse the structured Claude response into (explanation, hint).

    Expects the format:
        EXPLANATION: <text>
        HINT: <text>

    Falls back gracefully if the model does not follow the format exactly —
    returns the full response as the explanation and an empty hint.

    Parameters
    ----------
    raw:
        The raw text response from the Claude API.

    Returns
    -------
    tuple[str, str]
        (explanation, hint) — both stripped of leading/trailing whitespace.
    """
    explanation = ""
    hint = ""

    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("EXPLANATION:"):
            explanation = line[len("EXPLANATION:"):].strip()
        elif line.upper().startswith("HINT:"):
            hint = line[len("HINT:"):].strip()

    # Graceful fallback: if parsing failed, use full response as explanation
    if not explanation:
        explanation = raw.strip()

    return explanation, hint