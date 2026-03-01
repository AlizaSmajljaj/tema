"""
bridge.py — Invokes GHC on a Haskell source file and returns a CompilationResult.

Design decisions:
  - We run GHC with -fno-code so it only type-checks without producing .o/.hi files.
  - We write document content to a temp file so we can check unsaved editor buffers.
  - We attempt to enable JSON diagnostics (GHC >= 9.4); fall back to classic text.
  - Compilation is run in a thread pool so it doesn't block the async LSP event loop.
  - Results are cached by (content_hash) to avoid re-running GHC on identical code.
"""

import asyncio
import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from .models import CompilationResult
from .parser import parse_ghc_output

logger = logging.getLogger(__name__)

# Thread pool for running GHC (blocking subprocess) without blocking the event loop
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ghc-worker")

# Simple in-memory LRU-style cache: hash → CompilationResult
_cache: dict[str, CompilationResult] = {}
_CACHE_MAX = 50


def _find_ghc() -> str:
    """Locate the GHC executable, raising EnvironmentError if not found."""
    ghc = shutil.which("ghc")
    if ghc is None:
        raise EnvironmentError(
            "GHC not found on PATH. "
            "Please install GHC (https://www.haskell.org/ghc/) "
            "or set the GHC_PATH environment variable."
        )
    return os.environ.get("GHC_PATH", ghc)


def _build_ghc_command(ghc_path: str, hs_file: str, use_json: bool) -> list[str]:
    """
    Build the GHC invocation command.

    Flags used:
      -fno-code        — skip code generation; type-check only (fast)
      -Wall            — enable all warnings (educational value)
      -fshow-warning-groups — show which warning group each warning belongs to
      -fforce-recomp   — always recompile (we're checking a temp file)
    """
    cmd = [
        ghc_path,
        "-fno-code",
        "-Wall",
        "-fshow-warning-groups",
        "-fforce-recomp",
    ]
    if use_json:
        cmd.append("-fdiagnostics-as-json")
    cmd.append(hs_file)
    return cmd


def _run_ghc_sync(source: str, original_path: str) -> CompilationResult:
    """
    Synchronous GHC invocation. Runs in thread pool via run_ghc().

    Writes source to a temp file, runs GHC, parses output, cleans up.
    """
    # --- Cache lookup ---
    content_hash = hashlib.sha256(source.encode()).hexdigest()
    if content_hash in _cache:
        logger.debug("Cache hit for %s", original_path)
        return _cache[content_hash]

    # --- Write temp file ---
    # We preserve the original filename so GHC error messages show the real path.
    suffix = Path(original_path).suffix or ".hs"
    tmp_dir = tempfile.mkdtemp(prefix="haskell-lsp-")
    tmp_path = os.path.join(tmp_dir, Path(original_path).name)

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(source)

        ghc_path = _find_ghc()

        # --- Try JSON first (GHC >= 9.4), fall back to classic ---
        for use_json in (True, False):
            cmd = _build_ghc_command(ghc_path, tmp_path, use_json=use_json)
            logger.debug("Running GHC: %s", " ".join(cmd))

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,          # 30s hard timeout
                    cwd=tmp_dir,
                )
            except subprocess.TimeoutExpired:
                logger.error("GHC timed out on %s", original_path)
                return CompilationResult(file=original_path, success=False,
                                         raw_stderr="GHC timed out after 30 seconds.")

            stderr = proc.stderr
            # GHC on some platforms writes diagnostics to stdout in JSON mode
            if use_json and not stderr.strip():
                stderr = proc.stdout

            # If JSON mode produced no JSON lines, retry with classic
            if use_json and not any(l.strip().startswith("{") for l in stderr.splitlines()):
                logger.debug("JSON mode produced no JSON output, retrying classic")
                continue

            # Rewrite temp path → original path in messages so the editor
            # can correctly map errors back to the open document.
            stderr_mapped = stderr.replace(tmp_path, original_path)

            result = parse_ghc_output(
                stderr=stderr_mapped,
                target_file=original_path,
                exit_code=proc.returncode,
            )

            # Remap spans from temp path to original path
            for diag in result.diagnostics:
                if diag.span.file == tmp_path:
                    diag.span.file = original_path

            # --- Cache result ---
            if len(_cache) >= _CACHE_MAX:
                # Evict oldest entry (insertion-order dict in Python 3.7+)
                _cache.pop(next(iter(_cache)))
            _cache[content_hash] = result

            return result

    except EnvironmentError as e:
        logger.error("GHC environment error: %s", e)
        return CompilationResult(
            file=original_path,
            success=False,
            raw_stderr=str(e),
        )
    except Exception as e:
        logger.exception("Unexpected error running GHC on %s", original_path)
        return CompilationResult(
            file=original_path,
            success=False,
            raw_stderr=f"Internal error: {e}",
        )
    finally:
        # Always clean up temp directory
        try:
            import shutil as _shutil
            _shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


async def run_ghc(source: str, file_path: str) -> CompilationResult:
    """
    Asynchronously compile a Haskell source string.

    This is the primary entry point used by lsp_server.py and web_server.py.

    Args:
        source:     Full text content of the Haskell document.
        file_path:  The URI-decoded absolute path of the open document.
                    Used for error message path mapping.

    Returns:
        CompilationResult containing all diagnostics.

    Example:
        result = await run_ghc(document_text, "/home/user/Project.hs")
        for diag in result.errors:
            print(diag)
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor,
        _run_ghc_sync,
        source,
        file_path,
    )
    return result


def clear_cache() -> None:
    """Clear the compilation cache. Useful for testing."""
    _cache.clear()


async def get_ghc_version() -> Optional[str]:
    """Return the installed GHC version string, or None if GHC is not found."""
    try:
        ghc = _find_ghc()
        loop = asyncio.get_event_loop()

        def _version():
            proc = subprocess.run(
                [ghc, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            # "The Glorious Glasgow Haskell Compilation System, version 9.4.7"
            match = __import__("re").search(r"version\s+([\d.]+)", proc.stdout + proc.stderr)
            return match.group(1) if match else proc.stdout.strip()

        return await loop.run_in_executor(_executor, _version)
    except Exception:
        return None