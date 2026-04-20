"""
main.py — Entry point for the AI-Assisted Haskell Language Server.

Two operating modes
-------------------
  --mode lsp   Start the LSP server (JSON-RPC over stdio).
               Used by VS Code and any other LSP-capable editor.
               This is the default mode.

  --mode web   Start the web server (FastAPI + WebSocket).
               Used by the custom browser-based editor.

Usage
-----
  # VS Code / LSP mode (invoked automatically by the VS Code extension)
  python -m server.main

  # Web editor mode
  python -m server.main --mode web

  # Or via the installed entry points (after pip install -e .):
  haskell-lsp
  haskell-lsp-web

Environment variables
---------------------
  GROQ_API_KEY    Required. Your Groq API key.
  GHC_PATH        Optional. Path to GHC if not on PATH.
  GROQ_MODEL      Optional. Default: llama-3.1-8b-instant
  WEB_SERVER_PORT Optional. Default: 8765
  LOG_LEVEL       Optional. Default: INFO
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

# Load .env before anything else so env vars are available to all modules
load_dotenv()

# ── Logging setup ──────────────────────────────────────────────────────────

def _configure_logging(level_name: str = "INFO") -> None:
    """
    Configure logging for the server process.

    In LSP mode we log to stderr (stdout is reserved for JSON-RPC).
    In web mode we log to stdout as usual.
    """
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


# ── Entry points ───────────────────────────────────────────────────────────

def run_lsp() -> None:
    """
    Start the LSP server in stdio mode.

    pygls reads JSON-RPC messages from stdin and writes responses to stdout.
    All logging goes to stderr so it does not corrupt the JSON-RPC stream.

    This function is registered as the `haskell-lsp` console script entry
    point in pyproject.toml.
    """
    _configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
    logger = logging.getLogger(__name__)
    logger.info("Starting Haskell AI LSP server (stdio mode)")

    # Import here to avoid circular imports and to defer pygls loading
    from server.lsp_server import server
    server.start_io()


def run_web() -> None:
    """
    Start the FastAPI WebSocket server for the browser-based editor.

    This function is registered as the `haskell-lsp-web` console script
    entry point in pyproject.toml.
    """

    _configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
    logger = logging.getLogger(__name__)

    # FIX: Priority is Railway's PORT, then your custom WEB_SERVER_PORT, then 8765
    port = int(os.environ.get("PORT", os.environ.get("WEB_SERVER_PORT", 8765)))
    
    logger.info("Starting Haskell AI web server on port %d", port)

    import uvicorn
    from server.web_server import app
    
    # Ensure uvicorn uses the correct port variable
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    
# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="haskell-lsp",
        description="AI-Assisted Haskell Language Server",
    )
    parser.add_argument(
        "--stdio", 
        action="store_true", 
        default=False, 
        help="Ignored — stdio is always used in LSP mode."
    )

    parser.add_argument(
        "--mode",
        choices=["lsp", "web"],
        default="lsp",
        help="Operating mode: 'lsp' for editor integration (default), 'web' for browser editor",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level: DEBUG, INFO, WARNING, ERROR (overrides LOG_LEVEL env var)",
    )
    args = parser.parse_args()

    if args.log_level:
        os.environ["LOG_LEVEL"] = args.log_level

    if args.mode == "lsp":
        run_lsp()
    else:
        run_web()


if __name__ == "__main__":
    main()