#!/usr/bin/env python3
"""The Stage 3 MCP server: start and inspect one Stage 3 training.

Split out of the shared research server at the owner's request (2026-11-22): the
Stage 2 surface lives in ``tools/stage2/server.py``, this one carries Stage 3, and
``tools/genrec-mcp/server.py`` keeps the research-round and verification tools.
The three servers share one implementation, so the SID quality gate, the port and
the frozen protocol are the same code in every mount point.

Run it as a stdio MCP server:

    python3 tools/stage3/server.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "tools" / "genrec-mcp" / "server.py"

#: The tools this server serves. Order is the order of `tools/list`.
TOOL_NAMES = ("stage3_start", "stage3_status")


def load_shared():
    """Import the shared implementation without running its main loop."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("genrec_mcp_server", SHARED)
    if spec is None or spec.loader is None:
        raise RuntimeError("the shared MCP implementation cannot be loaded: " + str(SHARED))
    module = importlib.util.module_from_spec(spec)
    sys.modules["genrec_mcp_server"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    server = load_shared()
    missing = [name for name in TOOL_NAMES if name not in server.ALL_TOOLS]
    if missing:
        raise RuntimeError("the shared implementation lost these tools: " + ", ".join(missing))
    server.TOOLS = {name: server.ALL_TOOLS[name] for name in TOOL_NAMES}
    server.SERVER_NAME = "genrec-stage3"
    server.main()


if __name__ == "__main__":
    main()
