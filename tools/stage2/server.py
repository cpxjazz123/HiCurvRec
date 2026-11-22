#!/usr/bin/env python3
"""The Stage 2 MCP server: start, stop and inspect one Stage 2 training.

The Stage 2 and Stage 3 training surfaces used to share one server
(``tools/genrec-mcp/server.py``). The owner asked for them to be separate, so this
server exposes only the Stage 2 tools and that one keeps the rest. Both import the
same implementation, so there is one copy of the logic and two mount points: an
operator can attach or restart one stage without touching the other.

Run it as a stdio MCP server:

    python3 tools/stage2/server.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "tools" / "genrec-mcp" / "server.py"

#: The tools this server serves. Order is the order of `tools/list`.
TOOL_NAMES = ("stage2_start", "stage2_status", "stage2_stop")


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
    server.SERVER_NAME = "genrec-stage2"
    server.main()


if __name__ == "__main__":
    main()
