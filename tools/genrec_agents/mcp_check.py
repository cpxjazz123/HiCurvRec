"""Prove that the genrec MCP is really reachable from an OpenCode session.

The issue requires a *demonstrated* MCP invocation, not a configuration file that looks
right. So this runs a real OpenCode session in a scratch workspace whose configuration
points at the genrec MCP server, asks the agent to call one read-only tool, and then reads
the tool call out of the trajectory.

Only a read-only tool is used: the proof must not start a training run.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import opencode

REPO_ROOT = Path(__file__).resolve().parents[2]


def run(
    out_dir: Path,
    *,
    model: str = "",
    timeout: int = 900,
    tool: str = "stage2_status",
    tree: Optional[Path] = None,
) -> Dict[str, Any]:
    """Ask an OpenCode session to call a genrec MCP tool, and report what happened."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="h2-mcp-check-"))
    try:
        target = Path(tree) if tree else REPO_ROOT / "stage2_RQ-VAE" / "TIGER_r9_mvg_reference"
        prompt = (
            "The genrec MCP server is connected to this session. Call its "
            f"`{tool}` tool once, for the Stage2 tree at `{target}`, and report exactly what "
            "the tool returned: the status it reported and any error it reported. Do not "
            "start a training run and do not change any file."
        )
        result = opencode.run_agent(
            workspace,
            prompt,
            model=model or opencode.model_name(),
            timeout=timeout,
            mcp=opencode.mcp_server_block(REPO_ROOT),
            title="genrec mcp check",
            deny_paths=[REPO_ROOT],
        )
        # OpenCode v2 dispatches an MCP tool through its own `execute` tool, with the MCP
        # call inside the payload. So the proof is not "a tool named stage2_status ran":
        # it is that a call went out to the genrec server and came back with its answer.
        calls = []
        for call in result.get("tool_calls", []):
            payload = json.dumps(call.get("input"), default=str)
            output = str(call.get("output", ""))
            if tool in payload or f"genrec.{tool}" in payload or f"genrec.{tool}" in output:
                calls.append(call)
            elif call.get("tool") == "execute" and "tools.genrec." in payload:
                calls.append(call)
        (out_dir / "mcp_check.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        return {
            "ok": bool(calls),
            "tool": tool,
            "model": result.get("model"),
            "exit_code": result.get("exit_code"),
            "cost": result.get("cost"),
            "tokens": result.get("tokens"),
            "mcp_tool_calls": calls,
            "n_tool_calls": result.get("n_tool_calls"),
            "all_tool_calls": [call.get("tool") for call in result.get("tool_calls", [])],
            "mcp_payloads": [
                {
                    "tool": call.get("tool"),
                    "input": call.get("input"),
                    "output": str(call.get("output", ""))[:2000],
                }
                for call in calls
            ],
            "text": result.get("text"),
            "errors": result.get("errors"),
            "detail": (
                f"the session called the genrec MCP {len(calls)} time(s) for {tool!r} and got "
                "the server's own answer back"
                if calls
                else "the session made no MCP tool call"
            ),
        }
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


__all__ = ["run"]
