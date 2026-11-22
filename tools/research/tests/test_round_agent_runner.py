"""A detached round must run where the Agents SDK lives.

The fault this refuses: the MCP server is a stdlib-only python3 (it is started
as ``/usr/bin/python3``, which is 3.9 here), and the OpenAI Agents SDK is
installed for another interpreter (the settings name it as
``AGENT_RUNNER_PYTHON``).  The detached launchers used to start the server's
own interpreter, so every agent died on ``from agents import Agent`` and the
round recorded five failures with no answer - classified ``unknown``, because
an import error is not one of the runner's failure classes.  Measured on
round_20261106_134543_991: five ideation agents, five ``unknown`` failures, no
idea, and a round that died in its first second.

The test asserts two things: the command the detached launchers build runs the
round's agent phases through the runner interpreter, and that interpreter can
actually import the SDK the agents need.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tools.research import settings as research_settings  # noqa: E402


def _server_module():
    spec = importlib.util.spec_from_file_location(
        "genrec_server_for_test", REPO / "tools" / "genrec-mcp" / "server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_detached_command_runs_the_phases_in_the_agent_runner():
    server = _server_module()
    command = server._agent_runner_command("round_test_agents")
    assert command[0] == research_settings.AGENT_RUNNER_PYTHON
    assert command[1:3] == ["-m", "tools.research.agent_flow.run"]
    assert "--round" in command and "round_test_agents" in command


def test_the_agent_runner_interpreter_carries_the_sdk():
    runner = Path(research_settings.AGENT_RUNNER_PYTHON)
    assert runner.exists(), "the agent runner interpreter is missing: " + str(runner)
    import subprocess

    done = subprocess.run(
        [str(runner), "-c", "import agents, sys; print(agents.__file__, sys.version.split()[0])"],
        capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, ("the agent runner cannot import the Agents SDK: "
                                  + (done.stderr or done.stdout)[-400:])
    assert "agents" in done.stdout


def test_the_server_does_not_spawn_the_phases_in_its_own_interpreter():
    """The legacy worker path is gone: the phases run through the runner."""
    source = (REPO / "tools" / "genrec-mcp" / "server.py").read_text(encoding="utf-8")
    assert "ROUND_WORKER_TEMPLATE" not in source
    assert "write_round_worker" not in source
    assert "round_worker_main" not in source
