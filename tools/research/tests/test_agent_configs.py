"""The exported agent configurations must agree with the code that runs them.

``Agents/`` is a reviewable copy of every agent's configuration: the prompt
template, the model, the budget, the tools, the boundary and the shape of the
answer. It is DERIVED, so the only way it can be wrong is if the code changed
and the export was not re-run. This test is that alarm.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tools.research.agents import export_configs

REPO = Path(__file__).resolve().parents[3]
AGENTS = REPO / "Agents"


def test_every_role_has_a_directory():
    assert AGENTS.is_dir(), "the exported agent configurations are missing"
    for role in export_configs.ROLES:
        directory = AGENTS / export_configs.DIRECTORIES[role]
        assert directory.is_dir(), role
        for name in ("AGENT.md", "config.json", "output.schema.json", "prompt.md"):
            assert (directory / name).is_file(), "%s/%s" % (role, name)


def test_the_export_is_current():
    """The files in Agents/ are what the code says today."""
    for role in export_configs.ROLES:
        directory = AGENTS / export_configs.DIRECTORIES[role]
        expected = export_configs._config(role)
        actual = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        assert actual == expected, (
            "%s/config.json is stale: re-run "
            "`python3 -m tools.research.agents.export_configs`" % role)


def test_the_prompt_files_carry_the_code_they_name():
    """The prompt excerpt is the source the config names, not a copy of it."""
    for role in export_configs.ROLES:
        config = json.loads((AGENTS / export_configs.DIRECTORIES[role] / "config.json")
                            .read_text(encoding="utf-8"))
        source = config["prompt"]["source"]
        function = config["prompt"]["function"]
        text = (AGENTS / export_configs.DIRECTORIES[role] / "prompt.md").read_text(
            encoding="utf-8")
        assert source in text, role
        assert function in text, role
        # the excerpt is the code's own text
        body = export_configs._function_source(source, function)
        assert body, "%s: the prompt function %s was not found" % (role, function)
        first_line = body.splitlines()[0]
        assert first_line in text, "%s: the prompt excerpt is not the code's" % role


def test_the_schema_is_the_runtime_schema():
    from tools.research.agents import runtime

    for role in export_configs.ROLES:
        exported = json.loads((AGENTS / export_configs.DIRECTORIES[role] /
                               "output.schema.json").read_text(encoding="utf-8"))
        model = runtime.output_schema(role)
        if model is None:
            continue
        assert exported["title"] == model.__name__, role
        assert exported["properties"] == model.model_json_schema()["properties"], role


def test_the_readme_lists_every_role():
    text = (AGENTS / "README.md").read_text(encoding="utf-8")
    for role in export_configs.ROLES:
        assert "`%s`" % export_configs.DIRECTORIES[role] in text, role


def test_the_directories_are_named_after_their_step():
    """The order of the flow is legible from the directory name."""
    for role, name in export_configs.DIRECTORIES.items():
        if role == "verify":
            continue
        assert name.startswith("step"), name
        assert name.split("_", 1)[1] == role, name


def test_no_agent_may_write_outside_its_workspace():
    """The exported boundary says what the runtime enforces."""
    for role in export_configs.ROLES:
        config = json.loads((AGENTS / export_configs.DIRECTORIES[role] / "config.json")
                            .read_text(encoding="utf-8"))
        permissions = config["tools"]
        assert permissions["write_root"].startswith("the agent's own workspace"), role
        assert permissions["shell"] is False, role
        assert "append_progress" in permissions["tools"], role
