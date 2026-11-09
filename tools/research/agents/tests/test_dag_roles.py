"""The Issue #117 DAG tasks are document tasks, not ideation tasks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from tools.research.agents import runtime


def test_dag_roles_are_recognised():
    for role in ("behavior", "literature", "geometry", "review"):
        assert runtime.dag_role(role) is True
    assert runtime.dag_role("ideate") is False
    assert runtime.dag_role("verify") is False


def test_dag_roles_do_not_get_the_idea_schema():
    # Before the fix every unknown role fell through to IdeaPayload, and the
    # literature census was rendered as IDEA blocks the parser could not read.
    for role in ("behavior", "literature", "geometry", "review"):
        assert runtime.output_schema(role) is None
    assert runtime.output_schema("ideate") is runtime.IdeaPayload


def test_dag_instructions_are_not_idea_instructions():
    assert runtime.dag_role("literature")
    assert "IDEA" not in runtime.DAG_JSON_INSTRUCTIONS
    assert "JSON object" in runtime.DAG_JSON_INSTRUCTIONS
