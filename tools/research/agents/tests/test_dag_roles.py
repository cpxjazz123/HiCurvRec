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


def test_progress_is_not_the_answer_of_a_dag_task():
    records = [{"content": "STEP 1 - workspace inspection"}]
    assert runtime.answer_text_for("literature", '{"sources": []}', records) == '{"sources": []}'
    # ideation still reads progress, which is how a long block arrives
    assert runtime.answer_text_for("ideate", "", records) == "STEP 1 - workspace inspection"


def test_the_shared_prompt_does_not_impose_the_idea_block():
    # The old SYSTEM_PROMPT told every role to answer in an IDEA block, which the
    # literature census obeyed even after the role dispatch was fixed.
    assert "IDEA:" not in runtime.SYSTEM_PROMPT
    assert "END IDEA" not in runtime.SYSTEM_PROMPT
    assert "IDEA:" in runtime.IDEA_JSON_INSTRUCTIONS


def test_dag_instructions_are_not_idea_instructions():
    assert runtime.dag_role("literature")
    assert "IDEA" not in runtime.DAG_JSON_INSTRUCTIONS
    assert "JSON object" in runtime.DAG_JSON_INSTRUCTIONS
