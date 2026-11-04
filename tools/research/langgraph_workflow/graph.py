"""The research graph: ideation, probe, scoring, and nothing else.

Phase 1 replaces the orchestration of the first three phases only. The
implementation, training and evaluation phases stay exactly where they are, and
the round's state machine is the existing one: the graph calls the same
transitions, so the two workflows must agree.
"""
from __future__ import annotations

from pathlib import Path

from langgraph.graph import END, START, StateGraph

from tools.research.langgraph_workflow.nodes import ideate, probe, score
from tools.research.langgraph_workflow.state import ResearchState


def _prepare(state: dict) -> dict:
    """The node the branches leave from; it only carries the round's identity."""
    return {}


def build_graph(checkpointer=None):
    """Compile the research graph."""
    builder = StateGraph(ResearchState)
    builder.add_node("idea_agent", ideate.idea_agent)
    builder.add_node("merge_ideas", ideate.merge_ideas)
    builder.add_node("probe", probe.probe_node)
    builder.add_node("scorer", score.scorer)
    builder.add_node("finalize", score.finalize)
    builder.add_node("prepare", _prepare)
    builder.add_edge(START, "prepare")
    builder.add_conditional_edges("prepare", ideate.dispatch_ideation, ["idea_agent"])
    builder.add_edge("idea_agent", "merge_ideas")
    builder.add_edge("merge_ideas", "probe")
    builder.add_conditional_edges("probe", score.dispatch_scoring, ["scorer"])
    builder.add_edge("scorer", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)


def default_checkpointer():
    """A checkpointer when one is available, else an in-memory one.

    A round is resumed by its ``round_id``: the checkpoint keeps the state of a
    graph that was interrupted between two phases.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except Exception:                                          # noqa: BLE001
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()
    return SqliteSaver.from_conn_string("")


def run_phase1(root: Path, round_id: str, *, objective: str = "", extra: bool = False,
               checkpointer=None, config: dict | None = None) -> dict:
    """Run the first three phases of a round through the graph.

    ``thread_id`` is the round id, so the state of an interrupted run is
    recovered by running the same call again.
    """
    graph = build_graph(checkpointer=checkpointer)
    state = {"round_id": round_id, "objective": objective, "root": str(root),
             "extra": extra}
    configuration = config or {"configurable": {"thread_id": round_id}}
    return graph.invoke(state, configuration)
