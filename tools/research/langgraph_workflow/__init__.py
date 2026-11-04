"""Phase 1 of the LangGraph migration: ideation, probe and scoring.

The graph owns the orchestration (who runs, in parallel, over which state); the
work itself stays in the existing executors, so the two workflows cannot drift
apart. The round's state machine is unchanged: the graph drives the same
``apply_ideation`` / ``phase_probe`` / ``apply_scoring`` transitions the
concurrent runner drives.
"""
from tools.research.langgraph_workflow.graph import build_graph, run_phase1

__all__ = ["build_graph", "run_phase1"]
