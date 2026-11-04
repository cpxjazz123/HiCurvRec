"""The evidence probe: the project's own phase, driven by the graph."""
from __future__ import annotations

from pathlib import Path

from tools.research.langgraph_workflow.adapters import probe
from tools.research.rounds import pipeline as pipeline_module


def probe_node(state: dict) -> dict:
    """Measure every proposal and freeze the evidence bundles."""
    root = Path(state["root"])
    result = probe.run_probe_phase(root, state["round_id"])
    directory = pipeline_module.find_round_directory(root, state["round_id"])
    round_state = pipeline_module.load_state(directory)
    return {"probe_phase": result, "round_state": round_state,
            "phase_results": {"probe": result}}
