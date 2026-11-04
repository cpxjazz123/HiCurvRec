"""The evidence-probe bridge: the project's own phase, artefacts included."""
from __future__ import annotations

from pathlib import Path

from tools.research.rounds import worker


def run_probe_phase(root: Path, round_id: str) -> dict:
    """Measure every proposal and freeze the evidence bundles."""
    return worker.phase_probe(Path(root), round_id)
