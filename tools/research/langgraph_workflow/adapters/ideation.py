"""The ideation bridge: one agent, one answer, the existing executor."""
from __future__ import annotations

from pathlib import Path

from tools.research.rounds import worker


def ideate_one(root: Path, objective: str, extra: bool, index: int) -> tuple:
    """Run one ideation agent through the project's own executor.

    The unit is the same one the concurrent runner uses, so a graph branch and a
    thread ask the agent exactly the same question and read the same file.
    """
    return worker.ideate_one(Path(root), objective or "", extra, index)
