"""The review bridge: the same question, asked by the same reviewer code."""
from __future__ import annotations

from pathlib import Path

from tools.research.rounds import worker


def scoring_inputs(root: Path, ideas: list, *, repo: Path = None,
                   state: dict = None) -> tuple:
    """The frozen bundle, the proposal text and the evidence text."""
    return worker.scoring_inputs(Path(root), ideas, repo=repo, state=state)


def rank_one(root: Path, payload: str, evidence_text: str, index: int) -> dict:
    """One reviewer's answer."""
    return worker.rank_one(Path(root), payload, evidence_text, index)
