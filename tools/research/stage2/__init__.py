"""The launch kernel the frozen baseline and every candidate import.

Only three things survive here after the 2026-11-22 retirement of the research
orchestration: the repository-root/result-directory contract, the single four-rank
launcher, and the anti-collapse assignment floor the launcher installs. The
trainers import ``contract`` and ``launcher``; nothing else is needed to run a
Stage 2 training, so nothing else is kept.
"""
from . import contract, launcher

__all__ = ["contract", "launcher"]
