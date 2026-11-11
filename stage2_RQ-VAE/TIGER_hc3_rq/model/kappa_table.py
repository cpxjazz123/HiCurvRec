"""Loading the frozen prefix-conditioned curvature table and resolving per-row kappas (issue #177).

The table is a research artifact, so it is found through the repo marker rather than an absolute path (rule 19b0): walk up from
this file until a directory holds ``stage2_RQ-VAE``, ``results`` and ``stage1_GeneEmbedding``, then read
``results/research/issue177/kappa_prefix.json``. Every level's kappa is a pure function of (level, prefix), so every rank of a
DDP job resolves the same value from the same frozen table without any communication.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from . import kappa_metric as km

#: Relative location of the artifact inside the repository.
TABLE_RELPATH = Path("results") / "research" / "issue177" / "kappa_prefix.json"


def repo_root(start: Path | None = None) -> Path:
    """The repository root: the nearest ancestor holding the three marker directories."""
    here = (start or Path(__file__).resolve()).resolve()
    for candidate in (here, *here.parents):
        if all((candidate / marker).is_dir() for marker in ("stage2_RQ-VAE", "results", "stage1_GeneEmbedding")):
            return candidate
    raise FileNotFoundError("No repository root with stage2_RQ-VAE, results and stage1_GeneEmbedding above this module.")


_CACHE: dict | None = None


def load() -> dict:
    """The table, read once per process."""
    global _CACHE
    if _CACHE is None:
        path = repo_root() / TABLE_RELPATH
        _CACHE = json.loads(path.read_text())
    return _CACHE


def kappa_for_rows(prefixes, level: int, table: dict | None = None) -> torch.Tensor:
    """A ``[B, 1]`` float tensor of kappas for the rows' prefixes at ``level``.

    ``prefixes`` is a sequence of tuples (or an int tensor of shape ``[B, level-1]``); level 1 ignores it and returns the
    frozen 1.0 for every row, which is what keeps the first layer identical to the reference arm.
    """
    table = load() if table is None else table
    if level == 1:
        rows = len(prefixes)
        return torch.full((rows, 1), float(table["level_1"]["kappa"]), dtype=torch.float64)
    if torch.is_tensor(prefixes):
        prefixes = [tuple(int(value) for value in row) for row in prefixes.cpu().tolist()]
    values = [km.prefix_kappa(table, level, tuple(prefix)) for prefix in prefixes]
    return torch.tensor(values, dtype=torch.float64).reshape(-1, 1)
