"""An exact embedding-row -> item id map, for losses that need graph node identity.

The trainer's loader yields rows only, so a supervision that speaks in graph edges cannot tell
which item a row is. This map is built once from the frozen embedding matrix and matched on the
row's exact bytes: an unknown or perturbed row raises instead of being silently assigned to a
neighbour, which is what keeps the edge-restricted loss honest.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch


def _digest(row: np.ndarray) -> bytes:
    return hashlib.blake2b(np.ascontiguousarray(row).tobytes(), digest_size=16).digest()


def build(embeddings: np.ndarray) -> dict:
    """``bytes -> item id`` for every row of the frozen matrix, in order."""
    return {_digest(embeddings[index]): index for index in range(embeddings.shape[0])}


def lookup(rows: torch.Tensor, index: dict) -> torch.Tensor:
    """The item ids of these rows; a row that is not in the matrix raises ``KeyError``."""
    array = rows.detach().to(torch.float32).cpu().numpy()
    ids = []
    for row in array:
        key = _digest(row)
        if key not in index:
            raise KeyError("row not present in the frozen embedding matrix")
        ids.append(index[key])
    return torch.tensor(ids, dtype=torch.long)
