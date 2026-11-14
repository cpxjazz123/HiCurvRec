"""Per-process view of the current item ids.

The host sets this before *any* call that may look at per-item data:

- the training loop sets the current batch's ids before every forward;
- codebook initialisation sets the training ids (rows of ``train_embeddings``);
- export/eval inference sets ``arange(len(all_embeddings))`` (rows of the full table).

Any mechanism file (``distances/distance.py``, ``losses/*.py``, ``model/layers.py``) can
then say ``from runtime_context import item_ids`` and read the identity of each row
without touching the trainer, the dataset or the model signature.

Process-local by design: under DDP every rank sets its own batch, so there is no
cross-rank state to synchronise.
"""

from __future__ import annotations

import threading

import torch

_state = threading.local()


def set_item_ids(item_ids, *, phase: str = "train") -> None:
    """Set the ids for the rows the next call will see. ``None`` clears them."""
    if item_ids is None:
        _state.item_ids = None
    else:
        _state.item_ids = torch.as_tensor(item_ids).reshape(-1).to(torch.int64)
    _state.phase = phase


def item_ids():
    """The ids aligned with the current rows, or None when the host has not set them."""
    return getattr(_state, "item_ids", None)


def phase() -> str:
    """``train``, ``init`` or ``eval`` — useful for mechanisms with phase-dependent reads."""
    return getattr(_state, "phase", "train")


def clear() -> None:
    _state.item_ids = None
    _state.phase = "train"
