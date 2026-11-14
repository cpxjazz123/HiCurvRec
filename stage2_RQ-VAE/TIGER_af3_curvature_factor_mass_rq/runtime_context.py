"""Per-process channel for "the data the current rows carry".

The host does not decide what a mechanism needs. It publishes whatever fields the
batches carry as a **named mapping** before every call that might look at row-level
data, and any mechanism file reads what it wants:

    from runtime_context import get, fields, phase
    ids = get("item_id")        # None when the host did not publish that field

The default batch fields are ``item_id`` (and ``row`` for the full-table passes).
A mechanism that needs something else (user id, timestamp, per-item curvature, …)
adds that field in the dataset file ``dataset_fields.py`` and reads it here — both
are ordinary purpose files, no host change required.

Process-local by design: under DDP every rank publishes its own batch.
"""

from __future__ import annotations

import threading
from typing import Any, Mapping

_state = threading.local()


def set_fields(values: Mapping[str, Any] | None, *, phase: str = "train") -> None:
    """Publish the fields aligned with the rows the next call will see."""
    _state.values = dict(values) if values else {}
    _state.phase = phase


def get(name: str, default: Any = None) -> Any:
    """One published field, or ``default`` when the host did not provide it."""
    return getattr(_state, "values", {}).get(name, default)


def fields() -> dict[str, Any]:
    """All published fields (a copy; mutating it does not affect the channel)."""
    return dict(getattr(_state, "values", {}))


def has(name: str) -> bool:
    return name in getattr(_state, "values", {})


def phase() -> str:
    """``train``, ``init`` or ``eval``."""
    return getattr(_state, "phase", "train")


def clear() -> None:
    _state.values = {}
    _state.phase = "train"
