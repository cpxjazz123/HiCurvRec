"""Issue #230 §2: build the epoch order so a batch carries whole graphlet instances.

Round 2 measured the limit precisely. The structure term works on what it touches -- constrained
pairs share a level-1 codeword 19-25x more often than chance, at 12.6% of the reconstruction
gradient -- but a uniform batch of 4096 items contains only 104 whole triples and 766 pairwise
constraints, from banks holding 20,827 instances. Every epoch therefore pushes ~2% of the bank
and lets the reconstruction gradient undo it before the term returns.

This module is framework code: it decides the **order items are fed in**, nothing else. The
candidate tree keeps its own loader; the launcher hands it this order. Batch content stays a
uniform draw over items -- only the *arrangement* changes, so the epoch still visits every row
exactly once and the optimisation recipe (batch size, steps per epoch, seed) is untouched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

KINDS = ("chain", "fork", "merge")
#: The recipe seed; the order must not be a free parameter (§19g).
DEFAULT_SEED = 42


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "stage2_RQ-VAE").is_dir() and (parent / "results").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root above " + __file__)


#: The frozen banks the issue requires (C7-C11); the same files the term reads.
BANK_DIR = _repo_root() / "results" / "research" / "issue230" / "banks"


def load_banks(directory: Path = BANK_DIR) -> dict:
    """The supervision banks, with the same hash check the term performs."""
    manifest = json.loads((Path(directory) / "banks.json").read_text())
    banks = {}
    for kind in KINDS:
        bank = np.load(Path(directory) / f"{kind}_supervision.npy")
        digest = hashlib.sha256(np.ascontiguousarray(bank).tobytes()).hexdigest()
        expected = manifest["kinds"][kind]["partitions"]["supervision"]
        if digest != expected:
            raise RuntimeError(f"{kind} bank does not match banks.json: {digest} != {expected}")
        banks[kind] = bank
    return banks


def build_order(banks: dict, universe: int, seed: int = 42) -> np.ndarray:
    """A permutation of ``range(universe)`` that places instance partners adjacently.

    Items are emitted instance-first: for each bank row, its three items are written next to
    each other. An item that appears in several rows is emitted once, at its first occurrence,
    so the result is a permutation -- no duplicates, no drops, and the epoch still covers every
    row exactly once.

    The order is row-major over the concatenated banks, so a batch of any size takes a
    contiguous slice and receives whole triples until the slice runs out of fresh instances.
    Items no bank mentions are appended (shuffled) at the end: they carry no structure
    constraint, so their position does not affect the term.
    """
    parts = []
    seen = np.zeros(int(universe), dtype=bool)
    for kind in KINDS:
        if kind not in banks:
            continue
        for row in banks[kind].tolist():
            fresh = [value for value in row if not seen[value]]
            for value in fresh:
                seen[value] = True
            if fresh:
                parts.append(np.asarray(fresh, dtype=np.int64))
    mentioned = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    rest = np.setdiff1d(np.arange(int(universe), dtype=np.int64), mentioned)
    rng = np.random.default_rng(seed)
    order = np.concatenate([mentioned, rest[rng.permutation(len(rest))]])
    if len(order) != int(universe) or len(np.unique(order)) != int(universe):
        raise RuntimeError("build_order did not return a permutation")
    return order


def coverage(banks: dict, order: np.ndarray, batch_size: int) -> dict:
    """How many whole instances a batch of this size carries, per structure."""
    report = {}
    for kind in KINDS:
        if kind not in banks:
            continue
        batch = set(int(value) for value in order[:batch_size].tolist())
        rows = banks[kind].tolist()
        report[kind] = sum(1 for row in rows if all(value in batch for value in row))
    return report


#: The launcher writes the order here and the trainer reads it; the candidate never builds it.
ORDER_FILENAME = "graph_batch_order.npy"


def install(result_dir: Path | str, universe: int, seed: int = 42) -> Path | None:
    """Write the epoch order into a result directory; ``None`` when no banks exist.

    A tree that does not read the file is unaffected, so this is safe to call on every
    candidate: the change reaches only the trees the issue is about.
    """
    try:
        banks = load_banks()
    except (FileNotFoundError, RuntimeError):
        return None
    order = build_order(banks, universe=int(universe), seed=seed)
    target = Path(result_dir) / ORDER_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, order)
    return target
