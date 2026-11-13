"""The frozen ORCA role table and the geometry weights each arm routes with.

Issue #215 gives every arm the same frozen table: 73-dimensional ORCA five-node orbits, NMF with
k = 6 and seed 42, fitted on the TRAIN behaviour graph and written to disk once.  Nothing here
learns, and nothing here is allowed to change between the training run and the SID export: the
table is verified by sha256 on every load, and a mismatch is an error, never a silent fallback.

The four arms differ only in the routing:

    OFF       every row routes to the hyperbolic branch, which is the baseline's own geometry
    GLOBAL    every row routes to the population-weighted mean of the role routing
    ROLE      every row routes to the geometry of its own role
    SHUFFLED  the roles are permuted among the items with seed 42, so the routing histogram is
              the role arm's histogram with the labels moved elsewhere

Weights are per row and sum to one over the three branches: (euclidean, hyperbolic, spherical).
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import torch

#: The role -> space mapping of issue #215, in the role order the audit reports.
ROLE_SPACES: dict[int, str] = {
    0: "euclidean",    # dense hub
    1: "spherical",    # tight community
    2: "hyperbolic",   # two-hop / fan
    3: "euclidean",    # degree middle
    4: "spherical",    # cycle / long path
    5: "hyperbolic",   # star / chain
}

#: The order of the three branches in every weight vector.
SPACES = ("euclidean", "hyperbolic", "spherical")
SPACE_INDEX = {name: index for index, name in enumerate(SPACES)}

#: The seed the shuffled arm permutes with.
SHUFFLE_SEED = 42

MODES = ("OFF", "GLOBAL", "ROLE", "SHUFFLED")


class RoleTableError(RuntimeError):
    """The role table is missing, altered, or does not cover the items."""


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_role_table(path: Path, expected_sha256: str) -> dict[int, int]:
    """Read the frozen table and check its identity.

    The check is not a formality: the table is the only behaviour signal the arms may use, and an
    altered table would silently be a different mechanism.
    """
    path = Path(path)
    if not path.is_file():
        raise RoleTableError("the frozen role table is missing: " + str(path))
    digest = sha256_of(path)
    if digest != expected_sha256:
        raise RoleTableError(f"role table sha256 {digest} is not the frozen {expected_sha256}")
    roles: dict[int, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != ["item_id", "role"]:
            raise RoleTableError("unexpected role table header: " + repr(header))
        for row in reader:
            if len(row) != 2:
                raise RoleTableError("unexpected role table row: " + repr(row))
            item, role = int(row[0]), int(row[1])
            if role not in ROLE_SPACES:
                raise RoleTableError(f"unknown role {role} for item {item}")
            roles[item] = role
    if not roles:
        raise RoleTableError("the role table is empty")
    return roles


def role_of(roles: dict[int, int], item: int) -> int:
    """The role of one item; an item without a role is an error, not a default."""
    if item not in roles:
        raise RoleTableError(f"item {item} has no role in the frozen table")
    return roles[item]


def build_weights(mode: str, n_items: int, roles: dict[int, int],
                  train_ids: list[int] | None = None,
                  fallback_role: int = 0) -> tuple[torch.Tensor, int]:
    """The per-item routing weights, shape ``[n_items, 3]``, frozen.

    ``train_ids`` are the items the arms actually train on; the global arm's weights are the
    population-weighted mean over those items, as the issue specifies.

    An item the table does not cover -- an item with no TRAIN behaviour edge at all -- takes the
    documented fallback route instead of being dropped or given an empty mixture.  The number of
    such items is returned so the run can report it.
    """
    if fallback_role not in ROLE_SPACES:
        raise RoleTableError(f"unknown fallback role {fallback_role}")
    if mode not in MODES:
        raise RoleTableError(f"unknown role mode {mode}; expected one of {MODES}")
    if n_items <= 0:
        raise RoleTableError("no items")
    items = list(range(n_items))
    role_vector = torch.full((n_items,), int(fallback_role), dtype=torch.long)
    known = torch.zeros(n_items, dtype=torch.bool)
    for item in items:
        if item in roles:
            role_vector[item] = roles[item]
            known[item] = True
    if not bool(known.any()):
        raise RoleTableError("no item of the table is inside the item range")
    fallback_count = int((~known).sum().item())

    onehot = torch.zeros(n_items, len(SPACES), dtype=torch.float64)
    for role, space in ROLE_SPACES.items():
        onehot[role_vector == role, SPACE_INDEX[space]] = 1.0

    if mode == "OFF":
        weights = torch.zeros(n_items, len(SPACES), dtype=torch.float64)
        weights[:, SPACE_INDEX["hyperbolic"]] = 1.0
        return weights.to(torch.float32), fallback_count
    if mode == "ROLE":
        return onehot.to(torch.float32), fallback_count
    if mode == "GLOBAL":
        population = train_ids if train_ids is not None else items
        mask = torch.zeros(n_items, dtype=torch.bool)
        mask[torch.tensor(sorted(population), dtype=torch.long)] = True
        usable = mask & known
        if not bool(usable.any()):
            raise RoleTableError("the global arm has no usable training item")
        mean = onehot[usable].mean(dim=0)
        if float(mean.sum()) <= 0:
            raise RoleTableError("the global routing is empty")
        weights = mean.unsqueeze(0).repeat(n_items, 1)
        return weights.to(torch.float32), fallback_count

    # SHUFFLED: the same routing histogram, the labels moved to other items.
    generator = torch.Generator().manual_seed(SHUFFLE_SEED)
    permuted = role_vector[torch.randperm(n_items, generator=generator)]
    shuffled = torch.zeros(n_items, len(SPACES), dtype=torch.float64)
    for role, space in ROLE_SPACES.items():
        shuffled[permuted == role, SPACE_INDEX[space]] = 1.0
    return shuffled.to(torch.float32), fallback_count
