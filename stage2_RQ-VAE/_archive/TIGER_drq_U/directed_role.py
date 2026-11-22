"""Issue #221 phase 3: the directed Type+Role head and its instance batches.

The head reads a representation per node (encoder z, or a quantised q) and emits

    type logits over the directed SUBTYPE vocabulary (canonical directed pattern),
    role logits over a global directed-ORBIT vocabulary, masked to the orbits that
    the instance's own subtype can take.

Type readout pools over nodes (permutation invariant), role readout keeps one row
per node (permutation equivariant) and adds the pooled context, so a node's role is
decided by the instance it sits in.  Neither readout ever sees the adjacency, the
labels or a node-position id.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def _issue_dir() -> Path:
    """The issue directory, found by the repository marker.

    The module is copied into every arm tree under stage2_RQ-VAE/, so its own
    parent is not the vocabulary's home; the marker makes both copies resolve the
    same subtypes.json and instances.npz.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "stage2_RQ-VAE").is_dir() and (parent / "results").is_dir():
            return parent / "results" / "research" / "directed_graphlet_role"
    raise RuntimeError("cannot locate results/research/directed_graphlet_role above " + str(__file__))


HERE = _issue_dir()
FAMILIES = ("K3", "P3", "C4", "P4", "K1_3")


UNDIRECTED_POOL = "results/research/graphlet_roles/instances.npz"


def load_vocabulary(mode: str = "directed") -> dict:
    raw = json.loads((HERE / "subtypes.json").read_text())
    if mode == "undirected":
        # one class per family, and the undirected orbit inside that family: the
        # "direction removed" control reads exactly the labels #217's head read
        subtype_of = {family: index for index, family in enumerate(FAMILIES)}
        orbits_of = {index: list(range(4)) for index in subtype_of.values()}
        # joint_loss indexes this table by subtype id, so the undirected vocabulary
        # needs one too: inside a family every orbit class is admissible.
        allowed = torch.ones(len(subtype_of), 4, dtype=torch.bool)
        return {"mode": mode, "subtype_of": subtype_of, "orbits_of": orbits_of,
                "allowed": allowed,
                "n_subtypes": len(FAMILIES), "n_orbits": 4, "raw": raw}
    subtype_of: dict[str, int] = {}
    orbits_of: dict[int, list[int]] = {}
    orbit_vocab: dict[tuple[str, str, int], int] = {}
    for family in FAMILIES:
        for key, index in raw[family].items():
            if key == "__other__":
                continue
            subtype_id = len(subtype_of)
            subtype_of[f"{family}:{key}"] = subtype_id
            orbit_ids = []
            for orbit in range(8):          # at most four nodes -> at most four orbits
                orbit_vocab[(family, key, orbit)] = len(orbit_vocab)
                orbit_ids.append(orbit_vocab[(family, key, orbit)])
            orbits_of[subtype_id] = orbit_ids
    # One (subtype x orbit) boolean table instead of a Python loop that built
    # thousands of tiny GPU tensors per step: the loop was both slow and a
    # plausible source of the silent death the aux arms hit around step 10,560.
    allowed = torch.zeros(len(subtype_of), len(orbit_vocab), dtype=torch.bool)
    for subtype_id, orbits in orbits_of.items():
        allowed[subtype_id, orbits] = True
    return {"mode": mode, "subtype_of": subtype_of, "orbits_of": orbits_of,
            "n_subtypes": len(subtype_of), "n_orbits": len(orbit_vocab),
            "allowed": allowed, "raw": raw}


def load_pool(split: str, vocabulary: dict, max_instances: int | None = None) -> dict:
    """Instances of one split; `max_instances` caps each family deterministically.

    The heldout pass runs the encoder over every instance, and the pool holds
    hundreds of thousands per family: a fixed, seeded cap keeps the evaluation a
    measurement instead of an out-of-memory event.  The cap is symmetric across
    arms, so comparisons stay fair.
    """
    payload = np.load(HERE / "instances.npz", allow_pickle=False)
    flag = 1 if split == "heldout" else 0
    families = {}
    for family in FAMILIES:
        keep = payload[f"{family}_split"] == flag
        nodes = payload[f"{family}_nodes"][keep]
        orbits = payload[f"{family}_roles"][keep]
        subtypes = payload[f"{family}_subtype"][keep]
        if max_instances is not None and len(nodes) > max_instances:
            pick = np.random.default_rng(221).choice(len(nodes), max_instances, replace=False)
            pick.sort()
            nodes, orbits, subtypes = nodes[pick], orbits[pick], subtypes[pick]
        families[family] = (nodes.astype(np.int64), orbits.astype(np.int64),
                           subtypes.astype(np.int64))
    return families


def instance_targets(family: str, subtypes: np.ndarray, orbits: np.ndarray,
                     vocabulary: dict) -> dict:
    """Global subtype ids, global orbit ids and the per-instance orbit mask."""
    raw = vocabulary["raw"][family]
    keys = {index: key for key, index in raw.items() if key != "__other__"}
    global_subtype = np.empty(len(subtypes), dtype=np.int64)
    global_orbit = np.empty_like(orbits, dtype=np.int64)
    class_of_subtype = []
    for row, (local, orbit_row) in enumerate(zip(subtypes, orbits)):
        key = keys.get(int(local))
        if key is None:
            global_subtype[row] = -1
            global_orbit[row] = 0
            class_of_subtype.append(-1)
            continue
        subtype_id = vocabulary["subtype_of"][f"{family}:{key}"]
        global_subtype[row] = subtype_id
        orbit_ids = vocabulary["orbits_of"][subtype_id]
        global_orbit[row] = [orbit_ids[int(value)] for value in orbit_row]
        class_of_subtype.append(subtype_id)
    # per-subtype orbit mask over the global orbit vocabulary
    allowed = torch.zeros(vocabulary["n_orbits"], dtype=torch.bool)
    return {"subtype": global_subtype, "orbit": global_orbit,
            "subtype_ids": class_of_subtype}


class DirectedTypeRoleHead(nn.Module):
    """Permutation-invariant subtype readout + permutation-equivariant orbit readout."""

    def __init__(self, dim: int, hidden: int = 96, n_subtypes: int = 1,
                 n_orbits: int = 1, interaction: bool = True):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.type_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(),
                                       nn.Linear(hidden, n_subtypes))
        self.context_proj = nn.Linear(2 * hidden, hidden)
        role_in = 3 * hidden if interaction else 2 * hidden
        self.role_head = nn.Sequential(nn.Linear(role_in, hidden), nn.GELU(),
                                       nn.Linear(hidden, n_orbits))
        self.interaction = interaction

    def forward(self, node_features: torch.Tensor):
        hidden = self.phi(node_features)
        pooled = torch.cat([hidden.mean(dim=1), hidden.amax(dim=1)], dim=-1)
        type_logits = self.type_head(pooled)
        context = self.context_proj(pooled).unsqueeze(1).expand(-1, hidden.shape[1], -1)
        parts = [hidden, context, hidden * context] if self.interaction else [hidden, context]
        role_logits = self.role_head(torch.cat(parts, dim=-1))
        return type_logits, role_logits


class InstanceSampler:
    """Equal instances per family per step, from the frozen pool split."""

    def __init__(self, families: dict, per_family: int, seed: int = 221):
        self.families = {name: (torch.from_numpy(nodes), torch.from_numpy(orbits),
                                torch.from_numpy(subtypes))
                         for name, (nodes, orbits, subtypes) in families.items()}
        self.per_family = int(per_family)
        self.generator = torch.Generator().manual_seed(seed)

    def sample(self) -> dict:
        draws = {}
        for name, (nodes, orbits, subtypes) in self.families.items():
            pick = torch.randint(0, len(nodes), (self.per_family,), generator=self.generator)
            draws[name] = (nodes[pick], orbits[pick], subtypes[pick])
        return draws


def joint_loss(head: DirectedTypeRoleHead, features: dict, draws: dict,
               vocabulary: dict, orbit_masks: dict) -> tuple[torch.Tensor, dict]:
    """CE(subtype) + alpha * mean_family CE(orbit), both from the same representations."""
    type_logits, type_ids = [], []
    role_losses, diagnostics = [], {}
    for family, (nodes, orbits, subtypes) in draws.items():
        targets = orbit_masks[family]
        valid = targets["subtype"] >= 0
        if not valid.any():
            continue
        logits, role_logits = head(features[family])
        type_logits.append(logits[valid])
        type_ids.append(torch.from_numpy(targets["subtype"][valid]).to(logits.device))
        subtype_ids = torch.from_numpy(targets["subtype"]).to(role_logits.device)
        allowed_table = vocabulary["allowed"].to(role_logits.device)
        mask = allowed_table[subtype_ids.clamp(min=0)].unsqueeze(1).expand(
            -1, role_logits.shape[1], -1).clone()                # (N, k, n_orbits)
        mask[subtype_ids < 0] = True                             # nothing to mask off
        masked = role_logits.masked_fill(~mask, float("-inf"))
        target_orbits = torch.from_numpy(targets["orbit"]).to(role_logits.device)
        # Instances whose subtype is unknown (-1, the merged ``__other__`` or an
        # unsupported shuffled pattern) carry no role label either.  Feeding them
        # to the CE gave an unmasked loss against a bogus target, which is how the
        # shuffled arm reached a non-finite aux term at epoch 1.
        keep = torch.as_tensor(valid, device=role_logits.device)
        loss_role = F.cross_entropy(masked[keep].reshape(-1, masked.shape[-1]),
                                    target_orbits[keep].reshape(-1))
        role_losses.append(loss_role)
        diagnostics[family] = float(loss_role.detach())
    loss_type = F.cross_entropy(torch.cat(type_logits), torch.cat(type_ids))
    loss_role = torch.stack(role_losses).mean()
    diagnostics["type"] = float(loss_type.detach())
    diagnostics["role"] = float(loss_role.detach())
    return loss_type + loss_role, diagnostics
