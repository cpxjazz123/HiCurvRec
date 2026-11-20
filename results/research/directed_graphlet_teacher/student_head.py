"""Issue #222 B: the Student head reads quantised codes and nothing else.

Input is `q[B, k, codebook_dim]` -- the sum of the code words the real quantiser
picked for the k nodes of one instance.  Reference graphlets are only ever the KD
*target*; the head sees no structure, no label and no position id.

Two heads, same interface:

`StudentQuantisedHead`   mean/max pool + context-mixed node readout, a minimal
                         re-parameterisation of `directed_role.DirectedTypeRoleHead`;
`StudentRelationalHead`  the same plus explicit pairwise terms |q_i - q_j| and
                         q_i * q_j.  A subtype IS a relation between node codes, and
                         a pooled head cannot express one, so the relational head is
                         the positive control: if it can fit a synthetic relational
                         label but not the real one, the codes are what is missing.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class StudentQuantisedHead(nn.Module):
    """Permutation-invariant subtype readout + permutation-equivariant orbit readout."""

    def __init__(self, dim: int, hidden: int = 96, n_subtypes: int = 1,
                 n_orbits: int = 1, interaction: bool = True):
        super().__init__()
        # The codes arrive at ~0.05 per dimension; without normalising them the
        # first GELU stays in its linear regime and the whole head is a linear map
        # of the codes, which cannot express a subtype relation at all.
        self.norm = nn.LayerNorm(dim)
        self.phi = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.type_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(),
                                      nn.Linear(hidden, n_subtypes))
        self.context_proj = nn.Linear(2 * hidden, hidden)
        role_in = 3 * hidden if interaction else 2 * hidden
        self.role_head = nn.Sequential(nn.Linear(role_in, hidden), nn.GELU(),
                                       nn.Linear(hidden, n_orbits))
        self.interaction = interaction

    def forward(self, q: torch.Tensor):
        h = self.phi(self.norm(q))
        pooled = torch.cat([h.mean(1), h.amax(1)], -1)
        context = self.context_proj(pooled).unsqueeze(1).expand(-1, h.shape[1], -1)
        features = torch.cat([h, context, h * context], -1) if self.interaction \
            else torch.cat([h, context], -1)
        return self.type_head(pooled), self.role_head(features)

    def config(self) -> dict:
        return {"dim": int(self.phi[0].in_features),
                "hidden": int(self.phi[0].out_features),
                "n_subtypes": int(self.type_head[-1].out_features),
                "n_orbits": int(self.role_head[-1].out_features),
                "interaction": bool(self.interaction)}


def pairwise_terms(h: torch.Tensor) -> torch.Tensor:
    """Mean and max over pairs of |h_i - h_j| and h_i * h_j: 4*hidden wide.

    Pooling over pairs rather than concatenating them keeps the width independent
    of the instance size, so one head serves the 3-node and 4-node families alike.
    """
    differences, products = [], []
    nodes = h.shape[1]
    for i in range(nodes):
        for j in range(i + 1, nodes):
            differences.append((h[:, i] - h[:, j]).abs())
            products.append(h[:, i] * h[:, j])
    if not differences:
        return h.new_zeros(h.shape[0], 4 * h.shape[-1])
    difference = torch.stack(differences, 1)
    product = torch.stack(products, 1)
    return torch.cat([difference.mean(1), difference.amax(1),
                      product.mean(1), product.amax(1)], -1)


class StudentRelationalHead(nn.Module):
    """Pooled readout plus the pairwise terms the pooled readout cannot express."""

    def __init__(self, dim: int, hidden: int = 96, n_subtypes: int = 1, n_orbits: int = 1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.phi = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.type_head = nn.Sequential(nn.Linear(6 * hidden, hidden), nn.GELU(),
                                       nn.Linear(hidden, n_subtypes))
        self.context_proj = nn.Linear(2 * hidden, hidden)
        self.role_head = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(),
                                       nn.Linear(hidden, n_orbits))

    def forward(self, q: torch.Tensor):
        h = self.phi(self.norm(q))
        pooled = torch.cat([h.mean(1), h.amax(1)], -1)
        type_logits = self.type_head(torch.cat([pooled, pairwise_terms(h)], -1))
        context = self.context_proj(pooled).unsqueeze(1).expand(-1, h.shape[1], -1)
        orbit_logits = self.role_head(torch.cat([h, context, h * context], -1))
        return type_logits, orbit_logits

    def config(self) -> dict:
        return {"dim": int(self.phi[0].in_features),
                "hidden": int(self.phi[0].out_features),
                "n_subtypes": int(self.type_head[-1].out_features),
                "n_orbits": int(self.role_head[-1].out_features)}


def build_head(config: dict) -> StudentQuantisedHead:
    return StudentQuantisedHead(dim=int(config["dim"]), hidden=int(config["hidden"]),
                                n_subtypes=int(config["n_subtypes"]),
                                n_orbits=int(config["n_orbits"]),
                                interaction=bool(config.get("interaction", True)))


def build_relational_head(config: dict) -> StudentRelationalHead:
    return StudentRelationalHead(dim=int(config["dim"]), hidden=int(config["hidden"]),
                                 n_subtypes=int(config["n_subtypes"]),
                                 n_orbits=int(config["n_orbits"]))
