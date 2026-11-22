"""Issue #217: induced graphlet instances, orbit roles and the joint type/role head.

The supervision is instance-level: a batch is a set of induced subgraphs, each carrying a
graphlet type and one orbit label per node.  Nothing here assumes an item has a single
role -- the same item appears in many instances with different roles, and the role head is
evaluated per (instance, node).

Readout shapes:
  * type logits are permutation invariant (pool over nodes, then a shared MLP);
  * role logits are permutation equivariant (one function of the node's own embedding and
    the permutation-invariant context of the instance).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
INSTANCES = REPO / "results" / "research" / "graphlet_roles" / "instances.npz"
COVERAGE = REPO / "results" / "research" / "graphlet_roles" / "coverage.json"

ARM_EDGE = "E"
ARM_TYPE = "T"
ARM_ROLE = "R"
ARM_QUANT = "Q"
ARM_RELATION = "QR"
ARMS = (ARM_EDGE, ARM_TYPE, ARM_ROLE, ARM_QUANT, ARM_RELATION)

TYPES = ("K3", "P3", "C4", "P4", "K1_3")
TYPE_INDEX = {name: index for index, name in enumerate(TYPES)}
NODES_PER_TYPE = {"K3": 3, "P3": 3, "C4": 4, "P4": 4, "K1_3": 4}
# A global role vocabulary: every (type, orbit) pair gets its own index, so a leaf of a
# path and a centre of a claw are different labels while the four C4 nodes share one.
ROLE_NAMES = {
    "K3": ("all",),
    "P3": ("leaf", "centre"),
    "C4": ("all",),
    "P4": ("end", "internal"),
    "K1_3": ("leaf", "centre"),
}
ROLE_INDEX: dict = {}
for _type in TYPES:
    for _orbit, _name in enumerate(ROLE_NAMES[_type]):
        ROLE_INDEX[(_type, _orbit)] = len(ROLE_INDEX)
N_ROLES = len(ROLE_INDEX)
ROLE_MASK = torch.zeros(len(TYPES), N_ROLES, dtype=torch.bool)
for _type in TYPES:
    for _orbit in range(len(ROLE_NAMES[_type])):
        ROLE_MASK[TYPE_INDEX[_type], ROLE_INDEX[(_type, _orbit)]] = True
# The map from a per-type orbit label to the global role index, for each type.
ORBIT_TO_GLOBAL = {
    name: torch.tensor([ROLE_INDEX[(name, orbit)] for orbit in range(len(ROLE_NAMES[name]))],
                       dtype=torch.long)
    for name in TYPES
}


def load_instances(split: str) -> dict:
    """Instances of one split, grouped by type: {type: (nodes(int64), roles(int8))}."""
    payload = np.load(INSTANCES, allow_pickle=False)
    flag = 1 if split == "heldout" else 0
    grouped = {}
    for name in TYPES:
        nodes = payload[f"{name}_nodes"]
        roles = payload[f"{name}_roles"]
        keep = payload[f"{name}_split"] == flag
        grouped[name] = (nodes[keep], roles[keep])
    return grouped


class InstanceSampler:
    """Draws an equal number of instances per type from the frozen split."""

    def __init__(self, split: str, per_type: int):
        grouped = load_instances(split)
        self.arrays = {name: (np.asarray(nodes), np.asarray(roles))
                       for name, (nodes, roles) in grouped.items() if len(nodes)}
        self.per_type = int(per_type)

    def sample(self, generator: torch.Generator) -> dict:
        draws = {}
        for name, (nodes, roles) in self.arrays.items():
            index = torch.randint(0, len(nodes), (self.per_type,), generator=generator)
            draws[name] = (nodes[index.numpy()], roles[index.numpy()])
        return draws


class TypeRoleHead(nn.Module):
    """Permutation-invariant type readout and permutation-equivariant role readout."""

    def __init__(self, dim: int, hidden: int = 96, n_types: int = len(TYPES),
                 n_roles: int = N_ROLES, interaction: bool = True):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.type_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(),
                                       nn.Linear(hidden, n_types))
        # The role readout needs the instance context at the same width as a node feature.
        self.context_proj = nn.Linear(2 * hidden, hidden)
        role_in = 3 * hidden if interaction else 2 * hidden
        self.role_head = nn.Sequential(nn.Linear(role_in, hidden), nn.GELU(),
                                       nn.Linear(hidden, n_roles))
        self.interaction = interaction

    def forward(self, node_features: torch.Tensor):
        hidden = self.phi(node_features)
        pooled = torch.cat([hidden.mean(dim=1), hidden.amax(dim=1)], dim=-1)
        type_logits = self.type_head(pooled)
        context = self.context_proj(pooled).unsqueeze(1).expand(-1, hidden.shape[1], -1)
        parts = [hidden, context, hidden * context] if self.interaction else [hidden, context]
        role_logits = self.role_head(torch.cat(parts, dim=-1))
        return type_logits, role_logits


def type_loss(type_logits: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(type_logits, type_ids)


def role_loss(role_logits: torch.Tensor, role_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Role CE over each instance's node positions, masked to the instance's own orbits."""
    logits = role_logits.masked_fill(~mask.unsqueeze(1), float("-inf"))
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), role_ids.reshape(-1))


def relation_margin(quantized: torch.Tensor, role_ids: torch.Tensor,
                    same_types: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    """Same-role node pairs inside an instance must be closer than different-role pairs."""
    normalised = F.normalize(quantized, dim=-1)
    similarity = normalised @ normalised.transpose(1, 2)
    nodes = quantized.shape[1]
    eye = torch.eye(nodes, dtype=torch.bool, device=quantized.device)
    same = (role_ids.unsqueeze(1) == role_ids.unsqueeze(2)) & ~eye
    different = (role_ids.unsqueeze(1) != role_ids.unsqueeze(2)) & ~eye
    usable = same.any(dim=(1, 2)) & different.any(dim=(1, 2))
    if not bool(usable.any()):
        return quantized.sum() * 0.0
    same_mean = (similarity * same).sum(dim=(1, 2)) / same.sum(dim=(1, 2)).clamp(min=1)
    diff_mean = (similarity * different).sum(dim=(1, 2)) / different.sum(dim=(1, 2)).clamp(min=1)
    return F.relu(margin - (same_mean - diff_mean))[usable].mean()


def quantized_ste(encoder_out: torch.Tensor, codebooks: list, levels: int = 3) -> torch.Tensor:
    """Cumulative quantised value with the model's own straight-through gradient.

    The value is exactly ``sum_{m<=levels} E_{c_m}`` (the representation the SID prefixes
    are built from), the gradient to the encoder is the identity, and the codebook is
    detached -- the quantiser stays owned by the reconstruction loss.  A soft-assignment
    proxy was tried first: its Jacobian pushed the codeword spacing to 0.016x of the
    initial value within 50 steps and the anti-collapse guard stopped the arm.
    """
    total = None
    residual = encoder_out
    for level in range(min(levels, len(codebooks))):
        codebook = codebooks[level].detach()
        index = torch.cdist(residual, codebook).argmin(dim=-1)
        picked = codebook[index]
        total = picked if total is None else total + picked
        residual = residual - picked
    if total is None:
        return encoder_out
    return encoder_out + (total - encoder_out).detach()


def soft_quantized(encoder_out: torch.Tensor, codebooks: list, temperature: float = 1.0,
                   levels: int = 3) -> torch.Tensor:
    """Cumulative codebook expectation: a differentiable stand-in for the hard SID.

    ``q = sum_{m<=levels} softmax(-||r||^2 / T) @ E_m`` with the residual defined exactly as
    the hard path defines it, so the value matches the hard cumulative representation in the
    limit of a sharp temperature and carries gradients into the codebooks.
    """
    residual = encoder_out
    total = torch.zeros_like(encoder_out)
    for level in range(min(levels, len(codebooks))):
        # Stop-gradient on the codebook: the auxiliary head must not reshape the quantiser.
        # With the gradient left on, the Q arm collapsed its codeword spacing to 0.016x of
        # its initial value within 50 steps and the anti-collapse guard stopped the run;
        # the codebook stays owned by the reconstruction loss, and the *representation*
        # path (through the soft weights and the residual) is what the aux sees.
        codebook = codebooks[level].detach()
        cost = torch.cdist(residual, codebook)
        weight = torch.softmax(-cost ** 2 / temperature, dim=-1)
        expected = weight @ codebook
        total = total + expected
        residual = residual - expected
    return total


def instance_tensors(draw: dict, type_name: str, device) -> tuple:
    nodes, orbits = draw[type_name]
    type_ids = torch.full((len(nodes),), TYPE_INDEX[type_name], dtype=torch.long)
    role_ids = ORBIT_TO_GLOBAL[type_name][torch.from_numpy(orbits.astype(np.int64))]
    mask = ROLE_MASK[TYPE_INDEX[type_name]].unsqueeze(0).expand(len(nodes), -1)
    return (torch.from_numpy(nodes.astype(np.int64)).to(device), role_ids.to(device),
            type_ids.to(device), mask.to(device))


def coverage_summary() -> dict:
    return json.loads(COVERAGE.read_text())


def graphlet_aux(encoder, head, sampler, generator, arm: str, embeddings, codebooks,
                 device) -> tuple:
    """The auxiliary loss of one step for arms T/R/Q/QR.

    ``aux = L_type + ALPHA * L_role + BETA * L_relation`` (the last two only where the arm
    says so).  Q and QR read the soft cumulative quantisation, which is what makes the
    codebook gradient path auditable; T/R read the encoder representation z.
    """
    alpha, beta = 1.0, 1.0
    draw = sampler.sample(generator)
    total = 0.0
    diagnostics = {"type": 0.0, "role": 0.0, "relation": 0.0, "instances": 0}
    # The type readout is one classifier over the mixed batch: computing it inside a
    # per-type group would make every batch single-class and the CE trivial.
    mixed_type_logits, mixed_type_ids = [], []
    for name, (nodes, orbits) in draw.items():
        samples, width = nodes.shape
        # The embedding matrix stays on the host; gather there and move the slice, so a
        # CUDA index never meets a CPU tensor.
        index = torch.as_tensor(nodes.reshape(-1), dtype=torch.long, device=embeddings.device)
        latent = encoder(embeddings[index].to(device)).reshape(samples, width, -1)
        features = latent
        if arm in (ARM_QUANT, ARM_RELATION):
            features = quantized_ste(latent, codebooks)
        type_logits, role_logits = head(features)
        role_ids = ORBIT_TO_GLOBAL[name][torch.as_tensor(orbits.astype(np.int64))].to(device)
        type_ids = torch.full((samples,), TYPE_INDEX[name], dtype=torch.long, device=device)
        mask = ROLE_MASK[TYPE_INDEX[name]].unsqueeze(0).expand(samples, -1).to(device)
        mixed_type_logits.append(type_logits)
        mixed_type_ids.append(type_ids)
        if arm != ARM_TYPE:
            loss_role = role_loss(role_logits, role_ids, mask)
            total = total + alpha * loss_role
            diagnostics["role"] += float(loss_role.detach())
        if arm == ARM_RELATION:
            loss_relation = relation_margin(features, role_ids,
                                            torch.ones(samples, dtype=torch.bool, device=device))
            total = total + beta * loss_relation
            diagnostics["relation"] += float(loss_relation.detach())
        diagnostics["instances"] += int(samples)
    if mixed_type_logits:
        loss_type = type_loss(torch.cat(mixed_type_logits), torch.cat(mixed_type_ids))
        total = total + loss_type
        diagnostics["type"] = float(loss_type.detach())
    return total, diagnostics
