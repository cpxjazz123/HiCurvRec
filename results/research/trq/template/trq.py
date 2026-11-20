"""Issue #220: Graphlet Type+Role cross-entropy through the real RQ-VAE quantizer.

The supervised unit is one strict induced instance {v1..vk}: its Graphlet type
(K3/P3/C4/P4/K1,3) and one orbit label per node.  The head reads a representation
chosen by the arm -- the encoder's z, the true hard cumulative quantisation with the
old STE, or the hard-forward/soft-backward codeword proxy q_ST -- and the loss is

    L_pred = CE(type) + alpha * macro_mean_type[ CE(role) ]

Arms (Issue #220 section 3):
    A  no auxiliary term
    B  head on z, head trained jointly
    C  head on the hard cumulative quantisation, legacy STE (aux codebook grad 0)
    D  head on q_ST = sum_k h_ST[k] E_k with h_ST = onehot(h) + p - stopgrad(p)
    E  same q as D with a head pretrained on z and frozen
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from graphlet_roles import (ORBIT_TO_GLOBAL, REPO, ROLE_MASK, TYPE_INDEX, TYPES, InstanceSampler,
                            TypeRoleHead, load_instances, role_loss, type_loss)

ARM_NO_AUX = "A"
ARM_JOINT_Z = "B"
ARM_JOINT_Q_STE = "C"
ARM_JOINT_Q_CODEBOOK = "D"
ARM_FROZEN_HEAD_Q = "E"
ARMS = (ARM_NO_AUX, ARM_JOINT_Z, ARM_JOINT_Q_STE, ARM_JOINT_Q_CODEBOOK, ARM_FROZEN_HEAD_Q)
ARM_VARIANT = "A"

#: roles are averaged inside a type and then across types, so a type with two
#: orbits cannot outweigh one with a single role (Issue #220 section 1)
ROLE_WEIGHT = 1.0
#: instances per type per optimiser step
PER_TYPE = 64
#: encoder-side auxiliary gradient budget, calibrated over the first 20 steps
AUX_SHARE = 0.10
CALIBRATION_STEPS = 20

#: pre-registered health bounds (Issue #220 section 2.5); crossing one stops the arm
SPACING_FLOOR = 0.05
MAX_BUCKET_CEILING = 0.90
COLLISION_CEILING = 0.98


def instance_draw(sampler: InstanceSampler, generator: torch.Generator) -> dict:
    return sampler.sample(generator)


def hard_cumulative(tokens: torch.Tensor, codebooks: list) -> torch.Tensor:
    """sum_{k<=l} E_k[h_k]: the representation the SID prefix is built from."""
    total = None
    for level, book in enumerate(codebooks):
        picked = book[tokens[:, level]]
        total = picked if total is None else total + picked
    return total


def legacy_ste(z: torch.Tensor, hard_value: torch.Tensor) -> torch.Tensor:
    """z forward-value replaced by the hard quantisation, identity backward to z."""
    return z + (hard_value - z).detach()


def ste_onehot(hard_index: torch.Tensor, soft: torch.Tensor, n_codes: int) -> torch.Tensor:
    """h_ST = onehot(h) + p - stopgrad(p): hard forward, soft backward."""
    onehot = F.one_hot(hard_index, n_codes).to(soft.dtype)
    return onehot + (soft - soft.detach())


def q_st(soft_probs: list, hard_indices: torch.Tensor, codebooks: list,
         depth: int = 3) -> torch.Tensor:
    """Cumulative q_ST over the first `depth` levels (gradients to encoder and codebook)."""
    total = None
    for level in range(min(depth, len(codebooks))):
        book = codebooks[level]
        weight = ste_onehot(hard_indices[:, level], soft_probs[level], book.shape[0])
        picked = weight @ book
        total = picked if total is None else total + picked
    return total


def joint_loss(head: TypeRoleHead, features: dict, roles: dict, types: dict,
               masks: dict) -> tuple[torch.Tensor, dict]:
    """CE(type) + alpha * mean_type CE(role) over the per-type instance tensors."""
    type_logits, type_ids = [], []
    role_losses = []
    diagnostics = {}
    for name in features:
        logits, role_logits = head(features[name])
        type_logits.append(logits)
        type_ids.append(types[name])
        loss_role = role_loss(role_logits, roles[name], masks[name])
        role_losses.append(loss_role)
        diagnostics[name] = float(loss_role.detach())
    loss_type = type_loss(torch.cat(type_logits), torch.cat(type_ids))
    loss_role = torch.stack(role_losses).mean()
    total = loss_type + ROLE_WEIGHT * loss_role
    diagnostics["type"] = float(loss_type.detach())
    diagnostics["role"] = float(loss_role.detach())
    return total, diagnostics


def gather(embeddings: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Rows of a host-side matrix for indices that may live on the GPU."""
    return embeddings[index.detach().to("cpu")]


def level_softmax(layer, residual: torch.Tensor, temperature: float) -> torch.Tensor:
    latent = residual.reshape(-1, layer.dim)
    return F.softmax(-layer._distances(latent) / temperature, dim=-1)


def measure_temperature(layer, residual: torch.Tensor, quantile: float = 0.5) -> float:
    """The level's own quantisation scale, measured once and never tuned."""
    with torch.no_grad():
        latent = residual.reshape(-1, layer.dim)
        nearest = layer._distances(latent).min(dim=-1).values
        scale = float(torch.quantile(nearest, quantile))
    return max(scale, 1e-6)


def grad_norm(gradients) -> float:
    total = 0.0
    for grad in gradients:
        if grad is not None:
            total += float(grad.detach().pow(2).sum())
    return float(np.sqrt(total))


def calibrate_weight(reference: float, auxiliary: float, share: float = AUX_SHARE) -> float:
    if not np.isfinite(reference) or not np.isfinite(auxiliary) or auxiliary <= 0:
        return 0.0
    return float(share * reference / auxiliary)


def codeword_spacing(codebook: torch.Tensor) -> float:
    """Median pairwise codeword distance, in float64."""
    with torch.no_grad():
        codes = codebook.detach().double()
        distances = torch.cdist(codes, codes)
        n = codes.shape[0]
        upper = distances[torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)]
        return float(upper.median())


def health_verdict(codes: np.ndarray, codebook: torch.Tensor, initial_spacing: float) -> dict:
    """Pre-registered collapse/balance monitors (Issue #220 section 2.5)."""
    spacing = codeword_spacing(codebook)
    ratio = spacing / initial_spacing if initial_spacing > 0 else float("inf")
    counts = np.bincount(codes[:, 0], minlength=int(codebook.shape[0])).astype(np.float64)
    share = counts / max(len(codes), 1)
    collision = 1.0 - len(np.unique(codes, axis=0)) / max(len(codes), 1)
    verdict = {"spacing": spacing, "spacing_ratio": ratio,
               "max_bucket_l1": float(share.max()), "l1_used": int((counts > 0).sum()),
               "collision": float(collision)}
    problems = []
    if ratio < SPACING_FLOOR:
        problems.append(f"codeword spacing fell to {ratio:.4f}x of the initial spacing")
    if verdict["max_bucket_l1"] > MAX_BUCKET_CEILING:
        problems.append(f"L1 max bucket is {verdict['max_bucket_l1']:.3f} of the items")
    if verdict["collision"] > COLLISION_CEILING:
        problems.append(f"three-code collision is {verdict['collision']:.3f}")
    if problems:
        raise RuntimeError("trq health stop: " + "; ".join(problems))
    return verdict


def macro_f1(predictions: torch.Tensor, targets: torch.Tensor, n_classes: int) -> dict:
    """Macro-F1 and per-class recall over the classes that occur in the targets."""
    per_class = {}
    for label in range(n_classes):
        support = int((targets == label).sum())
        if support == 0:
            continue
        predicted = predictions == label
        true = targets == label
        tp = int((predicted & true).sum())
        precision = tp / max(int(predicted.sum()), 1)
        recall = tp / support
        per_class[label] = {"precision": precision, "recall": recall,
                            "f1": 0.0 if precision + recall == 0 else
                            2 * precision * recall / (precision + recall), "support": support}
    macro = sum(row["f1"] for row in per_class.values()) / max(len(per_class), 1)
    return {"macro_f1": macro, "per_class": per_class, "classes": len(per_class)}


@torch.no_grad()
def heldout_scores(head, encoder, embeddings, device, instances, temperature=0.0) -> dict:
    """Type and role macro-F1 on a heldout split for the arm's own representation."""
    scores = {}
    role_pool = {}
    type_predictions, type_targets = [], []
    for name, (nodes, orbits) in instances.items():
        if len(nodes) == 0:
            continue
        samples, width = nodes.shape
        index = torch.from_numpy(nodes.reshape(-1)).long()
        features = encoder(gather(embeddings, index).to(device)).view(samples, width, -1)
        type_logits, role_logits = head(features)
        mask = ROLE_MASK[TYPE_INDEX[name]].unsqueeze(0).expand(samples, -1).to(device)
        role_ids = ORBIT_TO_GLOBAL[name][torch.from_numpy(orbits.astype(np.int64))].to(device)
        masked = role_logits.masked_fill(~mask.unsqueeze(1), float("-inf"))
        predictions = masked.argmax(-1)
        type_predictions.append(type_logits.argmax(-1))
        type_targets.append(torch.full((samples,), TYPE_INDEX[name], dtype=torch.long))
        role_pool[name] = (predictions.cpu(), role_ids.cpu())
    type_ids = torch.cat(type_targets)
    scores["type"] = macro_f1(torch.cat(type_predictions).cpu(), type_ids, len(TYPES))
    roles = {}
    for name, (pred, target) in role_pool.items():
        n_classes = int(max(int(pred.max()), int(target.max()))) + 1
        roles[name] = macro_f1(pred.reshape(-1), target.reshape(-1), n_classes)
    # macro over types, so a single-orbit type (K3/C4) cannot inflate the role score
    # macro over the multi-orbit types only: K3/C4 have a single orbit, where a
    # constant prediction is always right and would inflate the role score
    multi = {name: row for name, row in roles.items() if row["classes"] > 1}
    scores["role"] = {"macro_over_types": (sum(row["macro_f1"] for row in multi.values())
                                           / max(len(multi), 1)),
                      "macro_over_all_types": sum(row["macro_f1"] for row in roles.values())
                                              / max(len(roles), 1),
                      "per_type": roles,
                      "single_orbit_types": sorted(name for name, row in roles.items()
                                                   if row["classes"] <= 1)}
    return scores
