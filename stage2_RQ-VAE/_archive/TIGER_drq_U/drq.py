"""Issue #221: directed Graphlet subtype + directed orbit supervision through the real quantizer.

The arm names, the quantiser plumbing and the health bounds are the ones #220's
verified template used; what changes is the label source: every instance is a set
of nodes whose DIRECTED induced subgraph is classified into a canonical subtype,
and each node's role is its orbit under that subtype's automorphism group.

Arms (Issue #221 section 4):
    D0  no auxiliary term
    D1  head on encoder z, trained jointly
    D2  head on the hard cumulative quantisation, legacy STE (aux codebook grad 0)
    D3  head on q_ST (hard forward / soft backward), codebook receives the gradient
    D4  same q as D3 with a head pretrained on z and frozen
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from directed_graphlets import canonical, classify, load_adjacency  # noqa: F401
from directed_role import (FAMILIES, DirectedTypeRoleHead, InstanceSampler,  # noqa: F401
                           instance_targets, joint_loss as directed_joint_loss,
                           load_pool, load_vocabulary)

ARM_NO_AUX = "D0"
ARM_JOINT_Z = "D1"
ARM_JOINT_Q_STE = "D2"
ARM_JOINT_Q_CODEBOOK = "D3"
ARM_FROZEN_HEAD_Q = "D4"
ARMS = (ARM_NO_AUX, ARM_JOINT_Z, ARM_JOINT_Q_STE, ARM_JOINT_Q_CODEBOOK, ARM_FROZEN_HEAD_Q)
ARM_VARIANT = "D1"
#: What the head is asked to predict (Issue #221 section 5 ablations):
#:   directed   -- the true canonical directed subtype and directed orbits
#:   undirected -- the undirected skeleton and #217's undirected orbit labels
#:   shuffled   -- the same instance, edges re-oriented at random (seeded)
LABEL_MODE = "undirected"

PER_TYPE = 16
AUX_SHARE = 0.10
CALIBRATION_STEPS = 20

SPACING_FLOOR = 0.05
MAX_BUCKET_CEILING = 0.90
COLLISION_CEILING = 0.98


def vocabulary():
    return load_vocabulary(LABEL_MODE)


def make_sampler(split: str, seed: int, max_instances: int | None = None):
    vocab = load_vocabulary(LABEL_MODE)
    families = load_pool(split, vocab, max_instances)
    sampler = InstanceSampler(families, PER_TYPE, seed=seed)
    return sampler, vocab


def instance_draw(sampler, generator):
    del generator                      # the sampler owns its seeded generator
    return sampler.sample()


def joint_loss(head, features, draws, vocab, masks):
    return directed_joint_loss(head, features, draws, vocab, masks)


def targets_for(draws, vocab) -> dict:
    """Labels for the arm's LABEL_MODE: directed, undirected or shuffled directions."""
    if LABEL_MODE == "directed":
        return {family: instance_targets(family, subtypes.numpy(), orbits.numpy(), vocab)
                for family, (_nodes, orbits, subtypes) in draws.items()}
    if LABEL_MODE == "undirected":
        # Recompute the orbit from the DRAWN instance's symmetrised adjacency.  The
        # first version sliced the first N rows of the whole undirected pool, which
        # is the wrong rows for every draw: the two pools are row-aligned with each
        # other, but a random draw is not a prefix of either.
        adjacency = load_adjacency()
        out = {}
        for family, (nodes, _orbits, _subtypes) in draws.items():
            subtype = int(vocab["subtype_of"][family])
            rows = []
            for instance in nodes.numpy():
                matrix = np.asarray(adjacency[instance][:, instance].todense(), dtype=bool)
                _key, orbit_ids, _canonical = canonical(matrix | matrix.T)
                rows.append(orbit_ids)
            out[family] = {"subtype": np.full(len(nodes), subtype, dtype=np.int64),
                           "orbit": np.asarray(rows, dtype=np.int64)}
        return out
    # shuffled: the instance's own induced matrix with its edges re-oriented
    adjacency = load_adjacency()
    out = {}
    for family, (nodes, _orbits, _subtypes) in draws.items():
        keys, role_rows = [], []
        for instance in nodes.numpy():
            matrix = np.asarray(adjacency[instance][:, instance].todense(), dtype=bool)
            shuffled = _shuffle_directions(matrix, instance)
            skeleton, key, orbit_ids = classify(shuffled)
            local = vocab["raw"][family].get(key, -1) if skeleton else -1
            keys.append(local)
            role_rows.append(orbit_ids)
        # Local (subtype, orbit) pairs must go through instance_targets: the role
        # head predicts over the GLOBAL orbit vocabulary, so passing local 0..3 ids
        # as targets made the CE meaningless and the masked term non-finite.
        out[family] = instance_targets(family, np.asarray(keys, dtype=np.int64),
                                       np.asarray(role_rows, dtype=np.int64), vocab)
    return out


def _undirected_pool() -> str:
    from directed_role import HERE, UNDIRECTED_POOL
    return str(HERE.parents[2] / UNDIRECTED_POOL)


def _shuffle_directions(matrix: np.ndarray, instance, seed: int = 221) -> np.ndarray:
    """Re-orient every undirected edge at random; edge count and skeleton stay put."""
    import hashlib

    key = ",".join(str(int(v)) for v in sorted(int(x) for x in instance))
    rng = np.random.default_rng(int.from_bytes(
        hashlib.blake2b(key.encode(), digest_size=8).digest(), "little") ^ seed)
    shuffled = np.zeros_like(matrix)
    for left in range(matrix.shape[0]):
        for right in range(left + 1, matrix.shape[0]):
            forward, backward = bool(matrix[left, right]), bool(matrix[right, left])
            if not (forward or backward):
                continue
            if forward and backward:
                # A reciprocal pair has no direction to remove; keep both arrows so
                # the control changes direction and nothing else.
                shuffled[left, right] = forward
                shuffled[right, left] = backward
            elif rng.random() < 0.5:
                shuffled[right, left] = True
            else:
                shuffled[left, right] = True
    return shuffled


def hard_cumulative(tokens: torch.Tensor, codebooks: list) -> torch.Tensor:
    total = None
    for level, book in enumerate(codebooks):
        picked = book[tokens[:, level]]
        total = picked if total is None else total + picked
    return total


def legacy_ste(z: torch.Tensor, hard_value: torch.Tensor) -> torch.Tensor:
    """The hard value forward, the encoder identity backward (codebook detached)."""
    return z + (hard_value - z).detach()


def ste_onehot(hard_index: torch.Tensor, soft: torch.Tensor, n_codes: int) -> torch.Tensor:
    onehot = F.one_hot(hard_index, n_codes).to(soft.dtype)
    return onehot + (soft - soft.detach())


def q_st(soft_probs: list, hard_indices: torch.Tensor, codebooks: list,
         depth: int = 3) -> torch.Tensor:
    total = None
    for level in range(min(depth, len(codebooks))):
        book = codebooks[level]
        weight = ste_onehot(hard_indices[:, level], soft_probs[level], book.shape[0])
        picked = weight @ book
        total = picked if total is None else total + picked
    return total


def gather(embeddings: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return embeddings[index.detach().to("cpu")]


def level_softmax(layer, residual: torch.Tensor, temperature: float) -> torch.Tensor:
    latent = residual.reshape(-1, layer.dim)
    return F.softmax(-layer._distances(latent) / temperature, dim=-1)


def measure_temperature(layer, residual: torch.Tensor, quantile: float = 0.5) -> float:
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
    with torch.no_grad():
        codes = codebook.detach().double()
        distances = torch.cdist(codes, codes)
        n = codes.shape[0]
        upper = distances[torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)]
        return float(upper.median())


def health_verdict(codes: np.ndarray, codebook: torch.Tensor, initial_spacing: float) -> dict:
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
        # A stopped arm must leave evidence of which bound it crossed: the
        # traceback is swallowed by the detached launcher, so the verdict goes to
        # disk before the raise.
        import json
        from pathlib import Path as _Path
        report = {**verdict, "problems": problems}
        for parent in _Path(__file__).resolve().parents:
            if (parent / "stage2_RQ-VAE").is_dir() and (parent / "results").is_dir():
                arm = _Path(__file__).resolve().parent.name
                out = parent / "results" / "stage2_RQ-VAE" / arm / "health_stop.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(report, indent=1))
                break
        raise RuntimeError("drq health stop: " + "; ".join(problems))
    return verdict


def macro_f1(predictions: torch.Tensor, targets: torch.Tensor, n_classes: int) -> dict:
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
def heldout_scores(head, encoder, embeddings, device, families, vocab) -> dict:
    """Subtype macro-F1 and orbit macro-F1 (per family, then over families)."""
    subtype_predictions, subtype_targets = [], []
    role_rows = {}
    for family, (nodes, orbits, subtypes) in families.items():
        if len(nodes) == 0:
            continue
        count, width = nodes.shape
        index = torch.from_numpy(nodes.reshape(-1)).long()
        features = encoder(gather(embeddings, index).to(device)).view(count, width, -1)
        type_logits, role_logits = head(features)
        # Evaluate in the arm's OWN label space: the undirected and shuffled arms
        # must not be scored against the directed labels they never saw (and the
        # undirected vocabulary cannot even key them: it raised KeyError on
        # 'K3:000100110' after the export, which is why these arms had no
        # trq_metrics.json).
        draws = {family: (torch.from_numpy(nodes), torch.from_numpy(orbits),
                          torch.from_numpy(subtypes))}
        targets = targets_for(draws, vocab)[family]
        valid = targets["subtype"] >= 0
        subtype_predictions.append(type_logits[torch.from_numpy(valid)].argmax(-1).cpu())
        subtype_targets.append(torch.from_numpy(targets["subtype"][valid]))
        subtype_ids = torch.from_numpy(targets["subtype"]).to(role_logits.device)
        allowed_table = vocab["allowed"].to(role_logits.device)
        mask = allowed_table[subtype_ids.clamp(min=0)].unsqueeze(1).expand(
            -1, role_logits.shape[1], -1).clone()
        mask[subtype_ids < 0] = True
        masked = role_logits.masked_fill(~mask, float("-inf"))
        keep = torch.from_numpy(valid)
        predictions = masked.argmax(-1).cpu()[keep]
        role_rows[family] = (predictions.reshape(-1),
                             torch.from_numpy(targets["orbit"])[keep].reshape(-1))
    scores = {"subtype": macro_f1(torch.cat(subtype_predictions), torch.cat(subtype_targets),
                                  vocab["n_subtypes"])}
    per_family = {}
    for family, (pred, target) in role_rows.items():
        n_classes = int(max(int(pred.max()), int(target.max()))) + 1
        per_family[family] = macro_f1(pred, target, n_classes)
    scores["role"] = {"macro_over_families": sum(row["macro_f1"] for row in per_family.values())
                                             / max(len(per_family), 1),
                      "per_family": per_family}
    return scores
