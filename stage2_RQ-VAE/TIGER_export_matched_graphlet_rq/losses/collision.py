"""Issue #233 §2: the collision term, and the gradient budgets that bound both auxiliaries.

Why this exists -- measured on #232's own run (`results/research/issue233/diagnose232.json`):

* L1 used all 256 codes yet its largest bucket held **10,700 of 24,587 items** (TIGER: 718).
* That pushed the pressure onto L2: its largest bucket went from 85 to **4,862**, and its
  conditional entropy fell from 4.979 to 3.524 bits.
* L3's conditional entropy actually ROSE (2.203 -> 2.923), i.e. L3 was still trying; the level that
  lost discrimination was L2, and the cause was an unbalanced L1.

So the collision term has two distinct jobs, and they act on different levels:

1. **L1 capacity** (`capacity_term`): a differentiable Balancer-style penalty on the L1 assignment
   margins. It never touches the codewords -- only the encoder -- and it is what stops one bucket
   from swallowing 43% of the catalogue.
2. **L3 duplicate penalty** (`duplicate_term`): for two DIFFERENT items whose first two codes agree
   (a collision candidate), penalise the probability that their third code also agrees. The
   gradient is therefore on L3, which is what issue §2 asks for ("学习信号应优先改变 L3,避免为了
   降碰撞拆散已正确的 L1/L2 前缀").

Both are bounded by a **gradient budget**, not by a fixed weight: `budget_scale` measures the
reconstruction gradient's norm and rescales the auxiliary so its own norm is at most
`budget_fraction` of it. Issue §2 forbids `RMS=1` normalisation and forbids amplifying a tiny
gradient: the scale is capped at 1.0, so a term smaller than its budget is left alone.
"""
from __future__ import annotations

import torch

#: Fraction of the reconstruction gradient an auxiliary term may carry (issue §2: "不超过原
#: RQ-VAE encoder 梯度的10%"). Pre-registered, not tuned.
GRAPHLET_GRAD_BUDGET = 0.10
COLLISION_GRAD_BUDGET = 0.10
#: A term whose own norm is below its budget is NOT amplified: the scale is capped here.
MAX_UPSCALE = 1.0


def encoder_grad_norm(model) -> float:
    total = sum(float(p.grad.detach().pow(2).sum())
                for p in model.encoder.parameters() if p.grad is not None)
    return total ** 0.5


def budget_scale(aux_norm: float, reference_norm: float, fraction: float,
                 cap: float = MAX_UPSCALE) -> float:
    """How much to multiply an auxiliary term so its gradient is `fraction` of the reference.

    `1.0` when the term is already at or below its budget -- shrinking a correct term to a fixed
    ratio would make the budget a weight, which is exactly what issue §2 rules out. `cap` bounds
    the upward direction so a vanishing gradient is not amplified into one.
    """
    if aux_norm <= 0.0:
        return 0.0
    if reference_norm <= 0.0:
        return 0.0
    target = fraction * reference_norm
    if aux_norm <= target:
        return 1.0
    return min(float(target) / float(aux_norm), cap)


def capacity_term(probs, roles_l1, n_items: int) -> torch.Tensor:
    """Balancer penalty on L1: `mean_k (load_k / mean_load - 1)^2`.

    `probs` is the level-1 assignment probability `(N, K)` for the supervised draw; `roles_l1`
    selects the rows the structure batch covers. The load is the summed probability mass per
    codeword, so the gradient reaches the encoder through `probs` and never the codewords.
    """
    # the index may arrive on CPU (numpy -> torch) while `probs` lives on a CUDA device
    selected = probs.index_select(0, roles_l1.to(probs.device))
    load = selected.sum(0)
    mean = load.sum() / float(load.shape[0])
    if float(mean) <= 0.0:
        return probs.sum() * 0.0
    return ((load / mean) - 1.0).pow(2).mean()


def duplicate_term(probs_l3, pair_a, pair_b) -> torch.Tensor:
    """`sum_k p_a[k] p_b[k]` for collision candidates: the probability they share the L3 code.

    `pair_a`/`pair_b` are rows of the draw whose L1 and L2 codes already agree, so the only
    remaining way to distinguish them is L3. Only L3's probability enters, so the gradient the
    term produces lands on the third level's assignment.
    """
    return (probs_l3[pair_a] * probs_l3[pair_b]).sum(-1).mean()


def candidate_pairs_from_prefix(prefix_keys: torch.Tensor, max_pairs: int, generator=None):
    """Sampled `(left, right)` row pairs that share the same L1+L2 prefix, across ALL items.

    Issue §2 requires the collision signal to cover the GLOBAL duplicate population, not only the
    three pairs one graphlet instance contains. Measured on the frozen products: TIGER has
    131,140 such pairs, #232's run had 16,921,977 -- sampling is mandatory, and the sample is
    taken per prefix so a large bucket is not drowned by many small ones.
    """
    prefix_keys = prefix_keys.cpu()
    order = torch.argsort(prefix_keys)
    sorted_keys = prefix_keys[order]
    boundary = torch.nonzero(sorted_keys[1:] != sorted_keys[:-1]).reshape(-1) + 1
    starts = torch.cat([torch.zeros(1, dtype=torch.long), boundary])
    ends = torch.cat([boundary, torch.tensor([len(order)], dtype=torch.long)])
    left_parts, right_parts = [], []
    budget = int(max_pairs)
    for start, end in zip(starts.tolist(), ends.tolist()):
        size = end - start
        if size < 2 or budget <= 0:
            continue
        take = min(int(size * (size - 1) // 2), max(1, budget // max(1, len(starts))))
        a = torch.randint(0, size, (take,), generator=generator)
        b = torch.randint(0, size, (take,), generator=generator)
        keep = a != b
        rows = order[start:end]
        left_parts.append(rows[a[keep]])
        right_parts.append(rows[b[keep]])
        budget -= int(keep.sum())
    if not left_parts:
        return None
    return torch.cat(left_parts).long().cpu(), torch.cat(right_parts).long().cpu()


def collision_candidates(tokens: torch.Tensor):
    """Rows whose first two codes agree -- the population §2 calls a collision candidate.

    Computed from the model's OWN training assignment (issue §2: "碰撞监督采用模型自身训练分配,
    不读取 held-out 行为标签"). The caller re-derives them every step, so the set tracks the model.
    """
    prefix = tokens[:, :2]
    keys = prefix[:, 0].long() * 100000 + prefix[:, 1].long()
    order = torch.argsort(keys)
    sorted_keys = keys[order]
    same = sorted_keys[1:] == sorted_keys[:-1]
    if not bool(same.any()):
        return None
    left = order[:-1][same]
    right = order[1:][same]
    return left.long(), right.long()


__all__ = ["capacity_term", "duplicate_term", "collision_candidates",
           "candidate_pairs_from_prefix", "budget_scale",
           "candidate_pairs_from_prefix", "budget_scale", "absolute_budget_scale",
           "encoder_grad_norm", "encoder_parameter_rms", "BUDGET_REFERENCE",
           "ABSOLUTE_STEP_FRACTION", "GRAPHLET_GRAD_BUDGET", "COLLISION_GRAD_BUDGET",
           "MAX_UPSCALE"]
