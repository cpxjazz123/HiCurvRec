"""Issue #231 §2/§3: graphlet supervision on the real three-level SID, inside the total loss.

What differs from #230, and why each difference is forced by #231:

1. **The term reads the quantizer's own hard codes.** `P_l(i,j) = prod_{k<=l} 1[c_ik = c_jk]`
   over the first three codes only; `S = P_1 + P_2 + P_3`. #230 built the hinge on a cost
   matrix before assignment; the issue now says the supervision must read the quantised output
   (`使用实际逐级硬残差轨迹,不另造与量化器不同的 argmin`).
2. **It enters the optimiser's graph.** `structure_loss` returns a scalar that is added to
   `L_RQVAE` and differentiated by the ordinary `backward()`; there is no
   `autograd.grad(...)`-then-patch-the-cost path (`不得…仅把 detached gradient 加到距离矩阵`).
3. **Forward hard, backward soft.** `ste_one_hot` returns `p + stopgrad(h - p)`, so the value is
   the hard one-hot and the gradient is the assignment probability's.

The level contributions stay 0/1 indicators; multiplying by the level index (the old `level * p`
convention) would make an L3 share worth three times an L1 share and is explicitly excluded.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

KINDS = ("chain", "fork", "merge")
CODES = 3
MARGIN = 1.0


def ste_one_hot(logits: torch.Tensor) -> torch.Tensor:
    """`y = p + stopgrad(h - p)`: forward equals the hard argmin one-hot, backward flows via p.

    ``logits`` are costs (lower is better), matching the quantizer's distance matrix.
    """
    p = torch.softmax(-logits.float(), dim=-1)
    index = torch.argmin(logits.detach(), dim=-1)
    h = F.one_hot(index, logits.shape[-1]).to(p.dtype)
    return p + (h - p).detach()


def hard_codes(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmin(logits.detach(), dim=-1)


def level_contributions(codes, left, right) -> list:
    """P_l for l = 1..3, each a 0/1 indicator; the running product is the prefix condition."""
    if codes.dim() == 1:
        # one level of codes: P_1 is the whole answer
        return [(codes[left] == codes[right]).to(torch.float32)]
    equal = codes[left] == codes[right]
    out, run = [], torch.ones(equal.shape[0], dtype=equal.dtype, device=equal.device)
    for level in range(CODES):
        run = run * equal[:, level].to(run.dtype)
        out.append(run)
    return out


def shared_prefix(codes, left, right) -> torch.Tensor:
    """`S(i,j) = P_1 + P_2 + P_3`, ignoring any column past the third.

    1-D input means a single level of codes, so the sum is P_1 alone.
    """
    sliced = codes if codes.dim() == 1 else codes[:, :CODES]
    return sum(level_contributions(sliced, left, right))


def delta(codes, roles, kind: str) -> torch.Tensor:
    """The issue's three contrasts, over the shared role tuple ``(A, B, C)``."""
    # one level of hard codes stays 1-D: shared_prefix reads it as P_1 alone
    if codes.dim() > 1:
        codes = codes[:, :CODES]
    a, b, c = roles[:, 0], roles[:, 1], roles[:, 2]
    s_ab = shared_prefix(codes, a, b)
    s_bc = shared_prefix(codes, b, c)
    s_ac = shared_prefix(codes, a, c)
    if kind == "chain":
        return (s_ab + s_bc) / 2.0 - s_ac
    # fork: siblings are B,C; merge: siblings are B,C with A the centre. Same formula (§2).
    return s_bc - (s_ab + s_ac) / 2.0


def structure_loss(soft_codes, hard_codes_list, roles, kind: str, *, margin: float = MARGIN,
                   weight: torch.Tensor | None = None) -> torch.Tensor:
    """`mean max(0, m - Delta)` computed from hard codes, differentiable through the soft ones.

    ``soft_codes`` is a list of per-level `(items, codebook)` tensors produced by
    :func:`ste_one_hot`; ``hard_codes_list`` is kept for symmetry with callers that already hold
    integer codes. The Delta that enters the value is the **hard** one, exactly as the export
    will see it (§3: `不能以软概率结果替代硬 SID 结果`).
    """
    hard = hard_codes_list
    if hard is None:
        hard = soft_codes
    value = (-delta(hard, roles, kind) + margin).clamp(min=0.0)
    if weight is not None:
        value = value * weight.to(value.dtype)
    soft_value = _soft_delta(soft_codes, roles, kind)
    # Straight-through on the scalar: the value is the hard hinge `m - Delta`, so its slope
    # must be `-d(Delta)`. Adding `+soft_delta` inverted the direction -- measured: the hard
    # delta fell from 0.0 to -1.0 under 60 descent steps, and rose to +0.5 under ascent.
    return (value.detach() - soft_value + soft_value.detach()).mean()


def _pair_collision(level, left, right):
    """`sum_k p_left[k] * p_right[k]`: the probability the two items take the SAME codeword.

    Using the expected value of the indicator product alone was measured to be exactly 0 for a
    uniform assignment, so its derivative vanished and the term could not move anything. The
    collision probability is the quantity whose maximisation actually pulls two items onto one
    codeword, and it is the natural differentiable reading of `1[c_i = c_j]`.
    """
    return (level[left] * level[right]).sum(-1)


def _soft_delta(soft_codes, roles, kind: str) -> torch.Tensor:
    """Differentiable surrogate of the same contrast, built from the STE one-hots."""
    a, b, c = roles[:, 0], roles[:, 1], roles[:, 2]
    total = None
    run_ab, run_bc, run_ac = 1.0, 1.0, 1.0
    s_ab = s_bc = s_ac = None
    for level in soft_codes:
        run_ab = run_ab * _pair_collision(level, a, b)
        run_bc = run_bc * _pair_collision(level, b, c)
        run_ac = run_ac * _pair_collision(level, a, c)
        s_ab = run_ab if s_ab is None else s_ab + run_ab
        s_bc = run_bc if s_bc is None else s_bc + run_bc
        s_ac = run_ac if s_ac is None else s_ac + run_ac
    if kind == "chain":
        return (s_ab + s_bc) / 2.0 - s_ac
    return s_bc - (s_ab + s_ac) / 2.0


def combine(*, base: torch.Tensor, graphlet: torch.Tensor, weight: float):
    """`L_total = L_RQVAE + weight * L_graphlet`, reported term by term.

    ``weight == 0`` must reproduce the baseline exactly (C27): the graphlet term is multiplied
    by zero before it joins the sum, and the reported component is zero rather than absent.
    """
    weighted = graphlet * float(weight)
    return base + weighted, {"graphlet": float(graphlet.detach()), "graphlet_weighted": weighted}
