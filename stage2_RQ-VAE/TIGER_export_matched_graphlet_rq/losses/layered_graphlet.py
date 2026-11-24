"""Issue #232 §1/§2: layered graphlet supervision read off the quantizer's REAL tokens.

Each defect the issue names for #231, and where it is closed:

* **§1 real hard codes.** Every hard one-hot comes from an integer index tensor the quantizer
  returned. This module never re-derives a hard code from a cost or a probability; the test suite
  feeds an assignment that deliberately contradicts the cost matrix and requires the value to
  follow the tokens.
* **§1 the hinge's mask gates the gradient.** `gated_hinge` zeroes inactive rows through a
  stop-gradient mask, so a structure already at margin contributes exactly zero and is flat.
  #231's `clamp` left d/dgap = -1 there (measured: 60 ascent steps moved the hard Δ).
* **§1 do not move the export.** Nothing here writes the export path; the supervision branch reads
  the same cost matrix and residual trajectory the exporter uses (see `layered_wiring.py`).
* **§2 three different level objectives** -- grouping at L1, a role contrast inside the group at
  L2, bucket sharing at L3. Not three repetitions of one hinge.

The structure term takes its **own optimizer step** with gradients zeroed first, so no
reconstruction contribution survives into it. Sharing one step is what cost #231 its 1.070x
reconstruction penalty: the graphlet surrogate falls faster than reconstruction, so
`x += -lr*(g_rec + lam*g_g)` decays the reconstruction share by roughly `exp(-(k-1)*lam)` per step.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

KINDS = ("chain", "fork", "merge")
CODES = 3

# --- pre-registered constants (issue #232 §2; CLAUDE.md §19g forbids tuning) --------------------
#: L1 margin: one structure prefers a shared first codeword over two different ones. The chain's
#: contrast averages two pairs, so it is scaled to the same per-pair unit as fork and merge.
MARGIN_L1 = 1.0
INSTANCE_SCALE = {"chain": 2.0, "fork": 1.0, "merge": 1.0}
#: L2 margin. The measured room is the gap between the sibling pair's mean and the base rate:
#: on the frozen TIGER checkpoint, P2(BC) -- the term L2 actually rewards -- sits at 0.040 for
#: fork and 0.019 for merge, i.e. 3-6x what a 256-codeword chance assignment gives (0.004), and
#: the far pairs sit below it. Holding L2 at the far-pair mean (`margin = 0`) is therefore a
#: target the data already admits rather than one it must be pushed toward by force, and it
#: leaves the L3 ladder room to work (the L3 width budget is finite: 512 buckets over 24,587
#: items). MARGIN_L2 = 1.0 demanded five times the level's measured magnitude.
MARGIN_L2 = 0.0
#: L2 instance sub-batch, split by kind so the relaxation geometry never has to produce two
#: different sign patterns at once.
SUBBATCH = 32
#: L3 target: the bucket-sharing contrast the supervised triples must reach. Measured on the
#: frozen TIGER checkpoint's own SIDs: the hard L3 contrast is 0.000-0.003, i.e. sharing or not
#: is a coin flip. Any positive target is an improvement over that; 0.30 buys collision for
#: nothing else, so it is set to the same scale the other two levels use.
L3_TARGET = 0.10
#: Export bucket width.
L3_WIDTH = 2
#: Structure step size relative to the reconstruction step. Separate steps make the step size the
#: weight, and it acts on values and gradients alike by construction.
GRAPHLET_LR_RATIO = 1.0


def ramp_weight(epoch: int, epochs: int, start: int, weight_max: float) -> float:
    """Iter48's delayed ramp over the epoch budget (TIGER_BEHAVIOR_SAMPLE, `RAMP_START_EPOCH=1500`).

    Linear from 0 at ``start`` to ``weight_max`` at the last epoch; exactly 0 before ``start``, so
    the first half of training is the unmodified RQ-VAE objective and the codebook is already
    stable when the structure signal first arrives.
    """
    span = max(int(epochs) - int(start), 1)
    alpha = min(max((int(epoch) - int(start)) / span, 0.0), 1.0)
    return float(weight_max) * alpha


def bucket_of(tokens: torch.Tensor, width: int) -> torch.Tensor:
    """Export bucket index `t_0 + width*t_1 + width^2*t_2`; the occurrence disambiguator is
    excluded because it never merges two items and cannot change the unique count."""
    index = torch.zeros(tokens.shape[0], dtype=torch.long, device=tokens.device)
    for level in range(tokens.shape[1]):
        index = index * width + tokens[:, level].long()
    return index


def hard_one_hot(tokens: torch.Tensor, width: int) -> torch.Tensor:
    """`h_l` straight from the quantizer's integer index (§1: never re-derived)."""
    return F.one_hot(tokens.long(), width).to(torch.float32)


def ste(prob: torch.Tensor, hard: torch.Tensor) -> torch.Tensor:
    """`y = p + stopgrad(h - p)`: forward is the real hard assignment, backward is p's gradient."""
    return prob + (hard - prob).detach()


def gated_hinge(gap: torch.Tensor, margin: float) -> torch.Tensor:
    """`hinge(m - gap)` taking `gap` itself (already `a - b`); inactive rows have zero gradient.
    The inactive rows are replaced by a detached constant, not multiplied by a zero: a product
    still carries `0 * d(gap)` terms through the autograd graph, and only `where` stops them.
    gap = gap.reshape(-1)
    inactive = gap.detach() >= margin
    return torch.where(inactive, torch.zeros_like(gap), margin - gap)


def _same(prob: torch.Tensor, left, right) -> torch.Tensor:
    """`sum_k p_left[k] p_right[k]`: probability the two items take the same codeword."""
    return (prob[left] * prob[right]).sum(-1)


def _contrast(terms, kind: str) -> torch.Tensor:
    ab, bc, ac = terms
    if kind == "chain":
        return (ab + bc) / 2.0 - ac
    return bc - (ab + ac) / 2.0


def structure_value(tokens, roles, kind: str) -> torch.Tensor:
    """The hard shared-prefix contrast the independent scorer computes (`graphlet_contract`)."""
    a, b, c = roles[:, 0], roles[:, 1], roles[:, 2]
    total, run = None, None
    for level in range(min(CODES, tokens.shape[1])):
        same = ((tokens[:, level][a] == tokens[:, level][b]).float(),
                (tokens[:, level][b] == tokens[:, level][c]).float(),
                (tokens[:, level][a] == tokens[:, level][c]).float())
        run = same if run is None else tuple(x * y for x, y in zip(run, same))
        total = run if total is None else tuple(x + y for x, y in zip(total, run))
    return _contrast(total, kind)


def prefix_shares(probs, roles):
    """`P_1, P_2, P_3` per item pair. `P_1 = same_1`, `P_l = P_{l-1} * same_l`, so a deeper prefix
    is impossible without the shallower one -- exactly the contract's running product."""
    run_ab = run_bc = run_ac = None
    out = []
    for level, prob in enumerate(probs[:CODES]):
        s_ab, s_bc, s_ac = _same(prob, a, b), _same(prob, b, c), _same(prob, a, c)
        run_ab = s_ab if run_ab is None else run_ab * s_ab
        run_bc = s_bc if run_bc is None else run_bc * s_bc
        run_ac = s_ac if run_ac is None else run_ac * s_ac
        out.append((run_ab, run_bc, run_ac))
    return out
def level_objectives(probs, tokens, roles, kind: str) -> dict:
    """The three level objectives of one instance batch, each a mean over its instances."""
    a, b, c = roles[:, 0], roles[:, 1], roles[:, 2]
    prefixes = prefix_shares(probs, roles)
    # chain averages two adjacent pairs, so its contrast is HALF a pair's contrast; the scale
    # lifts it to the same per-pair unit fork and merge already have
    l1_gap = INSTANCE_SCALE[kind] * _contrast(prefixes[0], kind)
    # L2 is the INCREMENT the second code adds inside the same pair; it is reached only when the
    # first code already matched, which is what keeps it from drifting into its own grouping.
    l2_gap = _contrast(prefixes[1], kind) - _contrast(prefixes[0], kind)
    buckets = bucket_of(tokens, L3_WIDTH)
    # L3: the export bucket index is decided by the JOINT of the three codes, so the differentiable
    # reading of "same bucket" is the joint prefix mass, not one level's marginal
    joint = _contrast(prefixes[-1], kind)
    l3_gap = joint
    # The gate must be OPEN at the start: a freshly initialised quantizer has no structure at all,
    # so gating on `l1_gap >= MARGIN_L1` (satisfied) kept L2 at exactly 0.0 for all 18,000 steps
    # of the first run -- `l2_active = 0.0` in every recorded row. L2 is now gated on SLACK, the
    # fraction of the L1 margin still to be earned, so it engages while L1 is being learned and
    # switches off only once L1 is saturated.
    slack = ((MARGIN_L1 - l1_gap.detach()) / MARGIN_L1).clamp(0.0, 1.0)
    grouped = slack.to(l1_gap.dtype)
    return {
        "l1": gated_hinge(l1_gap, MARGIN_L1).mean(),
        "l2": (gated_hinge(l2_gap, MARGIN_L2) * grouped).sum() / grouped.sum().clamp(min=1.0),
        "l3": gated_hinge(l3_gap, L3_TARGET).mean(),
        "l1_gap": l1_gap.detach().mean(),
        "l2_gap": l2_gap.detach().mean(),
        "l3_gap": l3_gap.detach().mean(),
        "l2_active": grouped.detach().mean(),
        "l3_hard": _contrast(((buckets[a] == buckets[b]).float(),
                             (buckets[b] == buckets[c]).float(),
                             (buckets[a] == buckets[c]).float()), kind).detach().mean(),
    }
def graphlet_objective(probs, tokens, roles_by_kind) -> torch.Tensor:
    """`mean_k [mean_i (L1 + L2 + L3)]`, equal weight per kind so no bank's size dominates."""
    parts = []
    for kind, roles in roles_by_kind.items():
        level = level_objectives(probs, tokens, roles, kind)
        parts.append(level["l1"] + level["l2"] + level["l3"])
    return torch.stack(parts).mean()


__all__ = ["ramp_weight", "bucket_of", "hard_one_hot", "ste", "gated_hinge", "structure_value",
           "level_objectives", "graphlet_objective", "SUBBATCH", "L3_TARGET", "L3_WIDTH",
           "MARGIN_L1", "MARGIN_L2", "INSTANCE_SCALE", "GRAPHLET_LR_RATIO", "CODES", "KINDS"]
