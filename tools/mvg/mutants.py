"""The mutation suite: named defects, and the gate that has to reject each of them.

A gate that never rejects anything is not a gate. Each entry below is a defect that a
plausible implementation could carry, injected into a scratch copy of the mechanism tree by
a source replacement, and the gate named in ``must_fail_gate`` has to notice.

The mutants are never applied to the repository's tree: each one lives in a temporary copy
that is deleted afterwards, so a mutant can never be committed and can never be run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Mutant:
    id: str
    why: str
    replacements: Tuple[Tuple[str, str], ...]
    must_fail_gate: str
    rationale: str


MUTANTS: Tuple[Mutant, ...] = (
    Mutant(
        id="gw_to_mse",
        why=(
            "the relational term is a discrepancy between two codebook assignments measured "
            "through the codebook's own geometry; a per-element mean squared error on the "
            "assignments is a different quantity"
        ),
        replacements=(
            (
                "    structural_term = (\n"
                "        src_assign.unsqueeze(2) * tgt_assign.unsqueeze(1) * "
                "codebook_geometry.unsqueeze(0)\n"
                "    ).sum(dim=(1, 2))",
                "    structural_term = ((src_assign - tgt_assign) ** 2).mean(dim=1)",
            ),
        ),
        must_fail_gate="M1",
        rationale="the forward value differs from the independent reference",
    ),
    Mutant(
        id="transpose_pc_cc",
        why=(
            "the assignment is a softmax over the CODEBOOK axis; computing it over the pair "
            "axis confuses a [B, C] matrix with a [C, C] one"
        ),
        replacements=(
            (
                "    src_assign = torch.softmax(-src_dist / temperature, dim=1)",
                "    src_assign = torch.softmax(-src_dist / temperature, dim=0)",
            ),
        ),
        must_fail_gate="M1",
        rationale="the forward value differs from the independent reference",
    ),
    Mutant(
        id="latent_only_zero_codebook_gradient",
        why=(
            "the specification declares a nonzero gradient on the codebook; detaching it "
            "leaves the encoder learning a term the codebook cannot see"
        ),
        replacements=(
            (
                "    src_delta = torch.abs(z_src.unsqueeze(1) - codebook.unsqueeze(0))",
                "    codebook = codebook.detach()\n"
                "    src_delta = torch.abs(z_src.unsqueeze(1) - codebook.unsqueeze(0))",
            ),
        ),
        must_fail_gate="M2",
        rationale="the codebook gradient is zero while the specification declares it nonzero",
    ),
    Mutant(
        id="no_op_loss",
        why="a term that contributes nothing still looks like a mechanism in the loss graph",
        replacements=(
            (
                "    return (per_pair * weights).sum() / valid",
                "    return per_pair.sum() * 0.0",
            ),
        ),
        must_fail_gate="M1",
        rationale="the forward value is zero while the reference is not",
    ),
    Mutant(
        id="symmetric_for_directed",
        why=(
            "the mechanism is a directed transition; averaging the two directions turns it "
            "into a similarity"
        ),
        replacements=(
            (
                "    margin_term = torch.clamp(margin + tgt_nearest - src_nearest, min=0.0)",
                "    _reverse_margin = torch.clamp(margin + src_nearest - tgt_nearest, min=0.0)\n"
                "    margin_term = 0.5 * (\n"
                "        torch.clamp(margin + tgt_nearest - src_nearest, min=0.0) + _reverse_margin\n"
                "    )",
            ),
        ),
        must_fail_gate="M1",
        rationale="the forward value differs from the independent reference",
    ),
    Mutant(
        id="detached_gradient",
        why=(
            "a loss that is detached from the graph trains nothing while still appearing in "
            "the loss line"
        ),
        replacements=(
            (
                "    return (per_pair * weights).sum() / valid",
                "    return ((per_pair * weights).sum() / valid).detach()",
            ),
        ),
        must_fail_gate="M2",
        rationale="the declared gradient targets receive no gradient at all",
    ),
    Mutant(
        id="wrong_masked_ddp_normalization",
        why=(
            "the denominator of the masked mean is the number of VALID pairs; dividing by the "
            "batch size rescales the term whenever a pair is masked out, and the "
            "rescaling is invisible in the loss line"
        ),
        replacements=(
            (
                "    return (per_pair * weights).sum() / valid",
                "    return (per_pair * weights).sum() / float(z_src.shape[0])",
            ),
        ),
        must_fail_gate="M1",
        rationale="the forward value differs from the independent reference on a masked case",
    ),
)


def by_id() -> dict:
    return {mutant.id: mutant for mutant in MUTANTS}


__all__ = ["MUTANTS", "Mutant", "by_id"]
