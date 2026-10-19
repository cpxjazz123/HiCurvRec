"""Track F (issue #85): a train-only soft repulsion on the L1 posterior.

Track E changes which direction the encoder moves. This changes what the L1 posterior is allowed to
look like, without touching the hard argmin that defines a SID.

For item ``i`` with latent ``z_i`` and codebook ``c``,

    p_i(k) = softmax_k(-d(z_i, c_k) / tau),
    P_same(i, j) = sum_k p_i(k) p_j(k),

and the separation term is ``lambda * mean_over_unrelated_pairs(P_same)`` over TRAIN-ONLY pairs
whose exclusion mask removes self, the observed target, and every known successor / false negative.
Positives keep the arm's own behavioural CE untouched.

Three traps this module is built to make visible rather than to avoid:

1. a uniform ``p_i`` drives ``P_same`` to ``1/K`` and scores well, so hard SID collisions, posterior
   entropy and the nearest-centre margin are reported next to the loss, separately;
2. finite differences are legitimate for this SMOOTH surrogate and illegitimate for the hard
   argmin, so the oracle differentiates the surrogate only and says so;
3. a soft separation term is a geometric RELAXATION, not a hard constraint: it can never promise
   that two items end up in different codes, and nothing here claims it does.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
for entry in (str(REPO_ROOT / "tools"), str(REPO_ROOT / "tools/cgv")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

DEFAULT_TAU = 1.0
DEFAULT_LAMBDA = 0.10
STABLE_MIN_TAU = 1e-3


def posterior(model, latents: torch.Tensor, *, tau: float = DEFAULT_TAU) -> torch.Tensor:
    """p_i(k) = softmax(-d(z_i, c_k)/tau), in the quantizer's OWN distance.

    The temperature is validated rather than clamped silently: a tau at or below the stability
    floor would turn the surrogate into a hard argmin with no gradient, which is the exact thing
    this lane exists not to do.
    """
    if tau < STABLE_MIN_TAU:
        raise ValueError(f"tau={tau} is below the stability floor {STABLE_MIN_TAU}: the soft "
                         f"surrogate would collapse to a hard argmin and carry no gradient")
    layer = model.rq.vq_layers[0]
    distance = layer._distances(latents.reshape(-1, layer.dim))
    return torch.softmax(-distance / tau, dim=-1)


def same_code_probability(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """P_same(i, j) = sum_k p_i(k) p_j(k), the Bhattacharyya-style inner product."""
    return (left * right).sum(dim=-1)


def separation_loss(posteriors: torch.Tensor, pair_index: torch.Tensor, *,
                    lambda_weight: float = DEFAULT_LAMBDA) -> dict[str, Any]:
    """Mean P_same over the unrelated pairs that survived the exclusion mask."""
    if pair_index.numel() == 0:
        return {"loss": torch.zeros((), device=posteriors.device, requires_grad=False),
                "pairs": 0,
                "note": "the exclusion mask removed every candidate pair at this step"}
    left = posteriors[pair_index[:, 0]]
    right = posteriors[pair_index[:, 1]]
    values = same_code_probability(left, right)
    return {"loss": lambda_weight * values.mean(), "pairs": int(pair_index.shape[0]),
            "mean_p_same": float(values.detach().mean()),
            "max_p_same": float(values.detach().max())}


def _successor_lookup(successors) -> dict[int, set[int]]:
    """Per-source successor sets from EITHER a host CSR (i, j) or a flat edge list."""
    if successors is None:
        return {}
    if hasattr(successors, "tocsr"):                    # scipy CSR built by the host trainer
        matrix = successors.tocsr()
        return {int(row): set(int(col) for col in matrix.indices[matrix.indptr[row]:
                                                                 matrix.indptr[row + 1]])
                for row in range(matrix.shape[0]) if matrix.indptr[row] < matrix.indptr[row + 1]}
    if isinstance(successors, torch.Tensor) and successors.dim() == 2:
        return {int(row): set(int(value) for value in successors[row].tolist())
                for row in range(successors.shape[0]) if int(successors[row].any())}
    return {}


def source_conditioned_mask(source_ids: torch.Tensor, target_ids: torch.Tensor, *,
                            candidates: torch.Tensor,
                            successors) -> torch.Tensor:
    """True where a candidate pair may be used as an UNRELATED pair.

    Built per source, because "unrelated" is a property of the SOURCE, not of the item:

      - ``j == i`` is excluded;
      - ``j`` equal to THIS source's observed target is excluded -- the old implementation ignored
        ``target_ids`` entirely and so admitted the true positive as an unrelated candidate;
      - ``j`` among THIS source's known train successors is excluded (the false-negative case the
        host's own CSR exists for);
      - ``j`` that happens to be a successor of a DIFFERENT source is KEPT. A global
        ``torch.isin`` over successor values wrongly dropped those, and did so silently.

    The graph is indexed by ITEM id. Candidate rows are pairs of item ids, never positions within
    a pair batch, and mixing the two is the other way this mask can be wrong without failing.
    """
    lookup = _successor_lookup(successors)
    keep = torch.ones(candidates.shape[0], dtype=torch.bool)
    for row in range(candidates.shape[0]):
        item_i = int(candidates[row, 0])
        item_j = int(candidates[row, 1])
        if item_i == item_j:
            keep[row] = False
            continue
        # The source of THIS candidate is item_i. The observed target is looked up per row from the
        # batch's own source/target pairing, which is what makes the exclusion source-conditioned.
        batch_position = int(source_ids.eq(item_i).nonzero()[0]) if bool(source_ids.eq(item_i).any()) \
            else -1
        if batch_position >= 0 and int(target_ids[batch_position]) == item_j:
            keep[row] = False
            continue
        if item_j in lookup.get(item_i, ()):
            keep[row] = False
    return keep


def valid_pair_indices(source_ids: torch.Tensor, target_ids: torch.Tensor, *,
                       candidates: torch.Tensor, successors) -> dict[str, Any]:
    """The surviving candidates plus the per-step counts the issue requires to be logged."""
    mask = source_conditioned_mask(source_ids, target_ids, candidates=candidates,
                                   successors=successors)
    return {
        "valid": candidates[mask],
        "mask": mask,
        "candidate_count": int(candidates.shape[0]),
        "valid_count": int(mask.sum()),
        "empty_count": int((~mask).sum()),
    }


def exclusion_mask(source_ids: torch.Tensor, target_ids: torch.Tensor, *,
                   candidates: torch.Tensor, successors) -> torch.Tensor:
    """Compatibility name, now source-conditioned.

    The #85 version of this function took a flat successor tensor, ignored ``target_ids`` and
    filtered globally. Both defects are demonstrated in tests/test_r7_f_masks.py before and after
    this change; the source-conditioned version is the one every call site now uses.
    """
    return source_conditioned_mask(source_ids, target_ids, candidates=candidates,
                                   successors=successors)


def gaming_diagnostics(posteriors: torch.Tensor, hard_indices: torch.Tensor,
                       margins: torch.Tensor) -> dict[str, Any]:
    """The three quantities that catch a uniform-posterior cheap pass.

    A loss that improves because every ``p_i`` went uniform shows up immediately as collapsing
    hard diversity and vanishing margins, even while ``P_same`` sits near its ``1/K`` floor.
    """
    entropy = -(posteriors * posteriors.clamp_min(1e-12).log()).sum(dim=-1)
    uniform = float(entropy.mean() / torch.log(torch.tensor(float(posteriors.shape[1]))))
    counts = torch.bincount(hard_indices, minlength=posteriors.shape[1]).to(torch.float64)
    share = counts / counts.sum().clamp_min(1.0)
    used = int((counts > 0).sum())
    return {
        "posterior_entropy_mean": float(entropy.mean()),
        "posterior_entropy_relative_to_uniform": uniform,
        "uniform_is_perfect_score": False,
        "hard_codes_used": used,
        "hard_effective_codes": float(torch.exp(-(share[share > 0] * share[share > 0].log()).sum())),
        "hard_max_share": float(share.max()),
        "margin_p05": float(margins.quantile(0.05)) if margins.numel() else None,
        "reading": "a uniform posterior scores well on the separation loss while collapsing hard "
                   "diversity; if hard_effective_codes falls while the loss falls, reject",
    }
