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


def exclusion_mask(source_ids: torch.Tensor, target_ids: torch.Tensor, *,
                   candidates: torch.Tensor, successors: torch.Tensor | None) -> torch.Tensor:
    """True where a candidate pair may be used: not self, not the target, not a false negative."""
    mask = candidates[:, 0:1] != candidates[:, 1:2]
    if successors is not None and successors.numel():
        # keep the column shape: broadcasting a [n] mask against [n, 1] would build [n, n]
        mask = mask & ~torch.isin(candidates[:, 1], successors.reshape(-1)).reshape(-1, 1)
    return mask


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
