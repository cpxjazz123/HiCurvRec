"""Interest models for the latent-interest-hierarchy study.

Every arm carries EXACTLY the same parameters: an item embedding table
(n_items x d) and 64 interest centroids (64 x d). "Hierarchical" is never an extra
parameter -- it is a two-level BOTTLENECK over how those same 64 centroids are
combined into a user profile:

  flat          m = sum_c q_c C_c                                with q over 64
  hier          m = sum_p q_p n_p,  n_p = sum_f q_{f|p} C_{p,f}  with q_p over 8
  random_hier   identical computation, but the 64 leaves are partitioned into
                8 groups of 8 by a FIXED RANDOM permutation fixed before training

so `hier` and `random_hier` have the same capacity and the same compute, and the
only difference is whether the bottleneck's grouping means anything. The item score
is identical everywhere, score(i|u) = <E_i, m(u)>, so a difference in validation
NLL cannot be attributed to the scoring function or to capacity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402


# Item id 0 is a REAL item in this dataset, so the embedding table carries one extra
# row and every raw id is stored shifted by one, leaving row 0 as the padding slot.
# Without this, a leading history item would be silently masked out.
N_ROWS = C.N_ITEMS + 1


def encode(ids):
    """Raw item ids -> embedding rows, with 0 reserved for padding."""
    return [i + 1 for i in ids]


def fixed_random_group(n_coarse: int, n_fine: int, seed: int) -> torch.Tensor:
    """A fixed random partition of the leaves into n_coarse groups of n_fine.

    Fixed BEFORE training and never updated: it is the negative control for the
    bottleneck, and it must carry the same number of levels and the same number of
    leaves as the learned structure.
    """
    k = n_coarse * n_fine
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(k, generator=g)
    return perm.view(n_coarse, n_fine)


class InterestModel(nn.Module):
    def __init__(self, n_items: int, dim: int, mode: str, seed: int = 42):
        super().__init__()
        if mode not in ("flat", "hier", "random_hier"):
            raise ValueError(mode)
        self.mode = mode
        self.n_items = n_items
        self.dim = dim
        self.embed = nn.Embedding(n_items, dim)
        # An item bias, identical in every arm, so the model can express the item
        # frequency prior. Without it the comparison against the popularity baseline
        # would be a comparison of a handicapped model against a fair one.
        self.bias = nn.Parameter(torch.zeros(n_items))
        self.centroids = nn.Parameter(torch.randn(C.N_INTEREST, dim, generator=
                                                  torch.Generator().manual_seed(seed)) * 0.05)
        if mode == "flat":
            self.register_buffer("group", torch.arange(C.N_INTEREST)
                                 .view(C.N_COARSE, C.N_FINE), persistent=False)
        elif mode == "hier":
            self.register_buffer("group", torch.arange(C.N_INTEREST)
                                 .view(C.N_COARSE, C.N_FINE), persistent=False)
        else:
            self.register_buffer("group", fixed_random_group(C.N_COARSE, C.N_FINE, seed),
                                 persistent=False)

    def profile(self, hist_emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """User profile m(u) in R^dim from the history embeddings (B, L, dim).

        `mask` (B, L, 1) marks real history positions. Padding positions carry the
        zero embedding, whose interest logits are all zero, so an unmasked average
        would mix a uniform 1/K distribution into every profile and, for a user with
        two purchases out of twenty slots, make the profile almost pure noise.
        """
        z = hist_emb @ self.centroids.t()                       # (B, L, K)
        if self.mode == "flat":
            q = torch.softmax(z, dim=-1)                        # (B, L, K)
            q = (q * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            return q @ self.centroids                           # (B, d)
        # two-level bottleneck: the profile is a mixture of n_coarse parents, each
        # weighted by a real softmax over the parents, and each parent is itself a
        # mixture over its own n_fine children of the SAME 64 centroids
        g = self.group
        zf = z[..., g.reshape(-1)].view(*z.shape[:-1], C.N_COARSE, C.N_FINE)
        qf = torch.softmax(zf, dim=-1)                          # (B, L, P, F)
        child_mix = torch.einsum("blpf,pfd->blpd", qf, self.centroids[g])
        # a parent's representative is the mean of its children: no new parameters
        parent_repr = self.centroids[g].mean(dim=1)             # (P, d)
        zp = hist_emb @ parent_repr.t()                         # (B, L, P)
        qp = torch.softmax(zp, dim=-1)                          # (B, L, P)
        qp = (qp * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)   # (B, P)
        cm = (child_mix * mask.unsqueeze(2)).sum(dim=1) \
            / mask.sum(dim=1).clamp(min=1.0).unsqueeze(-1)      # (B, P, d)
        return (qp.unsqueeze(-1) * cm).sum(dim=1)               # (B, d)

    def forward(self, hist_emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = self.profile(hist_emb, mask)                       # (B, d)
        logits = m @ self.embed.weight.t() + self.bias         # (B, n_items + 1)
        # Row 0 is the padding slot and is never a target. Without this its logit
        # would take part in the softmax normaliser and could occupy a top-k slot,
        # depressing Recall@10 for reasons that have nothing to do with the model.
        return logits.index_fill(1, logits.new_zeros(1, dtype=torch.long), -1e9)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def log1p_history_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    return (torch.arange(max_len, device=lengths.device).unsqueeze(0)
            < lengths.unsqueeze(1)).float()