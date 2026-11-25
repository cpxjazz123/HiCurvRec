"""Layer-wise curvature gating over direction-aware ego tokens.

One shared directed GNN produces four token types per item.  Every level owns
its own trainable queries, so the levels may learn different curvature
mixtures; nothing hard-wires a curvature to a level.  Attention normalises over
the token axis, the returned geometry weights normalise over the expert axis.

R05: the second level is no longer learned per item.  Its mixture is the
item's graphlet role (see ``model/role_mix.py``); the two learned gate rows
serve the first and the third level, so neither row is gradient-free and R04's
removal of the level-3 mixture is undone.
"""
from __future__ import annotations

import math

import torch
from torch import nn

#: The learned gate rows: row 0 serves semantic level 1, row 1 serves level 3.
#: The second level is role-conditioned, so the gate does not own a row for it.
LEVELS = 2
#: The three fixed curvature experts {-1, 0, +1}.
EXPERTS = 3
#: The token dimension of the shared GNN readout.
DIM = 32


class LayerExpertAttention(nn.Module):
    """Per-level expert queries over the unpooled token set.

    ``forward`` returns the geometry weights ``[N, LEVELS, EXPERTS]`` that
    normalise over experts, the token attention ``[N, LEVELS, EXPERTS, T+1]``
    that normalises over tokens, and the effective token mask ``[N, T+1]``.
    The final slot is the fixed zero null token and is valid exactly when no
    real token is valid.
    """

    def __init__(self, levels: int = LEVELS, experts: int = EXPERTS, dim: int = DIM):
        super().__init__()
        self.levels, self.experts, self.dim = int(levels), int(experts), int(dim)
        self.query = nn.Parameter(torch.empty(self.levels, self.experts, self.dim))
        nn.init.orthogonal_(self.query.reshape(self.levels * self.experts, self.dim))
        self.key = nn.Linear(self.dim, self.dim)
        self.value = nn.Linear(self.dim, self.dim)
        self.score = nn.Sequential(nn.Linear(self.dim, self.dim), nn.Tanh(), nn.Linear(self.dim, 1))
        self.register_buffer("null_token", torch.zeros(1, 1, self.dim))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim:
            raise ValueError("tokens must have shape [N,T,%d]" % self.dim)
        if mask.shape != tokens.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be bool with shape [N,T]")
        null_valid = ~mask.any(dim=-1, keepdim=True)
        effective_mask = torch.cat((mask, null_valid), dim=-1)
        real_tokens = tokens.masked_fill(~mask[..., None], 0)
        extended = torch.cat((real_tokens, self.null_token.expand(tokens.shape[0], -1, -1)), dim=1)
        keys = self.key(extended)
        values = self.value(extended)
        logits = torch.einsum("led,ntd->nlet", self.query, keys) / math.sqrt(self.dim)
        logits = logits.masked_fill(~effective_mask[:, None, None, :], -torch.inf)
        attention = torch.softmax(logits, dim=-1)
        summaries = torch.einsum("nlet,ntd->nled", attention, values)
        scores = self.score(summaries).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        return weights, attention, effective_mask


def cumulative_prefixes(level_codes):
    """The exact running sum of the hard level codewords: q^(l) = sum_{j<=l} e_j.

    ``level_codes`` is the ordered list of per-level hard codeword tensors.
    The result has shape ``[levels, N, dim]`` and is bitwise the sum of the
    first ``l`` selected codewords, i.e. the actual SID's first ``l`` codes.
    """
    codes = list(level_codes)
    if not codes:
        raise ValueError("at least one level is required")
    shape = codes[0].shape
    running = torch.zeros_like(codes[0])
    prefixes = []
    for code in codes:
        if code.shape != shape:
            raise ValueError("every level must contribute the same shape")
        running = running + code
        prefixes.append(running.clone())
    return torch.stack(prefixes, dim=0)


def aggregate_layer_losses(values, weights=None):
    """The alpha-weighted sum of the per-level structural losses.

    The first version uses equal weights ``alpha_l = 1/3``; the weights are
    explicit so a later change cannot silently move the level balance.
    """
    terms = [value for value in values]
    if not terms:
        raise ValueError("at least one level loss is required")
    if weights is None:
        weights = (1.0 / len(terms),) * len(terms)
    if len(weights) != len(terms):
        raise ValueError("one weight per level is required")
    total = terms[0] * float(weights[0])
    for term, weight in zip(terms[1:], weights[1:]):
        total = total + term * float(weight)
    return total
