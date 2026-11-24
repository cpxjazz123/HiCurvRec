"""Independent expert queries over unpooled local-subgraph tokens."""
import math

import torch
from torch import nn


class ExpertAttention(nn.Module):
    """Return expert weights, token attention, and the effective token mask.

    The final token slot is always the fixed zero null token, and is valid
    exactly when no original token is valid. Query initialization consumes the
    caller's RNG state; queries carry no predefined geometric role.
    """

    def __init__(self):
        super().__init__()
        self.query = nn.Parameter(torch.empty(3, 32))
        nn.init.orthogonal_(self.query)
        self.key = nn.Linear(32, 32)
        self.value = nn.Linear(32, 32)
        self.score = nn.Sequential(nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
        self.register_buffer("null_token", torch.zeros(1, 1, 32))

    def forward(self, tokens, mask):
        """Map tokens [N,T,32] and bool mask [N,T] to [N,3] weights.

        Attention [N,3,T+1] normalizes over tokens, whereas the returned
        geometry weights normalize over experts. Masked slots receive exact
        zero attention, including the null slot when any real token is valid.
        """
        if tokens.ndim != 3 or tokens.shape[-1] != 32:
            raise ValueError("tokens must have shape [N,T,32]")
        if mask.shape != tokens.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be bool with shape [N,T]")
        null_valid = ~mask.any(dim=-1, keepdim=True)
        effective_mask = torch.cat((mask, null_valid), dim=-1)
        # Invalid values must not leak through 0 * NaN in the value reduction.
        real_tokens = tokens.masked_fill(~mask[..., None], 0)
        extended = torch.cat((real_tokens, self.null_token.expand(tokens.shape[0], -1, -1)), dim=1)
        keys = self.key(extended)
        values = self.value(extended)
        logits = torch.matmul(self.query, keys.transpose(-1, -2)) / math.sqrt(32.)
        logits = logits.masked_fill(~effective_mask[:, None, :], -torch.inf)
        attention = torch.softmax(logits, dim=-1)
        summaries = torch.matmul(attention, values)
        scores = self.score(summaries).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        return weights, attention, effective_mask
