"""Preserve multiple induced-subgraph readouts until layer-wise expert attention."""
import numpy as np
import torch
from torch import nn
from tools.issue252.host_transfer import host_int64
from .structure import SubgraphTokens
from .layer_gate import LayerExpertAttention


class TopologyAttention(nn.Module):
    """Four token types in, per-level three-expert geometry weights out."""

    def __init__(self):
        super().__init__()
        self.tokens=SubgraphTokens()
        self.attention=LayerExpertAttention()
        self.last_tokens=None
        self.last_attention=None
        self.last_token_mask=None

    def gates_for_items(self,store,item_ids,device,chunk=1024,behavior_on=True):
        raw=item_ids.detach().cpu().numpy() if torch.is_tensor(item_ids) else np.asarray(item_ids)
        unique,inverse=np.unique(np.asarray(raw,dtype=np.int64),return_inverse=True)
        tokens,mask=self.tokens.tokens_for_items(store,unique,device,chunk=chunk)
        weights,attention,effective_mask=self.attention(tokens,mask)
        restore=host_int64(inverse,weights.device)
        self.last_tokens=tokens[restore]
        self.last_attention=attention[restore]
        self.last_token_mask=effective_mask[restore]
        if not behavior_on:
            # The same uniform mixture for every level: the experts stay
            # normalised per level, so this is not a different scale.
            weights=torch.full_like(weights,1./3.)
        return self.last_tokens,weights[restore]
