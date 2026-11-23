"""Preserve multiple induced-subgraph readouts until expert cross-attention."""
import numpy as np
import torch
from torch import nn
from .structure import SubgraphTokens
from .attention import ExpertAttention


class TopologyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens=SubgraphTokens()
        self.attention=ExpertAttention()
        self.last_tokens=None
        self.last_attention=None
        self.last_token_mask=None

    def gates_for_items(self,store,item_ids,device,chunk=32,behavior_on=True):
        raw=item_ids.detach().cpu().numpy() if torch.is_tensor(item_ids) else np.asarray(item_ids)
        unique,inverse=np.unique(np.asarray(raw,dtype=np.int64),return_inverse=True)
        tokens,mask=self.tokens.tokens_for_items(store,unique,device,chunk=chunk)
        weights,attention,effective_mask=self.attention(tokens,mask)
        restore=torch.as_tensor(inverse,device=weights.device,dtype=torch.long)
        self.last_tokens=tokens[restore]
        self.last_attention=attention[restore]
        self.last_token_mask=effective_mask[restore]
        if not behavior_on:
            weights=torch.full_like(weights,1./5.)
        return self.last_tokens,weights[restore]
