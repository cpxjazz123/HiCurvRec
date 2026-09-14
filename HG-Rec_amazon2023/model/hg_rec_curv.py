"""HG_Rec_Curv — Stage 3 learnable κ via Poincaré Embedding (input side).

R36 (#2: Stage 3 κ frozen→learnable): 替换 T5 shared embedding 为 PoincaréEmbedding
(expmap0 project), 引入可学习曲率 log_kappa; 其余 T5 层保持 Euclidean.

d_κ(x, y) = (1/√κ) arccosh(1 + 2κ ||x-y||² / ((1-κ||x||²)(1-κ||y||²)))
expmap0(v) = v * tanh(√κ ||v||/2) / (√κ ||v||)

约定: 输入 embedding 落在 Poincaré ball (B_κ^{d_model}, 内积空间曲率 κ).
lm_head 用原始 Euclidean weight (tied with shared.embedding.weight).
"""
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config
from typing import Optional, Dict, Any


class PoincareEmbedding(nn.Module):
    """Poincaré ball embedding with learnable curvature κ.

    Maps vocab indices → points on Poincaré ball B_κ^{d}.
    Forward returns expmap0(E[idx]).
    Exposes `.weight` (raw Euclidean param) for tied lm_head compatibility.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int,
                 kappa_init: float = 1.0):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        # Initialize small so ||v|| < 1/√κ initially (well inside ball)
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[padding_idx].zero_()
        # Learnable curvature κ ∈ [0.01, 10], init κ=1.0
        self.log_kappa = nn.Parameter(torch.tensor(float(kappa_init)).log())

    @property
    def kappa(self) -> torch.Tensor:
        return self.log_kappa.exp().clamp(min=0.01, max=10.0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        v = self.embedding(input_ids)            # (B, L, d), Euclidean
        k = self.kappa
        sqrt_k = k.sqrt()
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-7)
        # expmap0: v → v * tanh(√κ ||v||/2) / (√κ ||v||)
        factor = torch.tanh(sqrt_k * v_norm / 2.0) / (sqrt_k * v_norm)
        return v * factor                       # now ||x|| < 1/√κ

    @property
    def num_embeddings(self) -> int:
        return self.embedding.num_embeddings

    @property
    def embedding_dim(self) -> int:
        return self.embedding.embedding_dim

    @property
    def padding_idx(self) -> int:
        return self.embedding.padding_idx

    @property
    def weight(self) -> nn.Parameter:
        # T5 uses .weight for tied lm_head
        return self.embedding.weight


class HG_Rec_Curv(nn.Module):
    """HG_Rec with learnable κ on input embedding (Poincaré ball).

    Replaces T5's `shared` (input embedding) with PoincareEmbedding. The lm_head
    keeps tied Euclidean weight so logits remain standard CE on vocab distribution.
    κ starts at 1.0 and is jointly optimized with model parameters via Adam.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        t5config = T5Config(
            num_layers=config['num_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            num_heads=config['num_heads'],
            d_kv=config['d_kv'],
            dropout_rate=config['dropout_rate'],
            vocab_size=config['vocab_size'],
            pad_token_id=config['pad_token_id'],
            eos_token_id=config['eos_token_id'],
            decoder_start_token_id=config['pad_token_id'],
            feed_forward_proj=config['feed_forward_proj'],
        )
        self.model = T5ForConditionalGeneration(t5config)
        # Replace shared embedding with Poincaré version
        new_shared = PoincareEmbedding(
            num_embeddings=config['vocab_size'],
            embedding_dim=config['d_model'],
            padding_idx=config['pad_token_id'],
            kappa_init=1.0,
        )
        # Init: copy old shared weights to keep T5 initialization
        with torch.no_grad():
            new_shared.embedding.weight.data.copy_(self.model.shared.weight.data)
        self.model.shared = new_shared
        # Re-tie lm_head to new shared weight
        self.model.lm_head.weight = new_shared.weight

    @property
    def n_parameters(self) -> str:
        num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
        total = num_params(self.parameters())
        emb = num_params(self.model.shared.parameters())
        return (f"#Embedding params: {emb}\n"
                f"#Non-embedding params: {total - emb}\n"
                f"#Total trainable params: {total}\n")

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss, outputs.logits

    def generate(self, input_ids: torch.Tensor,
                 attention_mask: Optional[torch.Tensor] = None,
                 num_beams: int = 20, **kwargs):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=5,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            **kwargs,
        )