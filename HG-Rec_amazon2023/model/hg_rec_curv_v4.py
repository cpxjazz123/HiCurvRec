"""HG_Rec_Curv_V4 — Stage 3 fixed κ Poincaré embedding + curvature regularizer + HAB.

R36 #2 + #3 (稳定版):
  (a) Input embedding: PoincareEmbeddingFixed (κ FIXED at 0.5, no drift) + curvature regularizer
      - 曲率正则: L_reg = λ_reg * (||expmap(v)|| - target_norm)²
      - 强制 embedding 落在 Poincaré ball 中段 (target_norm=0.5)
      - 让模型自由学 embeddings, 但保持在双曲几何域内
  (b) Encoder attention: HAB frozen Dbar + learnable λ (与 v2 相同)

v3 教训: free κ 漂移 1.0→1.016→0.955, 引入方差拖慢 HAB 主线. v4 改为 κ 固定 + 曲率正则,
  既保持双曲几何 (κ=0.5 ≠ 1) 又避免 κ 不稳定扰动.
"""
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config
from typing import Optional, Dict, Any


class PoincareEmbeddingFixed(nn.Module):
    """Poincaré ball embedding with FIXED curvature κ.

    v4: κ is NOT learned (avoids v3's drift). κ=0.5 hardcoded.
    expmap0 projection enforces hyperbolic geometry without learnable parameters.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int,
                 kappa: float = 0.5, init_std: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        nn.init.normal_(self.embedding.weight, std=init_std)
        with torch.no_grad():
            self.embedding.weight[padding_idx].zero_()
        # FIXED κ (not learned)
        self.register_buffer("kappa", torch.tensor(float(kappa)))
        # 也存 log_kappa 给外部读取
        self.register_buffer("log_kappa", torch.tensor(float(kappa)).log())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        v = self.embedding(input_ids)
        k = self.kappa
        sqrt_k = k.sqrt()
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-7)
        factor = torch.tanh(sqrt_k * v_norm / 2.0) / (sqrt_k * v_norm)
        return v * factor

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
        return self.embedding.weight


class HG_Rec_Curv_V4(nn.Module):
    """HG_Rec + PoincareEmbeddingFixed (κ=0.5) — for v4.

    HAB installed separately via install_hab(model, hab_module, layer_id_lut).
    曲率正则项在 train_HG_Rec_curv_hab_v4_ddp.py 中通过额外的 loss 加入:
        L_total = L_ce + λ_reg * mean(||expmap(v)|| - target_norm)²
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
        new_shared = PoincareEmbeddingFixed(
            num_embeddings=config['vocab_size'],
            embedding_dim=config['d_model'],
            padding_idx=config['pad_token_id'],
            kappa=0.5,
            init_std=0.1,
        )
        with torch.no_grad():
            new_shared.embedding.weight.data.copy_(self.model.shared.weight.data)
        self.model.shared = new_shared
        # v3 fix: 同步 encoder/decoder embed_tokens
        self.model.encoder.embed_tokens = new_shared
        self.model.decoder.embed_tokens = new_shared
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