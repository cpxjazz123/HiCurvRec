"""HG_Rec_Curv_v3 — Stage 3 κ learning (fixed v1 plateau) + HAB (v2).

R36 (#2+#3 组合): 双重曲率机制注入
  (a) Input embedding: Poincaré expmap0 + learnable κ, embedding std=0.1 (5x v1)
      → κ gradient signal 显著增强 (||v|| ≈ 1.1, expmap factor ≈ 0.45)
  (b) Encoder attention: HAB frozen Dbar from Stage2 ckpt + learnable λ per layer

两个曲率机制独立:
  - κ ∈ [0.01, 10]   控制输入嵌入双曲投影强度
  - λ ∈ [0, 0.20]   控制 attention bias 几何强度

T5 lm_head 保持 tied Euclidean weight (与 baseline 兼容).
"""
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config
from typing import Optional, Dict, Any


class PoincareEmbeddingV3(nn.Module):
    """Poincaré ball embedding with learnable curvature κ.

    v3 fix vs v1:
      - Init std = 0.1 (vs 0.02) → embedding norms ||v|| ≈ 1.1 (vs 0.226)
      - With κ=1, √κ ||v||/2 ≈ 0.55 → tanh(0.55) ≈ 0.50 → factor ≈ 0.45
      - κ gradient via tanh derivative (≈ 0.71 at 0.55) → meaningful κ signal
      - κ clamp: [0.05, 8.0] to allow κ to wander (v1 used [0.01, 10])
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int,
                 kappa_init: float = 1.0, init_std: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        # v3 fix: std=0.1 (was 0.02) so expmap is non-trivial
        nn.init.normal_(self.embedding.weight, std=init_std)
        with torch.no_grad():
            self.embedding.weight[padding_idx].zero_()
        self.log_kappa = nn.Parameter(torch.tensor(float(kappa_init)).log())

    @property
    def kappa(self) -> torch.Tensor:
        # κ ∈ [0.05, 8.0] — allow κ to wander; prevent degenerate flat (κ→0) regime
        return self.log_kappa.exp().clamp(min=0.05, max=8.0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        v = self.embedding(input_ids)  # (B, L, d), Euclidean
        k = self.kappa
        sqrt_k = k.sqrt()
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-7)
        # expmap0: v → v * tanh(√κ ||v||/2) / (√κ ||v||)
        factor = torch.tanh(sqrt_k * v_norm / 2.0) / (sqrt_k * v_norm)
        return v * factor  # now ||x|| < 1/√κ

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


class HG_Rec_Curv_V3(nn.Module):
    """HG_Rec + PoincareEmbeddingV3 (κ learnable) — for v3.

    HAB installed separately via install_hab(model, hab_module, layer_id_lut).
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
        new_shared = PoincareEmbeddingV3(
            num_embeddings=config['vocab_size'],
            embedding_dim=config['d_model'],
            padding_idx=config['pad_token_id'],
            kappa_init=1.0,
            init_std=0.1,
        )
        with torch.no_grad():
            new_shared.embedding.weight.data.copy_(self.model.shared.weight.data)
        self.model.shared = new_shared
        # v3 fix: T5 encoder/decoder 各自缓存 embed_tokens 引用, 必须显式同步,
        # 否则 forward 路径仍走原 nn.Embedding → PoincareEmbeddingV3 收不到梯度
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