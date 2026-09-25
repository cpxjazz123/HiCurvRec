"""HG_Rec_Curv_V5 — Stage 3 LorentzEmbedding (sinh-based expmap0) + HAB.

R36 Poincaré-Minkowski-Lorentz 路径:
  v4 用 Poincaré 球 expmap0 = tanh(sqrt(c)*||v||/2) / (sqrt(c)*||v||/2) * v
  当 ||v||=11.26 (T5 默认 init), sqrt(c)*||v||/2 = 11.26/2 = 5.63, tanh 完全饱和 (tanh(5.63)≈1.0),
  所以 expmap(v) 卡在 Poincaré 球边缘 ||expmap(v)||=R/sqrt(2)=1.414, 不在球内 → 退化为 norm clipping, 几何失效.

  v5 改用 Lorentz hyperboloid expmap0 = sinh(sqrt(c)*||v||) / (sqrt(c)*||v||) * v
  sinh 永远不饱和, 对 ||v||=11.26 仍给出有意义的 x_spatial ∈ ℝ^d.
  几何上等价于 tangent vector → Lorentz hyperboloid → 投影回 spatial dims (Chami 2019 HGCN 标准做法).

  HAB 保持 v2/v4 同款 (frozen Dbar + learnable λ per layer).
  曲率正则项改为监控 Lorentz spatial norm (而不是 Poincaré expmap norm).
"""
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config
from typing import Optional, Dict, Any


class LorentzEmbedding(nn.Module):
    """Lorentz hyperboloid embedding (sinh-based expmap0, take spatial part only).

    Tangent vector v ∈ R^d → Lorentz hyperboloid point x ∈ R^{d+1}
      x_0 = cosh(sqrt(c) * ||v||)
      x_spatial = sinh(sqrt(c) * ||v||) / (sqrt(c) * ||v||) * v  ∈ R^d
    Output: x_spatial ∈ R^d (same shape as v, fed to T5 as embedding).

    For v=0 (PAD), x_spatial = 0.
    For v large (e.g., ||v||=11.26), sinh(11.26*sqrt(0.5))=sinh(7.96)≈ 2953, factor = sinh/||v|| ≈ 262.
    So x_spatial has norm ≈ 262 * 11.26 ≈ 2949. This is LARGE but mathematically correct —
    we're using raw Lorentz representation, not bounded Poincaré ball.

    To stay numerically stable, we also scale by a learnable temperature τ:
      output = x_spatial * τ
    Default τ=0.01 keeps output in similar magnitude to T5 default (~11.26). τ is learnable,
    so model can adjust.

    v5 also adds a curvature regularizer (computed via Lorentz distance norm), NOT via
    Euclidean expmap_norm (which was the v4 mistake).
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int,
                 kappa: float = 0.5, init_std: float = 0.02, tau_init: float = 0.01):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        nn.init.normal_(self.embedding.weight, std=init_std)
        with torch.no_grad():
            self.embedding.weight[padding_idx].zero_()
        # κ registered as buffer (FIXED, not learned)
        self.register_buffer("kappa", torch.tensor(float(kappa)))
        self.register_buffer("log_kappa", torch.tensor(float(kappa)).log())
        # learnable temperature τ to scale Lorentz output back to T5-magnitude
        self.log_tau = nn.Parameter(torch.tensor(float(tau_init)).log())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        v = self.embedding(input_ids)
        c = self.kappa
        sqrt_c = c.sqrt()
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-7)
        # Lorentz expmap0: x_spatial = sinh(sqrt(c)*||v||) / (sqrt(c)*||v||) * v
        sqrt_c_norm = sqrt_c * v_norm
        factor = torch.sinh(sqrt_c_norm) / sqrt_c_norm
        x_spatial = v * factor
        # scale by τ to keep magnitudes similar to T5 default init
        tau = self.log_tau.exp()
        return x_spatial * tau

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


class HG_Rec_Curv_V5(nn.Module):
    """HG_Rec + LorentzEmbedding (sinh-based, no saturation) — for v5.

    HAB installed separately via install_hab(model, hab_module, layer_id_lut).
    曲率正则项在 train script 中通过额外 loss 加入:
        L_total = L_ce + λ_reg * mean((||Lorentz_x|| - target_norm)²)
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
        new_shared = LorentzEmbedding(
            num_embeddings=config['vocab_size'],
            embedding_dim=config['d_model'],
            padding_idx=config['pad_token_id'],
            kappa=0.5,
            init_std=0.02,
            tau_init=0.01,
        )
        with torch.no_grad():
            # use T5 default init for backward consistency with baseline HG-Rec
            new_shared.embedding.weight.data.copy_(self.model.shared.weight.data)
            # keep PAD = 0
            new_shared.embedding.weight[config['pad_token_id']].zero_()
        self.model.shared = new_shared
        # 同步 encoder/decoder embed_tokens (T5 内部缓存, 必须显式同步)
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