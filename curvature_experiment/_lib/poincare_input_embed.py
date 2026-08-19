"""v41 (Issue261): Stage 3 输入 SID embedding 端 Poincaré projection.

与 v40 (PoincareT5LayerNorm) 的区别:
- v40: 改 T5 内部 LayerNorm (替换 attention/FFN block 内的 T5LayerNorm)
- v41: 改 T5 输入端 shared embedding (SID token id → embedding → Poincaré projection)

实现:
- 取 T5 `shared.weight` (vocab_size × d_model) 作为欧氏 embedding
- 每次 forward 时, 把 shared embedding 通过 logmap0 / 内部 linear / expmap0 投影到 Poincaré ball
- weight 起点与原欧氏 embedding 完全一致 (clamp_max 1/sqrt(c) 后保持 norm<1 边界)

R36 合规 — Stage 3 端几何变换, 无 LR/dropout/wd sweep.
"""
import math
import torch
import torch.nn as nn
from torch import Tensor


def _expmap0_t(x: Tensor, c: float) -> Tensor:
    """欧氏 → Poincaré ball."""
    sqrt_c = torch.sqrt(torch.tensor(c, device=x.device, dtype=x.dtype).clamp_min(1e-10))
    norm_x = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    max_norm = (1.0 / sqrt_c - 1e-5)
    norm_x_clamp = norm_x.clamp_max(max_norm)
    factor = torch.tanh(sqrt_c * norm_x_clamp) / (sqrt_c * norm_x)
    return factor * x


def _logmap0_t(x: Tensor, c: float) -> Tensor:
    """Poincaré ball → 欧氏."""
    sqrt_c = torch.sqrt(torch.tensor(c, device=x.device, dtype=x.dtype).clamp_min(1e-10))
    norm_x = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    max_norm = (1.0 / sqrt_c - 1e-5)
    norm_x_clamp = norm_x.clamp_max(max_norm)
    factor = torch.atanh(sqrt_c * norm_x_clamp) / (sqrt_c * norm_x)
    return factor * x


class PoincareInputEmbedding(nn.Module):
    """包装 T5 `shared.weight` 把欧氏 embedding 投影到 Poincaré ball.

    用法:
        embed_layer = T5.shared   # nn.Embedding(769, 128)
        poinc_embed = PoincareInputEmbedding(embed_layer, c=0.5)
        # 然后在 forward 时用 poinc_embed(token_ids) 替代 embed_layer(token_ids)
    """

    def __init__(self, base_embed: nn.Embedding, c: float = 0.5):
        super().__init__()
        self.base_embed = base_embed
        self.c = float(c)
        self.vocab_size = base_embed.num_embeddings
        self.d_model = base_embed.embedding_dim
        # 检查起点 norm 是否需要 normalize
        with torch.no_grad():
            emb = base_embed.weight.data
            emb_norms = emb.norm(dim=-1)
            max_norm = (1.0 / math.sqrt(self.c) - 1e-3)
            scale = torch.ones_like(emb_norms)
            exceed_mask = emb_norms > max_norm
            scale[exceed_mask] = max_norm / emb_norms[exceed_mask].clamp_min(1e-10)
            # base_embed.weight *= scale[:, None]  # 不修改 base_embed, 只在 forward 时缩放
        self.register_buffer("rescale", scale, persistent=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        """input_ids (B, L) → Poincaré ball embedding (B, L, d_model)."""
        emb = self.base_embed(input_ids)  # (B, L, d)
        # logmap0: 欧氏 → tanh space
        # 起点: rescale 缩放后, norm < 1/sqrt(c)
        emb_norm = emb.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        max_norm = (1.0 / math.sqrt(self.c) - 1e-5)
        # gather rescale for each token
        # rescale (vocab_size,)
        rescale_per_token = self.rescale[input_ids]  # (B, L)
        emb = emb * rescale_per_token.unsqueeze(-1)
        # logmap0 + internal identity (这里不加额外 linear, 保持 d_model 不变)
        x_tan = _logmap0_t(emb, self.c)
        # expmap0: tanh → Poincaré ball
        x_p = _expmap0_t(x_tan, self.c)
        return x_p


def replace_t5_shared_embedding(model, c: float = 0.5):
    """替换 T5.shared 为 PoincareInputEmbedding.

    必须保存 base_embed 引用以便 Stage 4 重新加载 ckpt 时能恢复结构。
    """
    shared = None
    for name, mod in model.named_modules():
        if name.endswith("shared") and isinstance(mod, nn.Embedding):
            shared = mod
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent = model.get_submodule(parent_name) if parent_name else model
            child_name = name.rsplit(".", 1)[-1]
            break
    if shared is None:
        return 0
    new_embed = PoincareInputEmbedding(shared, c=c)
    new_embed = new_embed.to(device=shared.weight.device, dtype=shared.weight.dtype)
    setattr(parent, child_name, new_embed)
    return 1