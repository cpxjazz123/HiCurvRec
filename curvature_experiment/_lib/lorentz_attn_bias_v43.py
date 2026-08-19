"""v43 (Issue263): Stage 3 T5 cross-attention Lorentz inner product bias.

设计: 在 T5 decoder 第一层 cross-attention 的 compute_bias 上, 附加 Lorentz 内积 bias
(Chen 2022 Lorentz Transformer 风格). 与 v42 (encoder self-attention 欧氏距离 bias) 不同:

- v42: 改 encoder self-attention, 用 ||q-k||^2 (Poincaré 距离近似)
- v43: 改 decoder cross-attention, 用 -<q, k>_L = -(-q_0 k_0 + q_{1..d} · k_{1..d}) (Lorentz 内积)

Lorentz 内积定义 (M={x∈R^{d+1}: -x_0^2 + x_1^2 + ... + x_d^2 = -1/c, x_0 > 0}, c>0):
  <u, v>_L = -u_0 * v_0 + sum(u_{1..d} * v_{1..d})

Lorentz 距离: d_L(u,v) = sqrt(c) * arccosh(-<u,v>_L * c)   (越大越远)

将 q/k 通过 expmap0 投到 Lorentz 双曲面:
  q_L = expmap0_QL(q, c): 把欧氏向量 → Lorentz 双曲面上的点

bias = lambda * d_L(q_L, k_L) (正向 bias = 距离越远分数越低)

Stage 4 评估时也要启用 (否则 ckpt 行为不一致, R51 baseline 不可比).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


HAB_V43_LAMBDA = 0.1
HAB_V43_C = 1.0  # Lorentz 标准曲率 c=1 (Chen 2022 默认)


def _expmap0_q_to_lorentz(x: torch.Tensor, c: float) -> torch.Tensor:
    """欧氏空间 → Lorentz 双曲面 (M={x_0^2 - x_1..d^2 = 1/c, x_0 > 0}).

    使用 Chen 2022 公式:
      u_0 = sqrt(1/c + ||x||^2 + 1/c * ||x||^2)
      u_{1..d} = x
    """
    d_norm2 = (x * x).sum(dim=-1, keepdim=True)
    sqrt_c = c ** 0.5
    u_0 = torch.sqrt(1.0 / c + d_norm2)
    return torch.cat([u_0, x], dim=-1)


def _lorentz_inner(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Lorentz 内积: -u_0 v_0 + sum(u_{1..d} v_{1..d}).

    u, v: (..., d+1) → (...,)
    """
    return -u[..., 0] * v[..., 0] + (u[..., 1:] * v[..., 1:]).sum(dim=-1)


def _lorentz_dist(u: torch.Tensor, v: torch.Tensor, c: float) -> torch.Tensor:
    """Lorentz 距离: sqrt(c) * arccosh(-<u,v>_L * c)."""
    inner = _lorentz_inner(u, v)  # (...,)
    arg = (-c * inner).clamp_min(1.0 + 1e-7)  # arccosh 定义域 ≥1
    return c ** 0.5 * torch.acosh(arg)


def wrap_t5_cross_attention_with_lorentz_bias(model, c: float = HAB_V43_C, lam: float = HAB_V43_LAMBDA) -> int:
    """包装 T5 decoder 第一层 cross-attention, 加 Lorentz 距离 bias.

    Returns: 替换 attention 模块数量.
    """
    from transformers.models.t5.modeling_t5 import T5Attention

    n_replaced = 0
    decoder_layers = getattr(model.decoder, "block", None)
    if decoder_layers is None:
        raise RuntimeError("T5 model.decoder.block not found — wrong model arch?")

    for layer in decoder_layers[:1]:  # 只第一层注入
        # T5DecoderLayer.layer[1] 是 cross-attention (0=自注意, 1=cross-attn)
        cross_attn = layer.layer[1].EncDecAttention
        if not hasattr(cross_attn, "_v43_wrapped"):
            _wrap_one_cross_attention(cross_attn, c=c, lam=lam)
            cross_attn._v43_wrapped = True
            n_replaced += 1

    return n_replaced


def _wrap_one_cross_attention(attn, c: float, lam: float):
    """包装一个 T5 cross-attention, 在 compute_bias 注入 Lorentz 距离 bias."""
    original_compute_bias = attn.compute_bias

    def compute_bias_with_lorentz(self, query_length, key_length, device=None, past_seen_tokens=None):
        try:
            bias = original_compute_bias(query_length, key_length, device)
        except TypeError:
            try:
                bias = original_compute_bias(query_length, key_length, past_seen_tokens)
            except TypeError:
                bias = original_compute_bias(query_length, key_length)

        if not hasattr(self, "cross_query_states") or not hasattr(self, "cross_key_states"):
            return bias

        q = self.cross_query_states  # (B, num_heads, q_len, head_dim)
        k = self.cross_key_states    # (B, num_heads, k_len, head_dim)

        B, H, qlen, dim = q.shape
        _, _, klen, _ = k.shape

        # 投影到 Lorentz 双曲面 (把 head_dim 维向量映射到 d+1 维)
        q_flat = q.reshape(B * H, qlen, dim)
        k_flat = k.reshape(B * H, klen, dim)
        q_lor = _expmap0_q_to_lorentz(q_flat, c)  # (B*H, qlen, dim+1)
        k_lor = _expmap0_q_to_lorentz(k_flat, c)  # (B*H, klen, dim+1)

        # Lorentz 成对距离 (B*H, qlen, klen)
        # 用广播: q_lor[..., None, :] vs k_lor[..., None, :, :]
        # 计算量: B*H * qlen * klen * (dim+1) FLOPs — 与欧氏距离同阶, 不爆内存
        # 直接扩展 + 内积:
        q_exp = q_lor.unsqueeze(2)  # (B*H, qlen, 1, dim+1)
        k_exp = k_lor.unsqueeze(1)  # (B*H, 1, klen, dim+1)
        inner = -q_exp[..., 0] * k_exp[..., 0] + (q_exp[..., 1:] * k_exp[..., 1:]).sum(dim=-1)
        arg = (-c * inner).clamp_min(1.0 + 1e-7)
        dist = (c ** 0.5) * torch.acosh(arg)  # (B*H, qlen, klen)

        # 归一化: 在每个 (B, H) 上 z-normalize
        mean_d = dist.mean(dim=(1, 2), keepdim=True)
        std_d = dist.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        norm_dist = (dist - mean_d) / std_d
        norm_dist = norm_dist.reshape(B, H, qlen, klen)

        # bias shape: (1, 1, qlen, klen)
        bias = bias - lam * norm_dist

        return bias

    def _compat_compute_bias(ql, kl, device=None, past_seen_tokens=None):
        return compute_bias_with_lorentz(attn, ql, kl, device, past_seen_tokens)
    attn.compute_bias = _compat_compute_bias


def install_v43_hook(model):
    """在 model 上注册 forward hook, 在 cross-attention forward 时把 query/key states 存入模块."""
    decoder_block = model.decoder.block[0]
    cross_attn = decoder_block.layer[1].EncDecAttention

    original_forward = cross_attn.forward

    def forward_with_capture(self, hidden_states, mask=None, key_value_states=None, position_bias=None, *args, **kwargs):
        batch_size, seq_length = hidden_states.shape[:2]

        real_q = self.q(hidden_states)
        if key_value_states is None:
            raise RuntimeError("Cross-attn must have key_value_states (encoder hidden)")
        real_k = self.k(key_value_states)
        real_v = self.v(key_value_states)

        # 保存到 attn 模块供 compute_bias 使用
        self.cross_query_states = real_q.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        self.cross_key_states = real_k.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        return original_forward(hidden_states, mask=mask, key_value_states=key_value_states,
                                position_bias=position_bias, *args, **kwargs)

    cross_attn.forward = lambda *a, **kw: forward_with_capture(cross_attn, *a, **kw)