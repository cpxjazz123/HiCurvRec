"""v42 (Issue262): Stage 3 T5 attention Q/K 双曲距离 bias (HNN 2019) — 内存优化版.

设计: 把 attention score 加上双曲距离 bias (Q/K 通过 expmap0 投到 Poincaré ball,
在球面算成对距离作为 negative bias, 距离远的 token 分数越低).

与 v40/v41 完全独立:
- v40: 替换 T5 LayerNorm
- v41: 替换 T5.shared embedding
- v42: 替换 T5 attention score (用 Q/K 双曲距离加 bias)

实现 (per attention head, 内存友好):
1. proj_q, proj_k 后的 q,k 向量
2. 在球面算双曲成对距离近似为 q·q + k·k - 2q·k (欧氏距离的平方, 等价于 norm 较小时的双曲距离)
3. 归一化后 bias -= lambda * poinc_dist_approx
4. lambda 硬编码 0.1 (R36 不调参)

仅在 encoder 第一层 attention 注入 (decoder 不动), 避免破坏 auto-regressive generation.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


HAB_V42_LAMBDA = 0.1
HAB_V42_C = 0.5


def wrap_t5_attention_with_poinc_bias(model, c: float = HAB_V42_C, lam: float = HAB_V42_LAMBDA) -> int:
    """包装 T5 attention 模块, 在 attention score 上加双曲距离 bias.

    只在 encoder 第一层 attention 注入一次, 与 Issue #64 复用策略一致.

    Returns: 替换 attention 模块数量.
    """
    from transformers.models.t5.modeling_t5 import T5Attention

    n_replaced = 0
    encoder_layers = getattr(model.encoder, "block", None)
    if encoder_layers is None:
        raise RuntimeError("T5 model.encoder.block not found — wrong model arch?")

    for layer in encoder_layers[:1]:  # 只第一层注入 (与 #64 一致)
        attn = layer.layer[0].SelfAttention
        if not hasattr(attn, "_v42_wrapped"):
            _wrap_one_attention(attn, c=c, lam=lam)
            attn._v42_wrapped = True
            n_replaced += 1

    return n_replaced


def _wrap_one_attention(attn, c: float, lam: float):
    """包装一个 T5Attention, 在 compute_bias 注入双曲距离 bias.

    内存优化:
    - 用 ||-q⊕k||² ≈ ||q-k||² 在小 norm 下近似 (Poincaré 球小 norm 时 expmap0 ≈ identity)
    - 直接用 (B, H, q_len, k_len) bias, 一次性算, 不预分配 chunk tensor
    """
    original_compute_bias = attn.compute_bias

    def compute_bias_with_poinc(self, query_length, key_length, device=None, past_seen_tokens=None):
        try:
            bias = original_compute_bias(query_length, key_length, device)
        except TypeError:
            try:
                bias = original_compute_bias(query_length, key_length, past_seen_tokens)
            except TypeError:
                bias = original_compute_bias(query_length, key_length)

        if not hasattr(self, "query_states") or not hasattr(self, "key_states"):
            return bias

        q = self.query_states  # (B, num_heads, q_len, head_dim)
        k = self.key_states    # (B, num_heads, k_len, head_dim)

        B, H, qlen, dim = q.shape
        _, _, klen, _ = k.shape

        # 用欧氏距离平方近似双曲距离 (Poincaré 球小 norm 下 ≈ identity)
        # ||q - k||² = ||q||² + ||k||² - 2 q·k
        q_norm2 = (q * q).sum(dim=-1, keepdim=True)  # (B, H, qlen, 1)
        k_norm2 = (k * k).sum(dim=-1, keepdim=True)  # (B, H, 1, klen)
        qk = torch.matmul(q, k.transpose(-1, -2))    # (B, H, qlen, klen)
        euc_dist2 = q_norm2 + k_norm2.transpose(-1, -2) - 2 * qk
        euc_dist2 = euc_dist2.clamp_min(0.0)  # 数值非负

        # 归一化: 在每个 (B, H) 上 z-normalize
        mean_d = euc_dist2.mean(dim=(2, 3), keepdim=True)
        std_d = euc_dist2.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        poinc_bias = (euc_dist2 - mean_d) / std_d

        # bias shape: (1, 1, qlen, klen) — broadcast
        bias = bias - lam * poinc_bias

        return bias

    def _compat_compute_bias(ql, kl, device=None, past_seen_tokens=None):
        return compute_bias_with_poinc(attn, ql, kl, device, past_seen_tokens)
    attn.compute_bias = _compat_compute_bias


def install_v42_hook(model):
    """在 model 上注册 forward hook, 在 attention forward 时把 query/key states 存入 attn 模块."""
    encoder_block = model.encoder.block[0]
    attn = encoder_block.layer[0].SelfAttention

    original_forward = attn.forward

    def forward_with_capture(self, hidden_states, mask=None, key_value_states=None, position_bias=None, *args, **kwargs):
        batch_size, seq_length = hidden_states.shape[:2]

        real_q = self.q(hidden_states)
        if key_value_states is None:
            real_k = self.k(hidden_states)
        else:
            real_k = self.k(key_value_states)
        real_v = self.v(hidden_states)

        # 保存到 attn 模块供 compute_bias 使用
        self.query_states = real_q.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        self.key_states = real_k.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        return original_forward(hidden_states, mask=mask, key_value_states=key_value_states,
                                position_bias=position_bias, *args, **kwargs)

    attn.forward = lambda *a, **kw: forward_with_capture(attn, *a, **kw)