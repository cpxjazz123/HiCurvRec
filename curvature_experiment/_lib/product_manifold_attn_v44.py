"""v44 (Issue264): Stage 3 T5 decoder cross-attention product manifold bias.

设计: 在 T5 decoder 第一层 cross-attention 的 compute_bias 上, 同时附加:
  1. Poincaré 距离 bias (v42 公式, c=0.5, lambda_p=0.05):
     bias -= lambda_p * normalize(||q-k||^2)
  2. Lorentz 距离 bias (v43 公式, c=1.0, lambda_l=0.05):
     bias -= lambda_l * normalize(sqrt(c)*arccosh(-<q_L, k_L>_L * c))

两种曲率机制组合 — R36 曲率机制组合 (Stage 3 端).

Stage 4 评估时也要启用 (否则 ckpt 行为不一致, R51 baseline 不可比).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from _lib.lorentz_attn_bias_v43 import _expmap0_q_to_lorentz, _lorentz_inner


HAB_V44_LAMBDA_POINC = 0.05  # 减半避免叠加过强
HAB_V44_LAMBDA_LORENTZ = 0.05  # 减半避免叠加过强
HAB_V44_C_POINC = 0.5
HAB_V44_C_LORENTZ = 1.0


def _lorentz_dist_norm(q_lor: torch.Tensor, k_lor: torch.Tensor, c: float) -> torch.Tensor:
    """Lorentz 成对距离 (B*H, qlen, klen), 在每 (B, H) 上 z-normalize.

    复用 v43 内积计算 + 加 z-normalize.
    """
    q_exp = q_lor.unsqueeze(2)  # (B*H, qlen, 1, dim+1)
    k_exp = k_lor.unsqueeze(1)  # (B*H, 1, klen, dim+1)
    inner = -q_exp[..., 0] * k_exp[..., 0] + (q_exp[..., 1:] * k_exp[..., 1:]).sum(dim=-1)
    arg = (-c * inner).clamp_min(1.0 + 1e-7)
    dist = (c ** 0.5) * torch.acosh(arg)  # (B*H, qlen, klen)
    mean_d = dist.mean(dim=(1, 2), keepdim=True)
    std_d = dist.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return (dist - mean_d) / std_d


def wrap_t5_cross_attention_with_product_bias(
    model, c_p: float = HAB_V44_C_POINC, lam_p: float = HAB_V44_LAMBDA_POINC,
    c_l: float = HAB_V44_C_LORENTZ, lam_l: float = HAB_V44_LAMBDA_LORENTZ,
) -> int:
    """包装 T5 decoder 第一层 cross-attention, 同时注入 Poincaré + Lorentz bias.

    Returns: 替换 attention 模块数量.
    """
    from transformers.models.t5.modeling_t5 import T5Attention

    n_replaced = 0
    decoder_layers = getattr(model.decoder, "block", None)
    if decoder_layers is None:
        raise RuntimeError("T5 model.decoder.block not found — wrong model arch?")

    for layer in decoder_layers[:1]:  # 只第一层
        cross_attn = layer.layer[1].EncDecAttention
        if not hasattr(cross_attn, "_v44_wrapped"):
            _wrap_one_cross_attention(cross_attn, c_p=c_p, lam_p=lam_p, c_l=c_l, lam_l=lam_l)
            cross_attn._v44_wrapped = True
            n_replaced += 1

    return n_replaced


def _wrap_one_cross_attention(attn, c_p: float, lam_p: float, c_l: float, lam_l: float):
    """包装一个 T5 cross-attention, 同时注入 Poincaré + Lorentz bias."""
    original_compute_bias = attn.compute_bias

    def compute_bias_with_product(self, query_length, key_length, device=None, past_seen_tokens=None):
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

        # === Poincaré 距离 bias (v42 公式) ===
        # 用 ||q-k||^2 = ||q||^2 + ||k||^2 - 2*q·k (matmul 形式, 无全张量)
        q_flat = q.reshape(B * H, qlen, dim)
        k_flat = k.reshape(B * H, klen, dim)
        q_norm2 = (q_flat * q_flat).sum(dim=-1, keepdim=True)  # (B*H, qlen, 1)
        k_norm2 = (k_flat * k_flat).sum(dim=-1, keepdim=True)  # (B*H, klen, 1)
        qk = torch.matmul(q_flat, k_flat.transpose(-1, -2))  # (B*H, qlen, klen)
        euc_dist2 = q_norm2 + k_norm2.transpose(-1, -2) - 2 * qk  # (B*H, qlen, klen)
        # z-normalize (per B*H)
        mean_d = euc_dist2.mean(dim=(1, 2), keepdim=True)
        std_d = euc_dist2.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        norm_dist_p = (euc_dist2 - mean_d) / std_d  # (B*H, qlen, klen)
        norm_dist_p = norm_dist_p.reshape(B, H, qlen, klen)

        # === Lorentz 距离 bias (v43 公式) ===
        # expmap0 → Lorentz 双曲面 → arccosh 距离 → z-normalize
        q_lor = _expmap0_q_to_lorentz(q_flat, c_l)  # (B*H, qlen, dim+1)
        k_lor = _expmap0_q_to_lorentz(k_flat, c_l)  # (B*H, klen, dim+1)
        norm_dist_l = _lorentz_dist_norm(q_lor, k_lor, c_l).reshape(B, H, qlen, klen)

        # === Product manifold: 两种 bias 加权组合 ===
        bias = bias - lam_p * norm_dist_p - lam_l * norm_dist_l

        return bias

    def _compat_compute_bias(ql, kl, device=None, past_seen_tokens=None):
        return compute_bias_with_product(attn, ql, kl, device, past_seen_tokens)
    attn.compute_bias = _compat_compute_bias


def install_v44_hook(model):
    """在 model 上注册 forward hook, 在 cross-attention forward 时把 query/key states 存入模块.

    复用 v43 hook 安装逻辑 (cross-attention 路径相同).
    """
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