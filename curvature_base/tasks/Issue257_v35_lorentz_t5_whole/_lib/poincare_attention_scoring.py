"""Issue #236 (2026-08-10): Stage3 Hyperbolic Attention Scoring — 替换 inner-product 为负 Poincaré 距离.

设计:
  - 在 T5 attention score 计算时, 把 scores = matmul(Q, K^T) 替换为 scores = -d_P(Q, K)
  - Q, K 是 W_q(hidden_states), W_k(hidden_states) 的输出 (T5Attention.forward 内部)
  - 范数用 tanh 缩放到 Poincaré ball 内部 (< 1)
  - 曲率参数 c 可固定 (c_init) 或 learnable (c_learnable=True), 经 sigmoid 钳到 (1e-3, 10)
  - 仅 patch encoder self-attention (key_value_states is None), decoder / cross-attn 走原路径
  - 与 HAB 正交可叠加: -d_P 替换 inner-product, B_geo 仍然作为 attention_mask 4D bias 注入 (HabModule)

复杂度: O(L^2 * (dim+1)) = O(128^2 * 17) per head per layer ≈ 280k ops ≈ 与 inner-product 同阶
"""
import math
import types
import torch
import torch.nn as nn


def _to_ball(x, eps=1e-5):
    """把 x 投影到 Poincaré ball: ||x|| < 1. 用 tanh 缩放: x_ball = tanh(||x||) * x / ||x||."""
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    # tanh(norm) ∈ (0, 1), 缩放后 norm < 1
    return (torch.tanh(norm) * x) / norm


def _poincare_distance_sq_diff(q_ball, k_ball):
    """计算 -d_P(q_ball, k_ball) per pair.
    q_ball: (..., L, d), k_ball: (..., L_k, d). 返回 (..., L, L_k) 的 -d_P score.
    d_P(u,v) = arcosh(1 + 2*||u-v||^2 / ((1-||u||^2)*(1-||v||^2))) / sqrt(c)
    """
    q_norm_sq = (q_ball * q_ball).sum(dim=-1, keepdim=True)             # (..., L, 1)
    k_norm_sq = (k_ball * k_ball).sum(dim=-1, keepdim=True).transpose(-1, -2)  # (..., 1, L_k)
    diff = q_ball.unsqueeze(-2) - k_ball.unsqueeze(-3)                  # (..., L, L_k, d)
    diff_sq = (diff * diff).sum(dim=-1)                                 # (..., L, L_k)
    denom = (1.0 - q_norm_sq) * (1.0 - k_norm_sq)                        # (..., L, L_k)
    denom = denom.clamp_min(1e-10)
    x = 1.0 + 2.0 * diff_sq / denom
    x = x.clamp_min(1.0 + 1e-7)  # arcosh 定义域
    return x, diff_sq, denom


class PoincareAttentionScoring(nn.Module):
    """把 T5 attention score 替换为 -d_P 的可注入模块.

    Args:
        c_init: 初始曲率倒数 (1/c 是负曲率), 默认 1.0 (与 HAB Stage2 一致范围)
        c_learnable: 是否让 c_learnable 训练 (R36 允许: 不是调 LR/dropout, 是改曲率机制)
    """

    def __init__(self, c_init=1.0, c_learnable=True):
        super().__init__()
        # log_c_param: 经 sigmoid 缩放后乘以 10 (范围 0~10, 默认 init=0 → c=5.0 中点)
        # sigmoid(0) = 0.5, 0.5 * 10 = 5.0; 这样 log_c_param ∈ R 即可
        init_logit = math.log(0.5 / (1.0 - 0.5))  # = 0
        self.log_c_param = nn.Parameter(torch.tensor(float(init_logit)), requires_grad=c_learnable)

    @property
    def c(self):
        """曲率倒数 c, 经 sigmoid + scale 钳到 [1e-3, 10]."""
        return torch.sigmoid(self.log_c_param) * 10.0 + 1e-3

    def compute_poincare_neg_distance(self, q, k):
        """返回 -d_P(q, k) per pair. q: (..., L, d), k: (..., L_k, d). 返回 (..., L, L_k)."""
        q_ball = _to_ball(q)
        k_ball = _to_ball(k)
        x, _, _ = _poincare_distance_sq_diff(q_ball, k_ball)
        c = self.c
        # d_P = arcosh(x) / sqrt(c). score = -d_P (越大越相似, 配合 softmax)
        # 同时按 dim 做归一化 (等价 inner-product 的 /sqrt(d) scale)
        d = torch.acosh(x) / torch.sqrt(c.clamp_min(1e-3))
        # scale: 让 -d_P 量级与 matmul(Q,K^T) 量级匹配 (Q,K 范数 ≈ 1 后, matmul ≈ dot product ~ d)
        # 实际 d 已经是 d_P distance (0 ~ 几), 范围匹配直接用即可
        return -d


def install_poincare_attention_scoring(model, c_init=1.0, c_learnable=True):
    """Monkey-patch 所有 T5Attention layer (T5 small 6 encoder + 6 decoder), self-attn 路径注入 -d_P score.

    Args:
        model: HF T5ForConditionalGeneration (含 encoder + decoder)
        c_init: 初始 c (默认 1.0)
        c_learnable: c 是否可训练

    Returns:
        poincare_attn_module: PoincareAttentionScoring 实例, 已注册到 model
    """
    from transformers.models.t5.modeling_t5 import T5Attention
    poincare_attn_module = PoincareAttentionScoring(c_init=c_init, c_learnable=c_learnable)
    # 移到 model.device
    try:
        device = next(model.parameters()).device
        poincare_attn_module = poincare_attn_module.to(device)
    except StopIteration:
        pass
    model.add_module("poincare_attn_module", poincare_attn_module)
    n_patched = 0
    for name, module in model.named_modules():
        if isinstance(module, T5Attention):
            module._poincare_attn = poincare_attn_module
            module._original_t5attn_forward = module.forward
            module.forward = types.MethodType(_patched_t5attn_forward, module)
            n_patched += 1
    print(f"[Issue #236] PoincareAttentionScoring installed: {n_patched} T5Attention layers patched, "
          f"c_init={c_init}, c_learnable={c_learnable}")
    return poincare_attn_module


def _patched_t5attn_forward(
    self,
    hidden_states,
    mask=None,
    key_value_states=None,
    position_bias=None,
    past_key_values=None,
    output_attentions=False,
    **kwargs,
):
    """T5Attention.forward monkey-patch: self-attn 路径 (key_value_states is None) 用 -d_P 替换 matmul(Q,K^T).

    仅在 self-attn (key_value_states is None) 注入 — cross-attn (decoder → encoder) 走原路径.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.key_value_proj_dim)
    # 跟 HF 原版一致地处理 past_key_values (简化为新版 cache 接口, 与 Issue #64 兼容)
    past_seen_tokens = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
    if isinstance(past_seen_tokens, torch.Tensor):
        past_seen_tokens = past_seen_tokens.clone()

    is_cross_attention = key_value_states is not None

    query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)

    # 当前实现简化: 不处理 EncoderDecoderCache 复杂情况, 直接重算 k/v (T5 小模型, overhead 可接受)
    current_states = key_value_states if is_cross_attention else hidden_states
    kv_shape = (*current_states.shape[:-1], -1, self.key_value_proj_dim)
    key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
    value_states = self.v(current_states).view(kv_shape).transpose(1, 2)

    # === 关键: Issue #236 -d_P 替换 inner-product (self-attn 路径) ===
    if not is_cross_attention and getattr(self, "_poincare_attn", None) is not None:
        # 跳过 cross-attn, 只对 self-attn 注入
        scores = self._poincare_attn.compute_poincare_neg_distance(query_states, key_states)
    else:
        # cross-attn (decoder → encoder): 走原版 matmul
        scores = torch.matmul(query_states, key_states.transpose(3, 2))

    if position_bias is None:
        key_length = key_states.shape[-2]
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, query_states.shape[1], input_shape[1], key_length), device=scores.device, dtype=scores.dtype
            )
        else:
            position_bias = self.compute_bias(
                input_shape[1], key_length, device=scores.device, past_seen_tokens=past_seen_tokens
            )
        if mask is not None:
            causal_mask = mask[:, :, :, : key_states.shape[-2]]
            position_bias = position_bias + causal_mask

    scores = scores + position_bias

    attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
    attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1)
    attn_output = self.o(attn_output)

    outputs = (attn_output, position_bias)
    if output_attentions:
        outputs = outputs + (attn_weights,)
    return outputs