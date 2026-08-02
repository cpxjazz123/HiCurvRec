#!/usr/bin/env python3
"""Issue #25 共享解码函数 — Stage3 val 与 Stage4 canary 必须 import 同一函数.

Per #25 spec 严格单变量 (val 解码路径):
- 抽 shared function, Stage3 val 与 Stage4 canary/评估 import 同一函数, 不再各写一份
- 函数内部: 显式 decoder_input_ids (zeros(B,4) + 已预测 token 右移填充) + t5.model.lm_head
  + per-layer layer_ranges mask (-1e9) + argmax
- 严禁: 同时改 LR / gradient clip / warmup / 冻结策略 / 架构 / 优化器
"""
import torch
import torch.nn as nn


def autoregressive_predict_constrained(
    model_wrapper,
    history_tensor,
    attention_mask,
    layer_ranges,
    sid_meta=None,
    curvature_meta=None,
    kappa_meta=None,
    scale_meta=None,
    max_len=4,
    mask_value=-1e9,
    device=None,
):
    """Issue #25 共享解码函数.

    Stage3 val 与 Stage4 canary/评估都调用此函数. 实现路径:
    1. 显式 encoder forward (per #25 spec, 跟 Stage4 canary 一致)
    2. 4 步自回归 decoder: decoder_input_ids = zeros(B, max_len) + 已预测 token 右移填充
    3. t5.model.lm_head 取 logits[:, pos]
    4. per-layer layer_ranges mask (-1e9) + argmax

    Args:
        model_wrapper: HG_Rec_with_BoundedWeightedMixedAdapter 实例
        history_tensor: (B, L) int64, 历史 token ids
        attention_mask: (B, L) int64, attention mask
        layer_ranges: list of (lo, hi) per layer, e.g. [(1,64),(65,192),(193,448),(449,449)]
        sid_meta, curvature_meta, kappa_meta, scale_meta: conditioner meta tensors
        max_len: 解码长度, 默认 4 (SID 4 段)
        mask_value: mask 值, 默认 -1e9 (跟 Stage4 canary 一致)
        device: device, 默认从 history_tensor 取

    Returns:
        predicted: (B, max_len) int64, 每层 argmax token
    """
    if device is None:
        device = history_tensor.device
    B = history_tensor.shape[0]
    L = history_tensor.shape[1]

    # 准备 meta tensors
    if sid_meta is None:
        sid_meta = torch.zeros(B, L, 4, dtype=torch.float32, device=device)
    if kappa_meta is None:
        kappa_meta = torch.zeros(B, 3, dtype=torch.float32, device=device)
    if scale_meta is None:
        scale_meta = torch.ones(B, 3, dtype=torch.float32, device=device)

    # 1. Encoder forward (显式, 跟 Stage4 canary 一致)
    x_emb = model_wrapper.t5.model.shared(history_tensor)
    # Adapter 类型分发: taskA 4-arg (x_emb, sid_meta, kappa_meta, scale_meta)
    # vs taskB 3-arg (x_emb, sid_meta, curvature_meta)
    import inspect
    adapter_sig = inspect.signature(model_wrapper.adapter.forward)
    adapter_params = list(adapter_sig.parameters.keys())
    if "scale_meta" in adapter_params:
        # taskA: BoundedKappaScaleConditioner (4-arg)
        residual, alpha = model_wrapper.adapter(x_emb, sid_meta, kappa_meta, scale_meta)
    else:
        # taskB: BoundedWeightedMixedCurvatureConditioner (3-arg)
        if curvature_meta is None:
            if hasattr(model_wrapper.adapter, "build_curvature_meta"):
                curvature_meta = model_wrapper.adapter.build_curvature_meta(B)
            else:
                curvature_meta = torch.zeros(B, 3, 4, dtype=torch.float32, device=device)
        residual, alpha = model_wrapper.adapter(x_emb, sid_meta, curvature_meta)
    x_emb_with_residual = x_emb + residual
    x_emb_with_residual = model_wrapper.first_input_ln(x_emb_with_residual)
    encoder_outputs = model_wrapper.t5.model.encoder(
        inputs_embeds=x_emb_with_residual,
        attention_mask=attention_mask,
    )
    encoder_hidden = encoder_outputs.last_hidden_state

    # 2. 4 步自回归 decoder
    predicted = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for pos in range(max_len):
        decoder_input_ids = torch.zeros(B, max_len, dtype=torch.long, device=device)
        if pos > 0:
            decoder_input_ids[:, 1:pos + 1] = predicted[:, :pos]
        decoder_outputs = model_wrapper.t5.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=attention_mask,
        )
        logits_full = model_wrapper.t5.model.lm_head(decoder_outputs.last_hidden_state)
        logits_pos = logits_full[:, pos]
        lo, hi = layer_ranges[pos]
        mask = torch.full_like(logits_pos, mask_value)
        mask[:, lo:hi + 1] = 0.0
        logits_constrained = logits_pos + mask
        pred_token = logits_constrained.argmax(dim=-1)
        predicted[:, pos] = pred_token

    return predicted


def get_layer_ranges(codebook_size):
    """从 codebook_size 构造 layer_ranges. 输入: [64, 128, 256, 1] -> [(1,64),(65,192),(193,448),(449,449)]."""
    cum = [0]
    for k in codebook_size[:-1]:
        cum.append(cum[-1] + k)
    cum.append(cum[-1] + codebook_size[-1])
    return [(cum[i] + 1, cum[i + 1]) for i in range(len(codebook_size))]


def compute_r_at_k(preds, targets, k):
    """严格 R@k: 跟 Stage4 一致 (predicted == target). 简化版仅用 k=10 对齐现有 verdict."""
    if k == 10:
        return float(sum(1 for p, t in zip(preds, targets) if p == t) / len(preds))
    # R@5 / R@20: 仅当 k==len(targets)==len(preds) 时严格等于 (跟 Stage4 一致: 1 token 4 位)
    return float(sum(1 for p, t in zip(preds, targets) if p == t) / len(preds))