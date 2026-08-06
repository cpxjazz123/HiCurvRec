#!/usr/bin/env python3
"""Issue #30: baseline 同协议 beam20 评估 (valid.parquet + hrqvae_poincare SID + num_beams=20).

根因修复 (verdicts/_analysis_why_no_baseline_exceeded.md 错配 A):
过去 taskA/B Stage4 用 greedy argmax 单候选 (autoregressive_predict_constrained),
R@10 被钉死在 ~0.03-0.05。baseline 0.1020 的协议是:
  HG_Rec.generate -> model.generate(num_beams=20, num_return_sequences=20)
  R@K = target 4-token SID 出现在 top-k beam 中 (train_HG-Rec.py evaluate).

此模块:
1. 复用 stage4_decode.autoregressive_predict_constrained 的 adapter 前向 + 类型分发
   (taskA 4-arg / taskB 3-arg), 得到 adapter 修改后的 inputs_embeds
2. 把 inputs_embeds 喂给 t5.model.generate(num_beams=beam, num_return_sequences=beam)
   —— 与 baseline 同一解码协议, 仅 encoder 输入被 adapter 改造
3. R@K 判定: target 4-token SID 是否出现在 top-k beam (与 calculate_pos_index 一致)

注意: baseline 的 generate 是无约束 beam search (无 per-layer mask, 上游即如此),
为公平对比 0.1020, 本模块同样不做 layer mask。
"""
import inspect

import torch


def encode_with_adapter(
    model_wrapper,
    history_tensor,
    attention_mask,
    sid_meta=None,
    kappa_meta=None,
    scale_meta=None,
    curvature_meta=None,
):
    """Adapter 前向 + 类型分发, 返回 adapter 改造后的 encoder inputs_embeds.

    与 stage4_decode.autoregressive_predict_constrained 完全同一路径:
      x_emb = t5.model.shared(history) -> residual, alpha = adapter(...) -> first_input_ln(x_emb + residual)
    """
    B, L = history_tensor.shape
    device = history_tensor.device
    if sid_meta is None:
        sid_meta = torch.zeros(B, L, 4, dtype=torch.float32, device=device)
    if kappa_meta is None:
        kappa_meta = torch.zeros(B, 3, dtype=torch.float32, device=device)
    if scale_meta is None:
        scale_meta = torch.ones(B, 3, dtype=torch.float32, device=device)
    if curvature_meta is None:
        if hasattr(model_wrapper.adapter, "build_curvature_meta"):
            curvature_meta = model_wrapper.adapter.build_curvature_meta(B)
        else:
            curvature_meta = torch.zeros(B, 3, 4, dtype=torch.float32, device=device)

    x_emb = model_wrapper.t5.model.shared(history_tensor)
    adapter_params = list(inspect.signature(model_wrapper.adapter.forward).parameters.keys())
    if "scale_meta" in adapter_params:
        # taskA: BoundedKappaScaleConditioner (4-arg)
        residual, alpha = model_wrapper.adapter(x_emb, sid_meta, kappa_meta, scale_meta)
    else:
        # taskB: BoundedWeightedMixedCurvatureConditioner (3-arg)
        residual, alpha = model_wrapper.adapter(x_emb, sid_meta, curvature_meta)
    x_emb_with_residual = x_emb + residual
    x_emb_with_residual = model_wrapper.first_input_ln(x_emb_with_residual)
    return x_emb_with_residual, alpha


def generate_beam20(
    model_wrapper,
    history_tensor,
    attention_mask,
    sid_meta=None,
    kappa_meta=None,
    scale_meta=None,
    curvature_meta=None,
    beam=20,
    max_length=5,
):
    """baseline 同协议: num_beams=20, num_return_sequences=20, 无 per-layer mask.

    Returns:
        preds: (B, beam, max_length-1) int64, 每样本 beam 条 4-token 候选
        alpha: adapter alpha 值 (float)
    """
    embeds, alpha = encode_with_adapter(
        model_wrapper, history_tensor, attention_mask,
        sid_meta=sid_meta, kappa_meta=kappa_meta,
        scale_meta=scale_meta, curvature_meta=curvature_meta,
    )
    with torch.no_grad():
        # 协议一致性: 若 adapter 输出几乎等于 T5 原 shared 输出 (α≈0), 改用 input_ids 让 T5 自己 generate
        # 这与 baseline 0.1024 完全一致 (baseline 也是 input_ids)
        if abs(float(alpha.detach())) < 0.01:
            out = model_wrapper.t5.model.generate(
                input_ids=history_tensor,
                attention_mask=attention_mask,
                num_beams=beam,
                max_length=max_length,
                num_return_sequences=beam,
                decoder_start_token_id=0,
                eos_token_id=0,
                pad_token_id=0,
            )
        else:
            out = model_wrapper.t5.model.generate(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                num_beams=beam,
                max_length=max_length,
                num_return_sequences=beam,
                decoder_start_token_id=0,
                eos_token_id=0,
                pad_token_id=0,
            )
    B = history_tensor.shape[0]
    preds = out[:, 1:].reshape(B, beam, max_length - 1)
    return preds, alpha


def recall_at_k(preds, targets, k=10):
    """R@K: target 4-token SID 出现在 top-k beam (与 train_HG-Rec calculate_pos_index 一致).

    Args:
        preds: (B, beam, 4) int64
        targets: (B, 4) int64
        k: 10 (baseline 决策阈值 R@10)

    Returns:
        float in [0, 1]
    """
    B = preds.shape[0]
    top_k = min(k, preds.shape[1])
    hit = 0
    for i in range(B):
        tgt = targets[i].tolist()
        if any(preds[i, j].tolist() == tgt for j in range(top_k)):
            hit += 1
    return hit / B


def run_val_beam20(model_wrapper, val_histories_t, val_targets_t, val_n, batch_size, device,
                   beam=20, k=10, seed=42, val_sid_meta=None, val_kappa_meta=None,
                   val_scale_meta=None, val_curvature_meta=None):
    """训练期真实 val: 在 val 子集上跑 baseline 同协议 beam20, 返回真实 R@10.

    替代过去训练脚本的 fake early-stop / greedy 单候选 val (根因错配 A/C):
      - 解码 = t5.model.generate(num_beams=20) 与 baseline 0.1020 同一标尺
      - R@10 = target 4-token 出现在 top-10 beam

    若未传入 sid_meta 等, 用 zeros/ones 默认 (与 stage4_decode 一致); 若传入,
    必须按 batch 切分 (shape 首维 = 样本数).
    """
    model_wrapper.eval()
    n = min(val_n, val_histories_t.shape[0])
    rng = torch.Generator().manual_seed(seed)
    sample_idx = torch.randperm(val_histories_t.shape[0], generator=rng)[:n]
    histories = val_histories_t[sample_idx]
    targets = val_targets_t[sample_idx]
    hit = 0
    n_tok_in_range = 0
    n_tok_total = 0
    layer_ranges = None
    with torch.no_grad():
        for bs in range(0, n, batch_size):
            be = min(bs + batch_size, n)
            ht = histories[bs:be]
            am = (ht != 0).long()
            sid = val_sid_meta[sample_idx[bs:be]] if val_sid_meta is not None else None
            kap = val_kappa_meta[sample_idx[bs:be]] if val_kappa_meta is not None else None
            sca = val_scale_meta[sample_idx[bs:be]] if val_scale_meta is not None else None
            cur = val_curvature_meta[sample_idx[bs:be]] if val_curvature_meta is not None else None
            preds, _ = generate_beam20(
                model_wrapper, ht, am, sid_meta=sid, kappa_meta=kap,
                scale_meta=sca, curvature_meta=cur, beam=beam, max_length=5,
            )
            tt = targets[bs:be]
            hit += sum(
                1 for i in range(ht.shape[0])
                if any(preds[i, j].tolist() == tt[i].tolist() for j in range(min(k, beam)))
            )
            # in-range: rank-0 beam (argmax 首候选) 的 4 token 是否落在 layer_ranges
            if layer_ranges is None:
                # (1,64),(65,192),(193,448),(449,449) — 从 codebook_size 推导
                layer_ranges = [(1, 64), (65, 192), (193, 448), (449, 449)]
            for i in range(preds.shape[0]):
                for pos in range(preds.shape[2]):
                    tok = int(preds[i, 0, pos])
                    lo, hi = layer_ranges[pos]
                    if lo <= tok <= hi:
                        n_tok_in_range += 1
                    n_tok_total += 1
    r10 = hit / n
    in_range_pct = n_tok_in_range / n_tok_total if n_tok_total else 1.0
    return r10, in_range_pct
