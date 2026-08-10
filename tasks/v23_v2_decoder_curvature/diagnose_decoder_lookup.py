#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0 验证: Stage4 beam KV-cache 阶段 decoder.forward 收到的 input_ids 形态.

假设: HF T5 generation 用 KV-cache 后, decoder.forward(input_ids=...) 每步只收到最新 token (shape=(B, 1)),
而不是完整 prefix. 此时 torch.roll(input_ids, 1) = input_ids 本身, lookup 完全错.

诊断: 在 monkey-patched decoder.forward (curv_decoder_forward) 外再 wrap 一层 log, log:
  - input_ids.shape
  - input_ids 实际 token
  - cache_position (如果存在)
  - past_key_values 是否存在 (length indicator)

不影响 curvature 注入, 保留 Stage4 完整评估流程. 只跑 2 个 sample.
"""
import os
import sys
import torch

REPO = "/home/wlia0047/ar57/wenyu/GeneRec"
os.chdir(REPO)
sys.path.insert(0, REPO)

# Stage4 eval 用 sys.argv, 设置 v23 v2 配置
sys.argv = [
    "stage4_diag",
    "--ckpt_path", "/home/wlia0047/ar57/wenyu/GeneRec/tasks/v23_v2_decoder_curvature/HG_Rec_best.pth",
    "--sid_npy", "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/v22b_full_1000ep/sid_output.npy",
    "--product_dir", "/tmp/diag_p0",
    "--expected_sid_sha", "332c948cce329944ee6a879636e91d261377b04766eec47df5b1fff9edc6eec5",
    "--hyperbolic_attn_bias", "--enable_residual_hab", "--hab_lambda_max", "0.2",
    "--residual_alpha_init", "-20.0",
    "--hab_stage2_ckpt", "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/v22b_full_1000ep/hrqvae_kappa_sync.ckpt",
    "--curvature_residual_enabled", "--curvature_residual_decoder_enabled",
    "--curvature_residual_layer", "1", "--curvature_residual_mlp_hidden", "64",
    "--curvature_residual_alpha_init", "0.0",
    "--tag", "diagnose_p0",
]

from common.stage4 import stage4_eval_pure_t5_v85p_4layer as st4
import types

_dec_calls = []
_orig_install = st4.install_curvature_residual_eval


def patched_install(hg_rec, curv_module, layer_id_lut_tensor, decoder_enabled=False):
    """在原 install 之后, 把 curv_decoder_forward 包一层 log (保留 curvature 注入)."""
    result = _orig_install(hg_rec, curv_module, layer_id_lut_tensor, decoder_enabled=decoder_enabled)
    if decoder_enabled:
        # 原 install 已把 hg_rec.model.decoder.forward 替换为 curv_decoder_forward
        existing_patched = hg_rec.model.decoder.forward

        def log_wrapper(self, input_ids=None, attention_mask=None, **kwargs):
            if input_ids is not None:
                cp = kwargs.get("cache_position", None)
                past_kv = kwargs.get("past_key_values", None)
                _dec_calls.append({
                    "input_ids_shape": tuple(input_ids.shape),
                    "input_ids_first_row": input_ids[0].tolist(),
                    "has_inputs_embeds": kwargs.get("inputs_embeds", None) is not None,
                    "cache_position": cp.tolist() if cp is not None and hasattr(cp, "tolist") else None,
                    "past_kv_present": past_kv is not None,
                })
            # existing_patched 是 bound method (MethodType 已 bind self), 不要重复传 self
            # 避免 input_ids 同时出现在显式 args 和 **kwargs 里导致 'multiple values' 错误
            call_kwargs = dict(kwargs)
            if input_ids is not None:
                call_kwargs["input_ids"] = input_ids
            if attention_mask is not None:
                call_kwargs["attention_mask"] = attention_mask
            return existing_patched(**call_kwargs)

        hg_rec.model.decoder.forward = types.MethodType(log_wrapper, hg_rec.model.decoder)
    return result


st4.install_curvature_residual_eval = patched_install

# 限制 batch_size / sample 数. 但 main() 内部 BATCH_SIZE=96 hardcoded, 我们用 max_eval_samples=2
# 如果 main() 不支持 max_eval_samples, 我们手动 break.
if hasattr(st4, "BATCH_SIZE"):
    st4.BATCH_SIZE = 2  # 只跑 2 个 sample


print("=" * 70, flush=True)
print("P0 诊断: Stage4 v23 v2 decoder curvature lookup 在 KV-cache beam 时的 input_ids 形态", flush=True)
print("=" * 70, flush=True)

try:
    st4.main()
except SystemExit:
    pass
except Exception as e:
    print(f"\n[warn] main() 异常 (继续分析): {type(e).__name__}: {e}", flush=True)

# 输出诊断结果
print("\n" + "=" * 70, flush=True)
print(f"DIAGNOSTIC RESULTS: 总 decoder.forward 调用 = {len(_dec_calls)}", flush=True)
print("=" * 70, flush=True)

# 分布统计
from collections import Counter
shape_dist = Counter()
seq_len_dist = Counter()
for c in _dec_calls:
    shape_dist[c["input_ids_shape"]] += 1
    seq_len_dist[c["input_ids_shape"][-1]] += 1

print("\ninput_ids.shape 全量分布:", flush=True)
for s, cnt in sorted(shape_dist.items(), key=lambda x: -x[1]):
    print(f"  shape={s}: {cnt} 次", flush=True)

# 打印前 30 个调用的细节
print("\n前 30 次调用详情:", flush=True)
for i, c in enumerate(_dec_calls[:30]):
    cache_str = f"cache_pos={c['cache_position']}" if c['cache_position'] is not None else "cache_pos=N/A"
    past_str = "past_kv=YES" if c['past_kv_present'] else "past_kv=None"
    print(f"  [{i:3d}] shape={c['input_ids_shape']} tokens={c['input_ids_first_row'][:8]}{'...' if len(c['input_ids_first_row'])>8 else ''} "
          f"{cache_str} {past_str}", flush=True)

# 抽样第 50, 100, 150
for idx in [50, 100, 150]:
    if idx < len(_dec_calls):
        c = _dec_calls[idx]
        print(f"  [{idx:3d}] shape={c['input_ids_shape']} tokens={c['input_ids_first_row'][:8]}{'...' if len(c['input_ids_first_row'])>8 else ''}", flush=True)

# P0 判断
total = len(_dec_calls)
seq_len_1_count = seq_len_dist.get(1, 0)
pct_1 = 100 * seq_len_1_count / total if total > 0 else 0

print("\n" + "=" * 70, flush=True)
print(f"P0 判定:", flush=True)
print(f"  seq_len=1 的调用占比: {pct_1:.1f}% ({seq_len_1_count}/{total})", flush=True)
if pct_1 > 80:
    print("\n❌ P0 CONFIRMED:", flush=True)
    print(f"   {pct_1:.1f}% 的 decoder.forward 调用收到 input_ids.seq_len=1", flush=True)
    print("   → KV-cache 截断完整 prefix", flush=True)
    print("   → torch.roll(input_ids, 1) = input_ids 本身, L1 lookup 把当前 token 当成 q_0 (错!)", flush=True)
    print("   → L2 lookup 更错: torch.roll(input_ids, 2) 也是 input_ids 本身", flush=True)
    print("   → 这是 train-test mismatch:", flush=True)
    print("      * 训练 (teacher forcing): decoder 一次看到完整 shifted sequence → lookup 正确", flush=True)
    print("      * 推理 (beam + KV-cache): decoder 每步只看当前 token → lookup 错误", flush=True)
    print("   → 这能解释为何 decoder α 学到 0.5-0.6 magnitude (训练正确消费)", flush=True)
    print("     但 test_R@10 跌到 0.0909 (推理 lookup 完全错)", flush=True)
    print("\n   建议: 撤掉 v23 v2 NO-GO attribution (decode curvature 本身可能有效, 但 lookup 错)", flush=True)
    print("         修复方案: 维护 beam prefix buffer, 在 generate 外层 wrap, decoder forward 用 buffer 全长算 curvature", flush=True)
elif pct_1 < 20:
    print("\n✅ P0 REFUTED:", flush=True)
    print(f"   只有 {pct_1:.1f}% 调用 seq_len=1, 大多数调用是完整 prefix", flush=True)
    print("   → KV-cache 没有截断 decoder.forward 的 input_ids", flush=True)
    print("   → 那么 0.0909 的 NO-GO 应归因于 decoder curvature 本身", flush=True)
else:
    print(f"\n⚠️ P0 PARTIAL: {pct_1:.1f}% 调用 seq_len=1, 需细看", flush=True)

print("=" * 70, flush=True)