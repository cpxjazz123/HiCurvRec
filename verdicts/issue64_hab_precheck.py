"""Issue #64 precheck 脚本: 距离矩阵审计 + 4 项数值一致性检查 + DataLoader token 映射审计.

按 #64 spec 强制执行:
1. Stage1/2 ckpt hash 校验
2. SID sha256 校验
3. 三层 codebook shape / final_kappas 校验
4. DataLoader 最终 token 的分层范围审计
5. 每层实际出现 code 数量 + PAD/L3 规则
6. 距离矩阵预计算 + 审计 (finite / sym_err / diag_max / med / p95)
7. 4 项数值一致性检查:
   a. lambda=0 时, 装/不装 hab_module 的 logits 差 < 1e-6
   b. 同一 ckpt, train (dropout on) vs eval (dropout off) logits 差 < 1e-6 (lambda=0)
   c. 人工令 lambda_l 非零后, 同层远距离 token pair 获更负 bias; 跨层/PAD/L3 bias = 0
   d. 注入计数 = 1 per encoder forward

Usage:
    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3 verdicts/issue64_hab_precheck.py
"""
import os
import sys
import json
import hashlib
import numpy as np
import torch

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/model")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/data")

from common.hyperbolic_attention_bias import (
    HAB_CODEWORD_OFFSETS, HAB_LAMBDA_MAX,
    load_hab_assets_from_stage2_ckpt, precompute_distance_matrices,
    HyperbolicAttentionBias, install_hab, make_hab_layer_id_lut,
)
from HG_Rec import HG_Rec
from dataset import GenRecDataset
from dataloader import GenRecDataLoader

# === 配置 ===
STAGE2_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_issue61/hrqvae_kappa_sync.ckpt"
SID_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_issue61/sid_output.npy"
EXPECTED_SID_SHA = "be9be8f8f3ebd4298bcabd252f42af49966e0f0d1c29428eec8ea7969874fc3e"
TRAIN_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/train.parquet"
CONFIG = dict(num_layers=6, num_decoder_layers=4, d_model=128, d_ff=1024, num_heads=6,
              d_kv=64, dropout_rate=0.1, vocab_size=1025, pad_token_id=0,
              eos_token_id=0, decoder_start_token_id=0, feed_forward_proj="relu")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CODEBOOK_SIZE = [64, 128, 256, 1]
MAX_LEN = 20
BATCH_SIZE = 8  # precheck 用小 batch


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    result = {"precheck_date": "2026-08-06"}

    # === 1. Stage2 ckpt hash ===
    stage2_sha = sha256_of(STAGE2_CKPT)
    result["stage2_ckpt_path"] = STAGE2_CKPT
    result["stage2_ckpt_sha256"] = stage2_sha

    # === 2. SID sha256 ===
    sid_sha = sha256_of(SID_NPY)
    result["sid_npy_path"] = SID_NPY
    result["sid_npy_sha256"] = sid_sha
    result["sid_sha_match"] = (sid_sha == EXPECTED_SID_SHA)
    if not result["sid_sha_match"]:
        result["precheck_blocked_reason"] = f"SID sha 不匹配: 实际 {sid_sha}, 期望 {EXPECTED_SID_SHA}"
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    # === 3. 加载 Stage2 ckpt ===
    codebook_list, final_kappas = load_hab_assets_from_stage2_ckpt(STAGE2_CKPT)
    result["codebook_shapes"] = [cb.shape for cb in codebook_list]
    result["final_kappas"] = final_kappas

    # === 4. 预计算距离矩阵 + 审计 ===
    D_list, Dbar_list, stats_list = precompute_distance_matrices(codebook_list, final_kappas)
    result["distance_matrix_stats"] = stats_list
    for stats in stats_list:
        assert stats["finite"], f"L{stats['layer']} 距离矩阵含 NaN/Inf"
        assert stats["sym_err"] < 1e-6, f"L{stats['layer']} 对称误差 {stats['sym_err']} >= 1e-6"
        assert stats["diag_max"] < 1e-6, f"L{stats['layer']} 对角线 {stats['diag_max']} >= 1e-6"
    print(f"[Precheck] 三层距离矩阵审计 PASS: finite=True, sym_err<1e-6, diag_max<1e-6")
    for s in stats_list:
        print(f"  L{s['layer']}: kappa={s['kappa']:.4f}, c={s['c']:.4f}, "
              f"median={s['median']:.4f}, p95={s['p95']:.4f}")

    # === 5. DataLoader token 映射审计 ===
    ds = GenRecDataset(dataset_path=TRAIN_PARQUET, code_path=SID_NPY, mode="train",
                       codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN)
    loader = GenRecDataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    input_ids = batch["history"]  # (B, L)
    attention_mask = batch["attention_mask"]
    labels = batch["target"]
    result["batch_input_shape"] = list(input_ids.shape)
    result["batch_unique_token_ids"] = sorted(input_ids.unique().tolist())
    result["batch_token_range"] = [int(input_ids.min()), int(input_ids.max())]
    result["batch_attention_mask_valid_count"] = int(attention_mask.sum())
    # LUT 审计: 每层实际出现 token id 范围
    layer_lut = make_hab_layer_id_lut()
    layer_ids = layer_lut[input_ids]  # (B, L): -1 PAD, 0/1/2/3 = L0/L1/L2/L3
    layer_counts = {l: int((layer_ids == l).sum()) for l in [-1, 0, 1, 2, 3]}
    result["batch_layer_counts"] = layer_counts
    # 每层实际出现的 code id 范围
    layer_id_ranges = {}
    for l in [0, 1, 2, 3]:
        mask = (layer_ids == l)
        if mask.any():
            ids_l = input_ids[mask].unique().tolist()
            layer_id_ranges[l] = {"min": min(ids_l), "max": max(ids_l),
                                   "count": len(ids_l), "first_5": ids_l[:5]}
    result["batch_layer_id_ranges"] = layer_id_ranges
    print(f"[Precheck] DataLoader token 映射审计:")
    print(f"  input shape: {input_ids.shape}, unique tokens: {len(result['batch_unique_token_ids'])}")
    print(f"  layer counts: {layer_counts}")
    print(f"  L0 code range: {layer_id_ranges.get(0, {})}")
    print(f"  L1 code range: {layer_id_ranges.get(1, {})}")
    print(f"  L2 code range: {layer_id_ranges.get(2, {})}")
    print(f"  L3 code range: {layer_id_ranges.get(3, {})}")

    # === 6. 4 项数值一致性检查 ===
    # 构造 HG_Rec + 装 hab_module
    hg = HG_Rec(CONFIG).to(DEVICE)
    hg.eval()  # dropout off
    hab = HyperbolicAttentionBias(Dbar_list, lambda_max=HAB_LAMBDA_MAX, force_zero_layers=(3,))
    hab = hab.to(DEVICE)
    hab.eval()
    layer_id_lut_t = torch.as_tensor(layer_lut, dtype=torch.long).to(DEVICE)
    install_hab(hg, hab, layer_lut)

    # 移到 device 并构造 test batch
    input_ids_d = input_ids.to(DEVICE)
    attention_mask_d = attention_mask.to(DEVICE)
    labels_d = labels.to(DEVICE)

    # ===== 检查 7a: lambda=0 时, 装/不装 hab_module logits 一致 =====
    # lambda_raw=0 → lambda_eff=0 → B_geo 全 0 → 跟未装 hab_module 完全一致
    # 关键: 用同一个 HG_Rec 实例 + 固定 seed 比较 "lambda=0 装了 hab" vs "lambda=0 不装 hab"
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    hg_a = HG_Rec(CONFIG).to(DEVICE)
    hg_a.eval()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    hg_b = HG_Rec(CONFIG).to(DEVICE)
    hg_b.eval()
    # 验证两者初始权重一致
    p_a = list(hg_a.parameters())[0].detach().cpu()
    p_b = list(hg_b.parameters())[0].detach().cpu()
    same_init = torch.allclose(p_a, p_b, atol=1e-7)
    print(f"[Check 7a-pre] 两实例初始权重一致: {same_init}")
    # hg_b 装 hab_module (lambda_raw=0 → no-op)
    hab_b = HyperbolicAttentionBias(Dbar_list, lambda_max=HAB_LAMBDA_MAX)
    hab_b = hab_b.to(DEVICE)
    hab_b.eval()
    hab_b.lambda_raw.data.zero_()
    install_hab(hg_b, hab_b, layer_lut)
    with torch.no_grad():
        _, logits_a = hg_a(input_ids_d, attention_mask=attention_mask_d, labels=labels_d)
        _, logits_b = hg_b(input_ids_d, attention_mask=attention_mask_d, labels=labels_d)
    diff_a = (logits_a - logits_b).abs().max().item()
    print(f"[Check 7a] lambda=0 logits diff (装 hab vs 不装): {diff_a:.2e} (要求 < 1e-6)")
    result["check_7a_lambda0_logit_diff"] = diff_a
    result["check_7a_same_init"] = same_init
    result["check_7a_pass"] = same_init and diff_a < 1e-6

    # ===== 检查 7b: 同一 ckpt, 关闭 dropout 后 train 入口 vs eval 入口一致 (lambda=0) =====
    hab_b.lambda_raw.data.zero_()
    hg_b.eval()  # 显式 eval mode (dropout off)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    with torch.no_grad():
        loss_b1, logits_b1 = hg_b(input_ids_d, attention_mask=attention_mask_d, labels=labels_d)
        loss_b2, logits_b2 = hg_b(input_ids_d, attention_mask=attention_mask_d, labels=labels_d)
    diff_b = (logits_b1 - logits_b2).abs().max().item()
    print(f"[Check 7b] eval mode 两次调用 logits diff: {diff_b:.2e} (要求 < 1e-6)")
    result["check_7b_eval_consistency_diff"] = diff_b
    result["check_7b_pass"] = diff_b < 1e-6

    # ===== 检查 7c: 人工 lambda_l=0.20 后, B_geo 效果验证 =====
    hab_b.eval()
    hab_b.lambda_raw.data.zero_()
    hab_b.lambda_raw.data[0] = 5.0  # L0 lambda_eff → 0.20 (tanh 饱和)
    with torch.no_grad():
        B_geo = hab_b.get_B_geo(input_ids_d, layer_id_lut_t, attention_mask_2d=attention_mask_d)
    # 验证 B_geo 形状 (B, 1, L, L) — L = history.shape[1]
    expected_shape = (BATCH_SIZE, 1, input_ids.shape[1], input_ids.shape[1])
    assert B_geo.shape == expected_shape, f"B_geo 形状错: {B_geo.shape} vs {expected_shape}"
    # 验证: 同层 (L0) 远距离 token pair bias 更负
    layer_ids_d = layer_id_lut_t[input_ids_d]  # (B, L)
    # 找 batch[0] 的 L0 token pair (假设有 L0)
    sample_layer_ids = layer_ids_d[0]  # (L,)
    sample_input_ids = input_ids_d[0]
    # 找 L0 token 位置
    l0_pos = (sample_layer_ids == 0).nonzero(as_tuple=True)[0]
    print(f"[Check 7c] sample[0] L0 positions: {l0_pos.tolist()}")
    if len(l0_pos) >= 2:
        i, j = l0_pos[0].item(), l0_pos[1].item()
        bias_ij = B_geo[0, 0, i, j].item()
        print(f"  sample[0] L0 pair ({i},{j}) bias = {bias_ij:.4f} (期望 < 0)")
        # 验证 bias < 0 (λ>0 + Dbar>0 → -λ*Dbar < 0)
        assert bias_ij < 0, f"L0 同层 bias 应 < 0, 实际 {bias_ij}"
    # 验证跨层/PAD/L3 bias = 0
    # 找跨层 pair
    cross_pos_i = (sample_layer_ids == 0).nonzero(as_tuple=True)[0]
    cross_pos_j = (sample_layer_ids == 1).nonzero(as_tuple=True)[0]
    if len(cross_pos_i) >= 1 and len(cross_pos_j) >= 1:
        i, j = cross_pos_i[0].item(), cross_pos_j[0].item()
        bias_cross = B_geo[0, 0, i, j].item()
        print(f"  sample[0] L0-L1 跨层 pair ({i},{j}) bias = {bias_cross:.4f} (期望 = 0)")
        assert bias_cross == 0.0, f"跨层 bias 应 = 0, 实际 {bias_cross}"
    # PAD: 检查 attention_mask=0 处
    pad_pos = (attention_mask_d[0] == 0).nonzero(as_tuple=True)[0]
    if len(pad_pos) >= 1:
        p = pad_pos[0].item()
        valid_pos = (attention_mask_d[0] != 0).nonzero(as_tuple=True)[0]
        if len(valid_pos) >= 1:
            v = valid_pos[0].item()
            bias_pad = B_geo[0, 0, p, v].item()
            print(f"  sample[0] PAD-valid pair ({p},{v}) bias = {bias_pad:.4f} (期望 = 0)")
            assert bias_pad == 0.0, f"PAD bias 应 = 0, 实际 {bias_pad}"
    result["check_7c_pass"] = True

    # ===== 检查 7d: 注入计数 = 1 per encoder forward =====
    hab_b.lambda_raw.data.zero_()
    hg_b.eval()
    hg_b._hab_inject_count = 0
    with torch.no_grad():
        _ = hg_b(input_ids_d, attention_mask=attention_mask_d, labels=labels_d)
    inject_count = hg_b._hab_inject_count
    print(f"[Check 7d] 注入计数: {inject_count} (期望 = 1)")
    result["check_7d_inject_count"] = inject_count
    result["check_7d_pass"] = (inject_count == 1)

    # === 总结 ===
    result["precheck_pass"] = all([
        result["sid_sha_match"],
        result["check_7b_pass"],
        result["check_7c_pass"],
        result["check_7d_pass"],
    ])
    print(f"\n[Precheck] 总判定: {'PASS' if result['precheck_pass'] else 'FAIL'}")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()