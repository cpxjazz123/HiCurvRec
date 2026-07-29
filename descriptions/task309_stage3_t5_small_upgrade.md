# Task #309 — Stage 3 T5-mini → T5-small 升级 (容量扩展验证)

**日期**: 2026-07-30
**目的**: 验证 K14 (Stage 3/4 训练协议是真 R@10 杠杆) 在**容量扩展**维度是否成立. T5-mini (9.18M params) → T5-small (60M params, ~6.5× params), 仅替换 Stage 3 训练架构, Stage 1/2 产物不变 (沿用 #30 codebook + SID).
**R11.5 自主决策**: T5-small 标准 config (d_model=512, d_ff=2048, num_heads=8, num_layers=6, num_decoder_layers=6). 沿用 #30 Stage 2 codebook (per-layer Codebook Transforms, K14 唯一 GO config). seed=42. Stage 4 eval 用 beam=50 (K14 最优).
**高 ROI 候选**: HG-Rec paper Table 1 用 T5-small, 当前复现用 T5-mini (size mismatch 可能解释部分 paper vs 复现 20.5% gap).
**预期**: ~4-5h Stage 3 训练 (GPU 1) + ~36s Stage 4 eval.
**关联**:
- K14: Stage 4 beam_size 20→50 +2.3% R@10 ✅
- K15: Stage 4 length_penalty 0% ❌ (length_penalty 不是 R@10 杠杆)
- Task #290/291/292 verdict: 攻 Stage 3/4 训练协议而非 Stage 1/2 quantizer 架构
- [[hgrec-paper-comparison]]: paper Table 1 用 T5-small, 复现用 T5-mini, 8/8 baseline 复现均低于 paper 18-61%

---

## Gate 协议 (5-Gate 简化版, 跟 Stage 3/4 一致)

- **Gate 0**: 配置 + ckpt 路径 + GPU 空闲检查 (✅ 已通过)
- **Gate 1**: Stage 3 训练 200 epoch (R12 save best_ckpt, R7 不抢卡, R9 编号 max+1=309)
- **Gate 2**: Stage 4 eval beam=50 (沿用 #30 best_ckpt 协议, K14 最优)
- **Gate 3**: 写 verdict + 更新 loop.md §16 + paper.md §6.7.4
- **Gate 4 (硬停止)**: 若 Stage 4 R@10 < baseline (#30 beam=50 = 0.1045), NO-GO 收口; 否则 +pp 比例升级

---

## 关键参数 (R11.3 透明记录)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | T5 配置 | d_model=512 d_ff=2048 num_heads=8 num_layers=6 num_decoder_layers=6 (T5-small standard) | T5-base (220M, d_model=768) | T5-small 是 paper Table 1 对照; T5-base 单卡 4-5h 训练可能 OOM |
| 2 | batch_size | 256 (跟 #30 一致) | 128 (OOM risk) | d_model=512 + d_ff=2048 显存 ~4x T5-mini, batch_size 256 实测可跑 |
| 3 | lr | 1e-4 (跟 #30 一致) | 5e-5 (T5-small 标准) | 单 seed 沿用 #30 lr 协议 |
| 4 | epochs | 200 (跟 #30 一致) | 100 (early stop) | 跟 #30 公平对比 |
| 5 | seed | 42 (单 seed, R11.5 禁 multi-seed) | - | [[user-no-multiseed-override]] 强制 |
| 6 | code_path | _t5_hrqvae_issue30_per_layer_transforms.npy (沿用 #30) | _t5_hrqvae_poincare.npy (baseline) | K14 验证 #30 GO config 是当前最优 Stage 2 产物 |
| 7 | Stage 4 beam_size | 50 (K14 最优) | 20 (历史默认) | K14 验证 beam=50 是 Stage 4 inference 协议层唯一杠杆 |
| 8 | GPU | 1 (R7 空闲) | 0/2/3 (备用) | R7 不抢卡, GPU 0 刚跑完 task308 释放 |
