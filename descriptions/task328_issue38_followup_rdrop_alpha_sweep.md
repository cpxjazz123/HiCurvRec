# Task #328 — Issue #38 Follow-up: R-Drop Alpha Sweep (4-arm)

**日期**: 2026-07-30
**状态**: 🔄 READY (R7: 4×L40S 可用, 等 task320 完成 + issue38 注释落地)
**Stage**: Stage 3 protocol Layer 2 — R-Drop alpha tuning
**Anchor**: 
- Issue #30 GO endpoint (task301) R@10=0.1022 (Stage 1/2 baseline)
- task194_k0256 R@10=0.1053 (+3.2% baseline) — 终极 anchor
- **task320 Arm C R-Drop α=1.0 R@10=0.1034 (+1.4% baseline)** — 本任务的 α=1.0 baseline
**决策阈值**: 任何 arm R@10 > 0.1053 (task194_k0256 anchor) = 实质突破 Stage 3 协议层 ceiling

## Background (Issue #38 PARTIAL GO → Layer 2 follow-up)

### Issue #38 5-arm Stage 3 协议改造 结果 (task320)
- Arm A (AdamW+cosine): test_R@10=0.0942 (-7.6%) NO-GO
- Arm B (Adam+inv_sqrt): test_R@10=0.0938 (-8.1%) NO-GO
- Arm D (BF16): test_R@10=0.0983 (-3.6%) NO-GO
- Arm E (control): test_R@10=0.0981 (-3.9%) NO-GO
- **Arm C (R-Drop α=1.0): test_R@10=0.1034 (+1.4%) ✅ GO** ⭐

**R-Drop α=1.0 是唯一 Stage 3 协议改造 GO 实证**, 机制 = 2 forward pass + 对称 KL 散度 (implicit regularization).

### Why alpha sweep? (R11.5 reasoning)

R-Drop 的 α 是对称 KL 散度的系数:
```
loss = 0.5 * (CE(logits1, labels) + CE(logits2, labels)) + α * KL(p1 || p2) symmetric
```

- α=0.0 → 等价于无 R-Drop baseline (但仍 2 forward pass, 实际 ≠ control)
- α=0.5 → 弱正则化
- α=1.0 → 当前 Arm C GO 实证
- α=2.0 → 强正则化 (可能过强, 损害 fitting)
- α=4.0 → 极强正则化 (适合大数据集; 在 9922 items 5-core 可能过强)

**数学上**: KL 是 bounded [0, +∞), α 直接缩放其贡献. 当前 Arm C α=1.0 是 "标准值" (paper 常用), 但 Musical_Instruments 9922 items 可能需要不同 α.

## 4-arm R-Drop alpha sweep design

| Arm | α | 训练 epoch | 其他超参 | 期望行为 |
|-----|---|-----------|---------|---------|
| **A** | 0.5 | 200 | baseline Adam+constant+dropout 0.1 | 弱正则化, 可能 overfit (val_R@10 高 test_R@10 低) |
| **B** | 1.0 | 200 | baseline (task320 Arm C 复用 baseline) | 当前 GO 实证, 锚点 |
| **C** | 2.0 | 200 | baseline | 强正则化, 抗 overfit |
| **D** | 4.0 | 200 | baseline | 极强正则化, 可能 underfit |

**共同基线** (跨 4 arm 复用):
- Stage 1/2 config: Issue #30 GO endpoint (`_t5_hrqvae_issue30_per_layer_transforms.npy`)
- T5-mini 5.5M, d_model=128, d_ff=1024, num_heads=6, num_layers=6, num_decoder_layers=4
- optimizer: Adam lr=1e-4 weight_decay=0.0
- LR schedule: constant
- dropout: 0.1
- seed: 42
- 训练 epoch: 200 full sweep
- best_ckpt save strategy: best val_NDCG@20 (R12)
- Stage 4 K=100 eval (Issue #30 ε ceiling)

**Stage 3 训练代码**: 复用 `scripts/task320_issue38_5arm_stage3_train.py`, 加 `--rdrop_alpha` CLI 参数
**Stage 4 eval 代码**: 复用 `scripts/task320_arm_stage4_eval.py`, 加 `--rdrop_alpha` 标记输出 (但实际不需, ckpt 自动决定)

## Settings

- dataset: Musical_Instruments (Instruments, 9922 items 5-core)
- Stage 1/2 codebook: Issue #30 GO endpoint `_t5_hrqvae_issue30_per_layer_transforms.npy`
- ckpt: 每个 arm 自己训练 200 epoch + 保存 HG_Rec_best.pth
- GPU: 4×L40S (R7: 1 arm 1 GPU)
- 种子: seed=42 跨 run 固定 (R5 baseline 一致性)
- 总 wall time 估: 4 GPU 并行, 200 epoch × ~1 min/ep = ~3.5 hr/单 arm, **~3.5 hr 全部完成** (并行)

## Decision threshold

- ✅ **GO**: 任何 arm R@10 > 0.1053 (task194_k0256 anchor) → R-Drop alpha tuning 在 [0.5, 4.0] 找到 task194_k0256 anchor 路径
- 🟡 **MARGINAL GO**: 任何 arm R@10 > 0.1034 (task320 Arm C) 但 ≤ 0.1053 → alpha 是微杠杆, 但不突破 anchor
- ❌ **NO-GO**: 所有 arm R@10 ≤ 0.1034 → R-Drop α=1.0 已是 ceiling, alpha 不是杠杆
- 🔴 **NEGATIVE**: 任何 arm R@10 < 0.0942 (task320 Arm A 最低) → alpha sweep 显著退化

## R10 推进条件

- 4×L40S 可用 (R7: 1 arm 1 GPU)
- task320 5-arm 全部完成 (含 Arm C ep101 R-Drop) → 已完成
- Issue #38 GitHub PARTIAL GO 状态记录 → 已完成
- verdict 占位: `verdicts/task328_issue38_followup_rdrop_alpha_sweep_result.md` (待写)

## Next steps

1. 写 `scripts/task328_issue38_followup_rdrop_alpha_sweep.py` (基于 task320 脚本, 加 alpha 参数)
2. 写 4 个 launcher shell (Arm A α=0.5 / Arm B α=1.0 / Arm C α=2.0 / Arm D α=4.0)
3. 4 GPU 并行启动, ~3.5 hr
4. Stage 4 K=100 eval per arm (~1 min/单 arm)
5. 收集 R@10/R@20/NDCG@5/10/20 → 写 task328 verdict
6. Issue #38 → re-comment + re-close with alpha sweep finding

## R9 compliance

- 本任务编号 #328 = max(327) + 1 ✅
- 跟 task320 (Issue #38 5-arm Layer 1) + task318 (Issue #38 5-arm Layer 1 50ep proxy) 连续
- Issue #38 = 同 issue 跨多 task (Layer 1 task320 → Layer 2 task328)
- 不破坏 R9 连续性

## Critical caveats (R11.5 风险 transparent)

1. **α sweep 范围风险**: α=0.5 可能太弱 (≈ baseline), α=4.0 可能太强 (underfit). 如果 [0.5, 4.0] 都没突破 anchor, 后续需要 α ∈ {0.1, 0.3, 8.0} 极值探索
2. **R-Drop 计算代价**: 2 forward pass / batch, 200 epoch × 4 arm 估 3.5 hr (跟 Adam 等价 wall time 因为 T5-mini 5.5M 够小)
3. **Val/test gap 仍 -0.021**: 即使找到最优 α, gap 仍由 Stage 2 SID 配置决定. 期望: R-Drop α tuning 抬 absolute level, 不缩 gap
4. **跟 task327 synergy**: task327 K=256 + Issue #30 synergy 当前 RUNNING (Stage 3 GPU 1 PID 1531589 ep47/200). task328 vs task327 = R-Drop alpha vs K=256, **互补非竞争** (一条攻 Stage 3 协议, 另一条攻 Stage 1/2 capacity)
5. **Issue #38 Layer 3 follow-up**: 若 alpha sweep 找到 ≥ 0.1053, 后续 Layer 3 = R-Drop + task194_k0256 协同 (R-Drop on K=256 baseline)

## Reference verdicts (跟当前 backbone 一致)

- task320 Arm C R-Drop α=1.0 ✅ GO (R@10=0.1034, +1.4%)
- task318 Adam=AdamW 50ep 数学等价 (optimizer 不是 R@10 杠杆)
- task194_k0256 anchor R@10=0.1053 (+3.2% baseline)
- task301 Issue #30 GO R@10=0.1022 (Stage 1/2 唯一 GO endpoint)
- Issue #38 Layer 1 task320 verdict (5-arm Stage 3 协议改造, PARTIAL GO)
