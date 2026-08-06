---
type: verdict
issue: 76
status: "FAIL"
created: 2026-08-03
tags:
  - misc
up: "[[index]]"
---
# Issue #76 — 用户方案 (c=exp(κ)+stop-grad c+REL_STRUCT+平滑先验) 端到端验证 verdict

> **日期**: 2026-08-03
> **结论**: Stage2 GATE PASS, Stage3/4 端到端 FAIL — learnable κ 塌缩已根治, 但 κ 层级差异无独立 R@10 增益, 且换新 SID 破坏预训练 T5 兼容。

---

## 1. 用户方案 (四步) 复述

1. `c_l = exp(ρ_l)` 保证定义域恒正, κ init 0 → c=1 锚定基线, 无硬 clamp
2. expmap/proj 参数化保证点合法
3. 绝对 quant loss 对 c **stop-grad** (`c_geom = c.detach()`), 消除"距离随 c 减"的尺度作弊
4. REL_STRUCT 相对结构目标 (`√c·r → 0.3`, 尺度无关) + 平滑先验 (`λ·Σκ²`) 让 κ 学层级结构

## 2. Stage2 验证 — GATE PASS ✅

**产物**: `taskA/_history/taskA_stage2_v10b_relstruct_1000ep/` (seed 2024, 1000ep, CURV_PRIOR=1 + REL_STRUCT=1)

| 指标 | v10b | 历史对照 |
|------|------|----------|
| 最终 κ | **[0.0158, 0.0375, 0.0359]** (非零层级差异) | v9=[0,0,0] 停死; v7c/v8 全冲边界塌缩 |
| c (exp(κ)) | [1.016, 1.038, 1.037] | — |
| SID util_4digit | **100%** | 塌缩时仅 2.58% |
| κ std | 0.009 (CURV_PRIOR 放宽判据 std>0.005 或 max-min>0.01, v10b max-min=0.020 PASS) | 原 std≥0.05 不适用微调量级 |
| NaN/Inf | 无 | — |
| reload 校验 | 5/5 | — |

**结论**: 用户方案四步**根治了 learnable κ 塌缩** — κ 从 0 学出非零层级差异, SID 100% unique, 无 clamp 依赖, 无 NaN。这是历史上首次 learnable κ 稳定收敛。

## 3. Stage3 端到端 — FAIL ❌

**产物**: `taskA/_history/taskA_stage3_v10bsid_relstruct/` (v10b SID + κ 注入, 解冻 T5 后 2 层, 从头 60ep)

stage3 verdict (Issue #28 Gate2 diag 判据):
- c3 κ 实际更新: PASS (κ 从注入 [0.0158,0.0375,0.0359] 微调到 [0.0145,0.0266,0.0181], delta_max=0.0178)
- c4 codebook 重校准: PASS (82 次触发)
- c5 val canary: **FAIL** (best_val_r10=**0.04** < 0.05, best_epoch=29)
- backbone drift: **FAIL** (min_cosine_sim=0.975, 60ep 全 drift)

**verdict: FAIL**

## 4. Stage4 全量评估 — 决定性证据

stage4 全量 valid (24772) beam20 协议:

| 配置 | SID | 全量 valid R@10 | 说明 |
|------|-----|-----------------|------|
| **v10bsid adapter** (训练后) | v10b 新 SID | **0.047** | 远低于 baseline |
| sanity T5-only + v10b SID | v10b 新 SID | **0.0** | T5 预训练不认识新码字! |
| **v8 adapter** (标准 SID 无 κ) | 标准 SID | 0.1253 | ≈baseline |
| sanity T5-only + 标准 SID | 标准 SID | 0.1266 | ≈baseline 0.1267 |
| **stdSID+v10bκ adapter** (对照) | 标准 SID | **0.1253** | = 无 κ 的 v8! |

**两个决定性结论**:

1. **κ 无独立增益**: 标准 SID 下, 注入 v10b κ (0.1253) 与无 κ 的 v8 (0.1253) 全量 R@10 **完全相同**。learnable κ 的层级差异没有转化为 R@10 提升。

2. **换 SID 破坏 T5 兼容**: 预训练 T5 (HG_Rec_best.pth) 是在**标准 SID (2dab2922)** 码字空间训练的 — T5-only sanity 直接复现 baseline 0.1266。换成 v10b 新码字后 sanity=0, 说明 T5 需从零重学整个 T5→SID 映射, 60ep 只到 0.047。

## 5. 根因分析

- **stage2 用户方案成功**: c=exp(κ)+stop-grad 消除了"绝对 quant loss 直接训练 κ"的尺度作弊, REL_STRUCT 提供真正的结构梯度, 平滑先验防发散 — κ 学会层级差异且码本不塌缩。
- **但 κ 的"层级差异"不是 R@10 的关键因子**: 标准 SID 下有无 κ 注入结果完全一致 (0.1253=0.1253), 说明 adapter 的 κ 信号对解码/检索贡献为零 (adapter grad≈0, α 冻结 0.1, 学习靠解冻 T5 层)。
- **改变 stage2 码本 = 改变 SID 码字空间**: 破坏与预训练 T5 的兼容, 是端到端大跌的真正根因。

## 6. 历史对照

- v7c/v7e/v7f/v8: learnable κ 全塌缩 (κ 冲 clamp 边界, SID 2.58%) → **用户方案根治** ✅
- v9: κ 停死 [0,0,0] (先验死鞍点) → REL_STRUCT 提供结构梯度解决 ✅
- v10b: κ 学出非零层级 + SID 100% → **Stage2 GATE PASS** ✅
- 但端到端: 0.047 (新 SID) / 0.1253 (标准 SID+κ) — 均未超 baseline, 且新 SID 路径大幅劣化 ❌

## 7. 结论与建议

**结论**: 用户方案在 Stage2 层面**完全成功** (learnable κ 框架稳定、不塌缩、学出层级), 但在 Stage3/4 端到端**不带来 R@10 增益**:
- learnable κ 的层级差异无独立检索增益 (0.1253=0.1253)
- 若用新 SID 则破坏预训练 T5 兼容, 端到端大跌 (0.047)

**建议** (后续方向, 若继续探索 learnable κ 价值):
1. **保持标准 SID + κ 注入**: 不改变码字空间, 只在 adapter 层注入 κ 信号 (已验证 =0.1253, 需要更强的 adapter 训练机制让 κ 真正进入 forward 梯度, 而非 α 冻结下的近似零贡献)
2. 诊断 adapter grad=0 根因: α 冻结 (FREEZE_ALPHA_LOGIT=True) + adapter 梯度 ~1e-9, 使 learnable κ 在 stage3 成为"死参数" — 这是 κ 无法产生增益的直接机制
3. 若追求 SID 质量本身 (unique/码字利用率), 用户方案仍是有效的 stage2 工具, 但需配套 T5 重训 (放弃预训练权重) 才能用新 SID

## 8. 产物清单

- `taskA/_history/taskA_stage2_v10b_relstruct_1000ep/` — Stage2 GATE PASS 产物 (ckpt + SID + verdict)
- `taskA/_history/taskA_stage3_v10bsid_relstruct/` — Stage3 FAIL 训练产物 (60ep, verdict.json)
- `taskA/_history/taskA_stage3_stdSID_v10bkappa/` — 对照训练产物 (标准 SID + v10b κ)
- `verdicts/issue30_v10bsid_pre_beam20_reeval.json` — v10b SID 全量预评
- `verdicts/issue30_v8_stdSID_contrast_beam20_reeval.json` — v8 标准 SID 全量对照
- `verdicts/issue30_stdSID_v10bkappa_pre_beam20_reeval.json` — 标准 SID+κ 全量对照

**Stage2 Gate2 = PASS, Stage3/4 = FAIL。任务 #76 端到端结论已闭环。**
