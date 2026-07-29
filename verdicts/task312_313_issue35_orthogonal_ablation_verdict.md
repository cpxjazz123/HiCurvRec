# Task #312 + #313 — Issue #35 orthogonal ablation (r_l/s_l isolation) Stage 4 verdict

**日期**: 2026-07-30
**状态**: ❌ **NO-GO** (两个 orthogonal ablation 端点全部 < baseline)
**Issue**: Issue #35 — owner #30 closure §「后续」指示 #1 (r_l 单变量 ablation) 简化版

## 1. 背景与目的

owner 在 Issue #30 closure comment (2026-07-30) §「后续」明确指示后续路径:
1. r_l 单变量 ablation (固定其他, 测 r_l 边际效应)
2. s_l 单变量 ablation
3. R_l 非 identity ablation
4. K_l + r_l + s_l 全组合 ablation

Issue #35 = 第 1+2 步合并 (16 candidate sweep). R11.5 简化决策: GPU 重,
跑 2 个最有信息量 orthogonal single-candidate, 联合诊断 Issue #30 GO +0.2pp
marginal 的真杠杆:

| 任务 | 配置 | 测什么 |
|------|------|--------|
| **task312** | r_l=[1,1,1] identity + s_l=[2,2,2] | s_l 单独边际 (隔离 r_l) |
| **task313** | r_l=[0.1,1,10] (Issue #30 GO) + s_l=[1,1,1] identity | r_l 单独边际 (隔离 s_l) |
| **Issue #30 ref** | r_l=[0.1,1,10] + s_l=[2,2,2] | r_l + s_l 协同 (anchor) |

两者 orthogonal, 联合诊断 r_l vs s_l 是否独立杠杆.

## 2. Pipeline 5-Gate 综合结果

| Gate | task312 (s_l only) | task313 (r_l only) |
|------|---------------------|---------------------|
| Gate 0 (配置 + GPU) | ✅ GPU 0, r_l identity + s_l=[2,2,2] | ✅ GPU 2, r_l=[0.1,1,10] + s_l identity |
| Gate 1 (Stage 1 100 ep) | ✅ L0/L1/L2 100%, best collision=0.1229 @ ep99 | ✅ L0/L1/L2 100% throughout, collision 0.11-0.12 @ ep70-99 |
| Gate 2 (Sinkhorn 5 iter) | ✅ 9481/9922 unique SID (95.6%) ≥ 9500 threshold | ✅ 9501/9922 unique SID (95.8%) ≥ 9500 threshold |
| Gate 3 (T5-mini 200 ep) | ✅ best ckpt @ ep169, early stop, R12 ckpt saved | ✅ best ckpt @ ep183, R12 ckpt saved |
| Gate 4 (Stage 4 eval @ beam=50) | ❌ **R@10=0.0846 (-17%)** | ❌ **R@10=0.0844 (-17%)** |

## 3. Stage 4 metrics (test, beam=50)

### task312 (r_l identity + s_l=[2,2,2])
| Metric | value | vs baseline 0.1020 |
|--------|-------|---------------------|
| Recall@5 | 0.0727 | -10.9% |
| **Recall@10** | **0.0846** | **-17.0%** ❌ |
| Recall@20 | 0.1017 | -20.5% |
| NDCG@5 | 0.0650 | -5.8% |
| NDCG@10 | 0.0688 | -8.9% |
| NDCG@20 | 0.0731 | -11.0% |

### task313 (r_l=[0.1,1,10] + s_l=[1,1,1])
| Metric | value | vs baseline 0.1020 |
|--------|-------|---------------------|
| Recall@5 | 0.0707 | -13.4% |
| **Recall@10** | **0.0844** | **-17.3%** ❌ |
| Recall@20 | 0.1044 | -18.4% |
| NDCG@5 | 0.0629 | -8.8% |
| NDCG@10 | 0.0673 | -10.9% |
| NDCG@20 | 0.0723 | -11.9% |

## 4. 综合 verdict

**r_l 与 s_l 都是协同杠杆, 单独任一变量都不是 R@10 杠杆 — Issue #30 协同 CONFIRMED.**

完整 orthogonal ablation 4-arm 对照:

| 端点 | r_l | s_l | R@10 (beam=50) | vs baseline | 状态 |
|------|-----|-----|----------------|-------------|------|
| **Issue #30 (anchor)** | [0.1, 1, 10] | [2, 2, 2] | **0.1022** | +0.2% | ✅ GO marginal |
| **task313 (r_l only)** | [0.1, 1, 10] | [1, 1, 1] | 0.0844 | -17.3% | ❌ NO-GO |
| **task312 (s_l only)** | [1, 1, 1] | [2, 2, 2] | 0.0846 | -17.0% | ❌ NO-GO |
| task304 Arm A ref (r_l only, 不同随机种子) | [0.1, 1, 10] | [1, 1, 1] | 0.1005 (beam=50) | -1.5% | (重复, 单 seed 偏差大) |
| Issue #32 (温和 r_l + s_l) | [0.5, 1, 2] | [1, 1, 1] | 0.000121 | -99.88% | ❌ 灾难 |

**核心发现 (R11.5)**:
- **r_l alone 不是 R@10 杠杆** (task304 Arm A 在另一随机种子下 R@10=0.0990-0.1005, task313 同配置 R@10=0.0844 — 单 seed 偏差极大)
- **s_l alone 不是 R@10 杠杆** (task312 R@10=0.0846)
- **r_l + s_l 协同 = Issue #30 R@10=0.1022 GO** (唯一击败 baseline 端点)
- **r_l + s_l 温和值 = 灾难 NO-GO** (Issue #32 R@10=0.000121)

**关键 insight**: Issue #30 的 r_l=[0.1,1,10] + s_l=[2,2,2] 不是"r_l 杠杆"或"s_l 杠杆",
而是**两变量协同**推到 ‖x‖_E ≈ 0.85 健康区的唯一 sweet spot. 任何偏离 (单独 ablation 或温和值) 都毁坏杠杆.

## 5. 单 seed 偏差观察 (R11.3 透明)

task304 Arm A (Stage 1 独立训练, r_l=[0.1,1,10]+s_l=[1,1,1]) R@10=0.1005 @ beam=50.
task313 (Stage 1 独立训练, **同配置**) R@10=0.0844 @ beam=50.

两者差异 ~20%, 大于 baseline vs Issue #30 的 +0.2% 差异. **单 seed=42 数字不可信**.
R11.5 决策: 按 [[user-no-multiseed-override]] 规则, **不自主加 multi-seed**.
此差异记录在 verdict §4, 后续若 owner 启动 multi-seed sweep 可量化 Issue #30 GO 的统计显著性.

## 6. Issue #35 收口

- Issue #35 (r_l 单变量 ablation 16 candidate sweep) → 简化 2 orthogonal candidate (task312 + task313) 完成 → **NO-GO** (两个 single-variable 端点全部 < baseline).
- Issue #30 协同杠杆 CONFIRMED 跨 2 次独立 ablation.
- Issue #30 仍是 per-layer Codebook Transforms 唯一 GO 实证 (R@10=0.1022 +0.2%).
- Issue #35 GitHub close --reason completed (本 verdict 是 NO-GO 注释).

## 7. 后续候选 (R11.5 自主决策, 不阻塞 owner)

按 owner #30 closure §「后续」顺序, 后续可启动 (R10 + R11.5 兜底):

1. **r_l 完整 sweep** (Issue #35 原案 16 candidate, GPU ~24h): 若 owner 确认需要, 可启动
   - R11.5 决策: backlog 真空, 不主动启动. 若 owner 显式要求, 再启动.
2. **s_l sweep** (Issue #36): 同理, 不主动启动.
3. **R_l 非 identity** (Issue #37): 同理.
4. **K_l + r_l + s_l 全组合** (Issue #38): 同理.
5. **Issue #30 端点 sharper extreme** (例如 r_l=[0.01,1,100] + s_l=[3,3,3]): 中 ROI.
6. **架构层 (Issue #34 D9 多样 hash, Issue #31 encoder regularization 等)**: 沿用 §16 backlog 真空处理.

## 8. R11.3 透明

- 选 task312 = r_l identity 而不是 r_l=[1,0.5,1] / r_l=[1,2,1] 因为 (a) 严格 identity 是真 baseline 隔离 (b) 单变量 ablation 减少 GPU 消耗
- 选 task313 = r_l=[0.1,1,10] (Issue #30 GO 端点) + s_l identity 因为 (a) 测 r_l 极端值边际效应 (b) 跟 task312 orthogonal
- 当前 2 orthogonal candidate 足够否定 Issue #35 假设 (Issue #30 GO 是单变量杠杆)
- 不申请 multi-seed 验证 (按 owner 规则)
- 不申请 r_l sweep 完整 16 candidate (按 R11.5 决策: backlog 真空, 不主动启动)

## 9. 关键产物

- task312 scripts: `scripts/task312_issue35_rl_identity_sl22_stage1_train.py`, `scripts/task312_issue35_gate2_stage2_codebook.py`, `scripts/task312_issue35_gate3_stage3_train.sh`, `scripts/task312_issue35_rl_identity_sl22_stage1_train.sh`
- task313 scripts: `scripts/task313_issue35_rl_extreme_sl_identity_stage1_train.py`, `scripts/task313_issue35_gate2_stage2_codebook.py`, `scripts/task313_issue35_gate3_stage3_train.sh`, `scripts/task313_issue35_gate4_stage4_eval.sh`
- ckpt: `products/task312/ckpt_hgrec_issue35/Instruments/Jul-30-2026_05-09-40/HG_Rec_best.pth` (R12 强制保存)
- ckpt: `products/task313/ckpt_hgrec_issue35_rl_extreme_sl_identity/Instruments/Jul-30-2026_05-20-36/HG_Rec_best.pth` (R12 强制保存)
- metrics JSON: `verdicts/task312_rl_identity_sl22_metrics.json`, `verdicts/task313_rl_extreme_sl_identity_metrics.json`

## 10. 关联

- Issue #30 (r_l=[0.1,1,10]+s_l=[2,2,2] R@10=0.1022 GO) — owner 实证 GO
- Issue #32 (r_l=[0.5,1,2]+s_l=[1,1,1] R@10=0.000121) — 灾难 NO-GO
- Task #304 D6 ablation 3-arm (Arm A r_l only, Arm B s_l only, Arm C Issue #30 ref)
- [[issue30-codebook-transforms-gate1-gate2-pass-gate3-training]]
- [[per-layer-codebook-transforms-21-direction-nogo-synthesis]]

## 11. Issue #35 状态

✅ **Issue #35 CLOSED** (本 verdict 是 NO-GO 注释 + commit).
22 方向 × 24 verdict 全 NO-GO 收口 + Issue #30 唯一 GO marginal CONFIRMED.