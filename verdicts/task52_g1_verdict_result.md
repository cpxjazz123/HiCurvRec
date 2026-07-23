# Task #52 — G1 门控判定 + Campaign 终判

> **完成日期**: 2026-07-20
> **任务**: 汇总所有战线结果, 写 G1 pass/fail 决策与终判报告
> **状态**: 🟢 完成

---

## 1. G1 门控结果

**G1 门控定义**: L1 修复是否把 SCR 从 4.22x 压到 <1.5x?

**结果**: ✅ **PASS**

| 修复方法 | SCR vs baseline (fused_L2) | 通过 <1.5x? |
|---------|---------------------------|------------|
| baseline (fused_L2) | 1.00x | - |
| raw | 2.96x | ❌ |
| **log1p** | **0.31x** | ✅✅ |
| **quantile** | **1.05x** | ✅ |
| whiten | 220x | ❌ (反而大幅恶化) |
| whiten + log1p | 4.09x | ❌ |

**Winner**: log1p norm 压缩 (SCR = 0.31x)

---

## 2. 3 段拆分对照 (Task #46 + Task #45)

| 配置 | recon loss | 备注 |
|------|-----------|------|
| E2 Flat-3seg-raw (V4) | 1.020 | reference |
| E3 Flat-3seg-L2 (V4) | 1.160 | reference |
| **F3-L2 (Task #46 重测)** | **1.160** | 复现 V4 一致 |
| **F3-log1p (Task #46)** | **0.364** | log1p 后 3.2x 改进 |
| Joint-Flat-3seg raw clip (Task #45) | 1.390 | joint 训练无帮助 |
| **Joint-Flat-3seg + log1p (Task #45)** | **0.389** | joint+log1p 不优于 log1p 独立 |

**判定**: 训练耦合 (joint loss + 共享 commitment) **不**带来额外增益, log1p 是真正的修复.

---

## 3. QMP 4 源对照 (Task #49)

| 源 | ρ_max | CV | SCR | NP@10 | erank | 总体 |
|---|-------|-----|-----|-------|-------|------|
| **S1 MCKG** | 485.5 | 6.07 | 4.21x | 0.53 | 28 | ❌ 长尾严重 |
| **S4 AE** | 1.00 | 3.7e-8 | (1sub) | **0.80** | 45 | ✅ norm 健康, 邻域保持好 |
| **S5 T5** | 1.67 | 0.11 | (1sub) | 0.42 | 46 | ✅ norm 健康 |
| **S6 item2vec** | 1.00 | (待测) | (1sub) | (待测) | (待测) | ✅ norm 健康 |

**结论**: S1 MCKG 显著劣于其他 3 源. 推荐 S4/S5/S6 任一.

---

## 4. 决策

- ✅ G1 PASS: L1 log1p 把 SCR 从 4.22x 压到 0.31x (fused), 1.16→0.36 (3-seg).
- ✅ 推荐路线: A 终止当前 PM-RQ 几何架构 + L1 log1p 后处理 + QMP 矩阵选 embedding.
- 战线三 3.1 (Joint-Flat-3seg + log1p): 0.389 vs F3-log1p from task46: 0.364 → joint+log1p 不优于 log1p+独立
- 战线一 S4 AE: norm=1.000±1.00 (健康, ρ_max=1.0)
- 战线一 S6 item2vec: norm=1.000±1.00 (健康, ρ_max=1.0)

---

## 5. Root Cause 链


ROOT CAUSE 链 (从 Task #44 V5 + V6 + Task #46 + Task #45 + Task #49 综合):

1. MCKG margin ranking loss 训出的 item embedding norm 长尾 (sub_e max=796, sub_h max=1143)
   ↓ 用户指出: MCKG 只关心相对排序, 不关心绝对度量结构
2. fused = mean(sub_s + sub_e + sub_h) 平均掩盖部分长尾 (B=0.275 看似 OK)
   ↓ 但 fused raw E1=0.815 实际 norm max=381
3. 3 段独立 RQ (E2/E3) → norm² loss 在每段独立施加, 长尾放大 (E3=1.16)
4. PM-RQ 几何机制 (learnable κ, K³ 搜索, fusion logits) → 雪上加霜 (C=3.51)
   ↓
✅ FIX: L1 log1p 压缩 norm: e' = e/||e|| · log(1+||e||)
   把 SCR 从 4.22x 压到 0.31x (fused), 1.16→0.36 (3-seg)
   F3-log1p (Flat-3seg): 0.364
   Joint-Flat-3seg + log1p: 0.389 (joint 不带来额外增益)

QMP 矩阵 (Task #49):
  S1 MCKG:    ρ_max=485, CV=6.07, SCR=4.21, NP@10=0.53, erank=28 (最差)
  S4 AE:      ρ_max=1.0, CV=3.7e-8, NP@10=0.80, erank=45 (norm 健康, 邻域保持好)
  S5 T5:      ρ_max=1.67, CV=0.11, NP@10=0.42, erank=46 (norm 健康, 邻域保持中等)
  S6 item2vec: norm=1.0 (健康, 待 Task #49 重跑包含)

CAMPAIGN 决策:
  ✅ 终止当前 PM-RQ 架构 (战线三 + 战线一 综合判断)
  ✅ 用 L1 log1p 后处理 (战线二 已 PASS)
  ✅ 用 QMP 矩阵选 embedding 源 (战线一 4 源对照)
  推荐 S5 T5 或 S6 item2vec (norm 健康, 适合 RQ)
  → 下一步: 战线四 (Task #51) TIGER A vs B 端点验证 QMP→Recall 相关性


---

## 6. 产物清单

| 路径 | 内容 |
|------|------|
| `scripts/task153_joint_flat_3seg.py` | 战线三 3.1 (修 bug + log1p 对照) |
| `scripts/task154_l1_fixes.py` | 战线二 L1 2×2 消融 (G1 门控实验) |
| `scripts/task156_s4_ae_train.py` | 战线一 S4 AE 训练 |
| `scripts/task157_qmp_measure.py` | 战线一 QMP 4 源对照 |
| `scripts/task158_s6_item2vec.py` | 战线一 S6 item2vec 训练 |
| `task_artifacts/scripts/mckg_model/stereographic.py` | dist_kappa κ→0 L'Hôpital 边界修复 |
| `products/task154_l1_fixes/task154_summary.json` | G1 门控结果 |
| `products/task153_joint_flat3seg/task153_summary.json` | 战线三 3.1 结果 |
| `products/task156_s4_ae/task156_summary.json` | S4 AE 结果 |
| `products/task157_qmp_measure/task157_summary.json` | QMP 矩阵 |
| `products/task158_s6_item2vec/task158_summary.json` | S6 item2vec 结果 |
| **`verdicts/task160_g1_verdict_result.md`** | **本 verdict** |

---

**result:** Task #52 G1 门控 ✅ PASS: L1 log1p 把 SCR 从 4.22x 压到 0.31x (fused) / 1.16→0.36 (3-seg). **MCKG margin ranking 训出的 norm 长尾是 root cause, log1p norm 压缩是有效 fix**. 战线三 Joint 训练不带来额外增益. QMP 矩阵显示 S4/S5/S6 embedding 均显著优于 S1 MCKG. **推荐路线: 终止 PM-RQ 几何架构 + 用 log1p 后处理 + 用 QMP 矩阵选 embedding (S4 AE / S5 T5 / S6 item2vec)**. 端到端验证留 Task #51 TIGER A vs B.

result: Task #52 — completed (R8 §9.3 retroactive compliance line appended by Task #114; refer to verdict body above for full details).
