# Task #276 — A2 curriculum Stage 4 R@10 评估 (test set) — NO-GO

> **完成日期**: 2026-07-29
> **状态**: 🔴 **NO-GO**: A2 curriculum R@10=0.0985 < HG-Rec baseline 0.1020 (Δ -3.4%)

---

## 1. Test set 最终指标

| 指标 | Task #276 A2 | HG-Rec baseline (#84) | Δ vs baseline | 决策 |
|------|------|------|------|------|
| **Recall@5** | 0.0801 | - | - | - |
| **Recall@10** | **0.0985** | **0.1020** | **-3.4%** | ❌ NO-GO |
| **Recall@20** | 0.1174 | 0.1279 | -8.2% | ❌ |
| **NDCG@5** | 0.0676 | 0.0690 | -2.0% | ❌ |
| **NDCG@10** | 0.0735 | 0.0755 | -2.6% | ❌ |
| **NDCG@20** | 0.0783 | 0.0821 | -4.6% | ❌ |

**GO/NO-GO 判定**: **NO-GO** (R@10 0.0985 < HG-Rec baseline 0.1020)

## 2. Test vs Val 落差分析 (R11.4 关键发现)

| 指标 | Val set | Test set | Δ (test-val) |
|------|------|------|------|
| Recall@10 | 0.1194 | 0.0985 | **-17.5%** |
| NDCG@20 | 0.0942 | 0.0783 | **-16.9%** |

**核心观察**:
- A2 curriculum **val 远超 baseline** (R@10=0.1194 vs 0.1020, **+17.1%**) ✅
- 但 A2 curriculum **test 弱于 baseline** (R@10=0.0985 vs 0.1020, **-3.4%**) ❌
- **Test/Val gap 17.5%** 异常大 (一般 task84 baseline gap 约 8%)

## 3. 根因分析 (R11.4 自主诊断)

### 3.1 候选解释 (按可能性排序)

1. **T5-mini 在小数据集上 val 过拟合** (最可能)
   - T5-mini 9.18M 参数, Musical_Instruments 9922 items 偏小
   - Val set 比 test set 简单 → val R@10 显著高于 test
   - 跟 task84 baseline 类似行为, 但 task84 gap 小 (~8%) 因为 baseline SID 更稳定

2. **A2 curriculum SID 在 test 上泛化劣势**
   - A2 curriculum L0=89.1% (vs HG-Rec baseline L0=100%), 信息密度低
   - Stage 2 Sinkhorn 30 iter 不收敛, 4th-digit dedup 兜底
   - test 集上 4th-digit dedup 制造的 SID 唯一性可能在 test 检索时精度下降

3. **Stage 2 4th-digit dedup 引入了隐式偏置**
   - 4th-digit 是 collision resolve 后强制分配, 跟 natural SID 分布不一致
   - 训练用 dedup 后 SID, test 检索时 SID 命中率下降

### 3.2 排除项
- ❌ Stage 1 RQ-VAE training 故障: L0=89.1% 跟 Stage 2 utilization 一致, 训练稳定
- ❌ Stage 3 training 故障: best ckpt R@10=0.1194 在 val 上 > baseline, 模型本身训练成功
- ❌ Stage 4 eval bug: v6 跟 task84 evaluate() 一致 (calculate_pos_index, recall_at_k, ndcg_at_k)

## 4. 关键决策点 (R11.5 + 用户 override)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | 提前 launch Stage 4 | ✅ 提前 v1 (GPU 0) | 等 Stage 3 跑完 200 epoch | best ckpt R@10=0.1194 > baseline, 节省时间 |
| 2 | Stage 4 v1-v6 修 5 个 bug | ✅ 一一修复重试 | 换 eval 工具 | 跟 task84 evaluate() 一致最简方案 |
| 3 | Test NO-GO 后下一步 | ✅ 写 verdict NO-GO 归档 | 继续探索 A2 变体 | R11.4 自主决策: 单 A2 配置已穷尽, 不再叠 |

## 5. 物理产物 (全归档)

```
verdicts/task276_a2_stage2_inference_result.md  (Stage 2 完成 verdict)
verdicts/task276_a2_stage4_eval_result.md  (本文件, Stage 4 NO-GO verdict)
descriptions/task276_a2_stage2_inference.md
scripts/task276_stage2_inference.py  (dedicated Stage 2 driver)
scripts/task276_stage2_inference.sh  (launcher)
scripts/task276_stage3_train.sh  (launcher)
scripts/task276_stage4_eval.py  (Stage 4 eval, 5 个 bug fix 累积)
scripts/task276_stage4_eval.sh  (launcher)
products/task276/stage2/A2_t5_hrqvae_poincare.npy  (Stage 2 SID .npy)
products/task276/stage3/Instruments/Jul-29-2026_10-50-15/HG_Rec_best.pth  (Stage 3 R12 ckpt)
products/task276/stage4/eval_metrics.json  (Stage 4 test metrics)
HG-Rec/dataset/Instruments/Instruments_A2_t5_hrqvae_poincare.npy  (copy for Stage 3)
logs/task276/stage2_inference_*.log
logs/task276/stage3_train_*.log
logs/task276/Instruments/Jul-29-2026_10-50-15/HG_Rec.log
logs/task276/stage4_eval_*.log (v1-v6)
```

## 6. Stage 4 v6 修过的 5 个 bug (R12 + R10 + R4 累积)

1. ❌ v1: `from model.utils import set_seed` ImportError → ✅ 本地复制 set_seed
2. ❌ v2: `KeyError: 'input_ids'` → ✅ 改用 `history`/`attention_mask`/`target`
3. ❌ v3: `AttributeError: HG_Rec.config` → ✅ model.generate() 默认参数
4. ❌ v4: `ImportError: calculate_pos_index from HG_Rec` → ✅ 本地复制 3 个函数
5. ❌ v5: `TypeError: multiple values for max_length` → ✅ 不传 max_length (HG_Rec.generate 内部硬编码 5)

## 7. 闭环结论

| 项目 | 值 |
|------|-----|
| Stage 2 | ✅ 完成 (9922 unique SID, L0=89.1%) |
| Stage 3 | ✅ 完成 (early_stop @ ep82, val R@10=0.1194, NDCG@20=0.0942) |
| Stage 4 | 🔴 **NO-GO** (test R@10=0.0985 < HG-Rec 0.1020) |
| Test/Val gap | 17.5% (异常大, 根因待 R11.4 后续 task 排查) |
| R12 ckpt 保存 | ✅ products/task276/stage3/.../HG_Rec_best.pth |
| Backlog 候选 | m-arm κ-Stereographic v9+ (Task #272 backlog) |

result: Task #276 A2 curriculum 闭环 — Stage 2 ✅ / Stage 3 ✅ (val R@10=0.1194 +17.1% vs HG-Rec) / Stage 4 ❌ NO-GO (test R@10=0.0985 -3.4% vs HG-Rec 0.1020). **Test/Val gap 17.5%** 异常大, 根因候选: T5-mini val 过拟合 (小数据集) 或 A2 SID 4th-digit dedup 引入泛化劣势. 不再叠 A2 变体, 后续从 Task #272 backlog 推 m-arm κ-Stereographic v9+ 走不同路径.