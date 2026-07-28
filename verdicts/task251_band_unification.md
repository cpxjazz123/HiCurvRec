# Phase 0 一致率带口径锁定 (与 task236 collision metric 同规格)

> **背景**: 本仓库自 task231 (2026-07-26) 起, Phase 0 一致率被用作 Stage 1 是否该跑的闸门, 但 4 条带定义散落在多处 verdict / description 里, 90-95% 灰区无明示. Issue #15 (2026-07-28) 把这一缺口作为 Gate 0 audit 对象. 本文件与 `verdicts/task236_collision_metric_unification_result.md` 同规格, 锁定权威定义.

## 1. 4 条带的权威定义 (沿用现状, 不重画)

| 带 | 范围 | 语义 | 原始出处 |
|----|------|------|----------|
| **TOO_STRONG** | `< 50%` | 几何主导过头 | `descriptions/task231_pck_phase0_ckrange_sweep.md:35` "`< 50% → TOO_STRONG (几何主导过头)`" |
| **OPEN** | `60% ≤ x ≤ 90%` | per-codeword κ 有区分力 | `descriptions/task231_pck_phase0_ckrange_sweep.md:34` "`60-90% → ✅ OPEN (per-codeword κ 有区分力)`" |
| **FAIL** | `> 95%` | per-codeword κ 影响被淹没 | `descriptions/task231_pck_phase0_ckrange_sweep.md:33` "`> 95% → FAIL (per-codeword κ 影响被淹没)`" |
| **Safety** | `agreement < 0.90` (单层协议) | Stage 1 L0 利用率 ≥ 90% 必要条件 | `verdicts/task241_issue11_gate0_perlayer_ck_combo_pass.md §1` "三层全 OPEN (60-90%) + 全 agreement < 0.90 (§6.7.4 safety)" |

**灰区 50-60% + 90-95%**: Issue #15 §4 已诚实标注该区间三种口径给出三种结论. 本文件锁定处置:
- **`50% ≤ x < 60%`** → **WEAK** (新分类): 一致率偏低, per-codeword κ 起作用但力度弱, 训练动力学可能拉到坍缩也可能拉到健康. **不算 OPEN, 不算 TOO_STRONG, 单独标 WEAK**.
- **`90% < x ≤ 95%`** → **WEAK_OPEN** (新分类): 一致率偏高, 几何起作用但接近被淹没边界. **不算 OPEN, 不算 FAIL, 单独标 WEAK_OPEN**. **本项目唯一健康 L1/L2 (task222 92.37% / 94.04%) 落在此带** —— 这是 Issue #15 §2 的核心证据, 不能用纯 FAIL 拒掉.

> ⚠️ WEAK / WEAK_OPEN 是 Issue #15 §4 §5 的诊断用分类, **不构成 go/no-go 闸门**. Gate 1/2 用它们做交叉分析.

## 2. 与 collision 口径的对应关系

| collision 口径 (task236) | Phase 0 一致率口径 (本文件) |
|--------------------------|-----------------------------|
| `collision_rate = 1 − uniqueness_rate` | `agreement = mean(argmin_poincare == argmin_euclidean)` |
| Gate 0 通过: `collision < 0.37` (task222 best known) | Gate 0 通过: 三层全 OPEN (60-90%) + 全 agreement < 0.90 |
| §反证 collision 不是 R@10 杠杆 (Issue #12 / task237) | **本 issue §反证 一致率也不是 Stage 1 存活杠杆** |

## 3. 配对档案字段规范

每个配对点必须填:
- **run_id** (task220 / task221 / task222 / task235 / task242 ArmA / task242 ArmA+)
- **mechanism** (PC κ U(0.5,5) / Gromov weight=0.5 / hybrid / PC κ per-layer)
- **ckpt_epoch** (ep29 / ep14 / ep34)
- **per_layer**: `{L0, L1, L2}` × `{agreement, consistency, stage1_util, collision}`
- **source_files**: 每个数字的 verdict 路径 + 行号

## 4. 历史溯源

| 数字 | 原始来源 |
|------|----------|
| 一致率带 4 条 | `descriptions/task231_pck_phase0_ckrange_sweep.md §3` |
| `agreement < 0.90` safety | `descriptions/task234_hybrid_phase0_composition.md §阈值` + `verdicts/task241_issue11_gate0_perlayer_ck_combo_pass.md §1` |
| stop-loss (i) L0 ≥ 90% | `verdicts/task225_pck_stage4_eval_result.md §6.7.4` |
| collision 权威定义 | `verdicts/task236_collision_metric_unification_result.md` |

## 5. 适用范围

本口径锁定文件适用于:
- Issue #15 Gate 1/2 (PC κ 一致率 vs Stage 1 利用率同向关系 + 可行域存在性)
- Issue #13 Gate 1 (Möbius 残差算子 argmin 一致率; 已用 60-90% OPEN 带, 90-95% 改判 WEAK_OPEN)
- 后续任何 Phase 0 一致率相关的闸门判定

**不适用**: collision / utilization / entropy / Gini 等其它 SID 分布量, 见 `verdicts/task244_issue12_gate0_sid_distribution_result.md`.

---

result: **Phase 0 一致率带口径已锁定. 4 条权威带 + 2 条灰区新分类 (50-60% WEAK / 90-95% WEAK_OPEN). 灰区处置: WEAK / WEAK_OPEN 是诊断分类不构成闸门, Gate 1/2 用作交叉分析**.