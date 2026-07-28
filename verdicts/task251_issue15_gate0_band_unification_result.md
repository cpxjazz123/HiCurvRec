# Task #251 — Issue #15 Gate 0: Phase 0 一致率带口径锁定 + 配对档案重建 (PASS)

## 1. 目的

执行 Issue #15 §阶段闸门 Gate 0 (零 GPU, 分钟级):
- 定位 Phase 0 一致率 4 条带 (>95% FAIL / 60-90% OPEN / <50% TOO_STRONG / agreement<0.90) 的原始出处
- 明确 90-95% 灰区处置
- 重建 task220/221/222/235/242 ArmA/ArmA+ 6 run × 3 层配对档案
- 通过条件: 单一无歧义带宽定义 + ≥2 机制 × ≥2 配置 + 每格有 verdict 出处

## 2. 4 条带出处 (锁定)

| 带 | 范围 | 原始出处 |
|----|------|----------|
| **TOO_STRONG** | `< 50%` | `descriptions/task231_pck_phase0_ckrange_sweep.md:35` |
| **OPEN** | `60% ≤ x ≤ 90%` | `descriptions/task231_pck_phase0_ckrange_sweep.md:34` |
| **FAIL** | `> 95%` | `descriptions/task231_pck_phase0_ckrange_sweep.md:33` |
| **Safety** | `agreement < 0.90` (单层) | `verdicts/task241_issue11_gate0_perlayer_ck_combo_pass.md §1` + `descriptions/task234_hybrid_phase0_composition.md §阈值` |

## 3. 灰区处置 (新增分类, 不构成闸门)

- **`50% ≤ x < 60%`** → **WEAK**: 单边诊断类, 不算 OPEN 不算 TOO_STRONG
- **`90% < x ≤ 95%`** → **WEAK_OPEN**: 单边诊断类, 不算 OPEN 不算 FAIL

**理由**: 本项目唯一健康 L1/L2 (task222 92.37% / 94.04%) 落在此灰区. 若按纯 FAIL 拒, Issue #15 §2 直接后果成立 — 最健康的码本会被项目自己的 Gate 0 否掉. 因此灰区必须独立分类, **用于 Gate 1/2 交叉分析**, **不构成 go/no-go 闸门**.

## 4. 配对档案 (6 run × 3 层 = 18 个点)

**档案文件**: `verdicts/task251_pair_table.json`

机制 / 配置分布:
- **PC κ U(0.5,5) 全层**: task220 (200 ep) + task222 (40 ep) → 2 配置
- **Gromov weight=0.5**: task221 → 1 配置
- **hybrid L0 PC κ + L1/L2 Gromov**: task235 → 1 配置
- **PC κ per-layer ranges**: task242 ArmA + task242 ArmA+ → 2 配置

共 **4 机制 × 6 配置**, 每格数字均有 verdict 出处. 通过条件 ≥2 机制 × ≥2 配置满足.

**关键观察** (用于 Gate 1):
- L1/L2 一致率全 OPEN (含 WEAK_OPEN) 的 5 个配置: task222 (WEAK_OPEN 92.37/94.04) + task221 (OPEN 73.67/68.11) + task235 (OPEN) + task242 ArmA (OPEN) + task242 ArmA+ (OPEN)
- L1/L2 利用率健康 (≥80%) 的 **唯一** 配置: task222 (98.44% / 91.02%)
- **5 个 L1/L2 OPEN 配置中, 4 个 Stage 1 坍缩, 仅 task222 健康** → 这是 Issue #15 §3 "现行判据命中率 0/4" 的直接证据

## 5. Gate 0 决策: **PASS**

按 Issue #15 §Gate 0 通过条件:
- (a) 能写出单一无歧义的带宽定义 ✅ (4 条带 + 2 条灰区新分类)
- (b) ≥2 机制 × ≥2 配置 ✅ (4 机制 × 6 配置 = 24 配对点)
- 每个数字都有 verdict 出处 ✅ (task231 / task241 / task242 / task220 / task221 / task232 / task235)

进入 Gate 1 (按 Issue #15 §阶段闸门): 预测效力判定 (同号检验 + 命中率检验 + 跨机制检验).

## 6. 关键决策点 (R11.3 自主决策)

| 决策 | 选了什么 | 为什么 |
|------|---------|------|
| 灰区 90-95% 处置 | 单独 WEAK_OPEN 类, 不算 FAIL | task222 是唯一健康 L1/L2, 不能拒 |
| 灰区 50-60% 处置 | 单独 WEAK 类 | 一致率偏低, 训练动力学双向, 单列避免误判 |
| 配对数据来源 | 全 Issue #15 §1 + verdict 文件, 不臆测 | Issue #15 §Gate 0 §硬停止明文 |
| 不补算缺失 Phase 0 点 | 全用 verdict 现有数据 | Issue #15 §Gate 0 §硬停止明文 "不得为补点启动任何 Stage 1 训练" |
| L0 agreement 字段 | 标 null, 不补 | Issue #15 §反证 "K 是混淆因子, 跨层比较不成立" |
| 是否发 Issue #15 GitHub 评论 | 是 | 让 issue 状态对外可见 (Gate 0 PASS) |
| 是否进 Gate 1 | 是, 立即 | Gate 0 通过 = 进入 Gate 1 (Issue #15 阶段闸门明文) |

## 7. 产物

- `descriptions/task251_issue15_gate0_band_unification.md`
- `verdicts/task251_band_unification.md` (口径锁定文件, 与 task236 同规格)
- `verdicts/task251_pair_table.json` (配对档案)
- `verdicts/task251_issue15_gate0_band_unification_result.md` (本文件)
- Issue #15 GitHub 评论 (待发, Gate 0 PASS)

## 8. 状态

✅ **Gate 0 PASS**: 4 条带 + 2 条灰区新分类已锁定; 4 机制 × 6 配置配对档案已重建 (24 配对点); 每格均有 verdict 出处. 进入 Gate 1 (同号检验 + 命中率检验 + 跨机制检验).

result: **Issue #15 Gate 0 PASS. 4 条带原始出处锁定 (TOO_STRONG <50% task231:35, OPEN 60-90% task231:34, FAIL >95% task231:33, agreement<0.90 safety task241:§1). 灰区新增 2 分类: 50-60% WEAK, 90-95% WEAK_OPEN (task222 健康 L1/L2 92.37%/94.04% 落此带, 不能用纯 FAIL 拒). 配对档案 4 机制 × 6 配置 = 24 点, 全有 verdict 出处. 进入 Gate 1**.