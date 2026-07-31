# Task #392 / Issue #99 [方向C Gate2] 基于公式修复的 SID/metadata 多样性准入 — FAIL 收口 (framework 完成, baseline 确认坍缩, 等 #97/#98 修复)

**日期**: 2026-07-31
**触发**: Issue #99 [方向C Gate2] 基于公式修复的 SID/metadata 多样性准入
**前置**:
- Issue #96 task389 HypPreEncoder SID FAIL (unique=1/9922, collision=99.99%, util 1.56%/0.78%/0.39%, metadata kappa/scale var=0)
- Issue #93 task386 metadata uniform FAIL (on_off_diff=1.179 真信号, shuffle_diff=0 跟 #90 一致)
- 依赖: Issue #97 (κ-Stereographic 公式修复) + Issue #98 (product dist 修复) PASS

**任务**: 建立 Stage2 SID/metadata 多样性准入的 audit 框架 + baseline 验证
**结果**: ❌ Gate 2 FAIL (依赖 #97/#98 修复, 当前 baseline 确认坍缩 baseline_collapse_confirmed=True)

---

## 1. R17 Gate 决策 (R20 强制详细)

### Gate 1 (= Stage 1): ⚠️ 复用 #86 PASS (commit 4ad7890) 但记录依赖风险
- **关键数据**:
  - #86 Gate 1 PASS (50 epoch 训练完成, util ≥90%, collision ≤20%)
  - **依赖风险**: #87 / #96 证明原 SID 生成坍缩 — SID unique 256 → 1 退化链
  - 必须等 #97 / #98 公式修复后, 重新跑 Gate 1 才能用新产物

### Gate 2 (= Stage 2 SID/metadata diversity): ❌ FAIL per spec (依赖 #97/#98 修复)
- **关键数据** (baseline audit, 应复现 #96 FAIL):
  - **SID unique = 64/9922 = 0.65%** ❌ (目标 ≥ 9500/9922, baseline 坍缩确认)
  - **SID collision rate = 99.35%** ❌ (目标 ≤ 0.20)
  - **L0 util = 100.00%** (注: 单一 component 时 unique=K=64, 但这是因为 single K layer, 跟 multi-layer 95% 阈值不可比)
  - **metadata kappa variance = 0.0** ❌ (目标 > 1e-4, 完全没有 per-item 区分)
  - **metadata scale variance = 0.0** ❌
  - **metadata conf variance = 0.0** ❌
  - **item↔SID alignment: 1-1 perfect** ✓ (每 item 唯一 SID 索引)
  - **baseline_collapse_confirmed = True** ❌
- **失败原因**:
  1. 当前 #97 / #98 公式 bug 未修复 (utils.py poincare_distance 没 proj_to_ball(diff) + Issue #47 Taylor 替换未应用)
  2. baseline 单一 component 距离 (M=1) 不能产生 per-item metadata 区分 — metadata kappa/scale 都是单值, 没 variance
  3. 跟 #96 task389 HypPreEncoder SID FAIL (unique=1) 一致 — collapse 在 baseline + 修复尝试中都存在
- **实施**: scripts/task392_issue99_formula_based_sid_diversity.py (R4 py_compile OK)
- **evidence**: products/task392_issue99_formula_based_sid_diversity/evidence_package.json
- **framework**: products/task392_issue99_formula_based_sid_diversity/gate2_acceptance_framework.md

### Gate 3 (= Stage 3 T5): ⏸ STOP per Issue #99 spec
- 原因: Gate 2 FAIL, Issue #99 spec "复跑 #93 同模型 on/off/shuffle 消融, 但只在 Gate 2 PASS 后执行"

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per Issue #99 spec
- 原因: Gate 3 STOP

---

## 2. Gate 2 acceptance framework (Issue #99 spec PASS criteria)

| 指标 | 阈值 | 当前 baseline | PASS? |
|------|------|---------------|-------|
| SID unique count | ≥ 9500/9922 | 64/9922 (0.65%) | ❌ |
| SID collision rate | ≤ 0.20 | 99.35% | ❌ |
| L0/L1/L2 utilization | ≥ 90% each | 100% (单层, 不可比) | ⚠️ N/A |
| L0/L1/L2 entropy | ≥ log(K_m)/2 | log(64)/2 = 3.0 | ❌ (entropy=0) |
| metadata kappa variance | > 1e-4 | 0.0 | ❌ |
| metadata scale variance | > 1e-4 | 0.0 | ❌ |
| metadata conf variance | > 1e-4 | 0.0 | ❌ |
| shuffle metadata distance | > 1e-4 | 0.0 (没 metadata) | ❌ |
| item↔SID 1-1 对齐 | True | True | ✓ |

→ **8/9 FAIL** — 当前 baseline 完全不满足 Issue #99 Gate 2 PASS criteria.

---

## 3. 依赖状态

- **Issue #97 [方向A Gate1]** (task390): ❌ FAIL (审计完成, 修复待启动) — 3 个公式 bug 找到 (proj_to_ball 缺失 + κ→0 发散 + c=1.0 硬编码)
- **Issue #98 [方向B Gate1]** (task391): ⚠️ PARTIAL PASS (init 通过, 训练坍缩未修复) — 3 个 component scale bug 找到
- **Issue #96** (task389): ❌ FAIL (HypPreEncoder SID unique=1, 比 baseline 64 更差, 验证 HypPreEncoder 方向错误)
- **Issue #93** (task386): ❌ FAIL (metadata uniform, shuffle_diff=0)

→ **本 Issue #99 Gate 2 当前无法 PASS**, 必须等 #97+#98 修复后, 跑新 Gate 1 → 新 Gate 2 → 才能 PASS.

---

## 4. Post-fix framework (供 #97/#98 修复后使用)

| Step | 操作 |
|------|------|
| 1 | Load FreeCurvHRQVAE with #97+#98 fix code applied |
| 2 | Run on 9922-item full dataset; collect SID per layer (L0/L1/L2) + per-item metadata |
| 3 | Compute: SID unique, collision, util, entropy, metadata variance (3 types) |
| 4 | Shuffle metadata across items; recompute on/off/shuffle T5 ablation |
| 5 | Verify item↔SID 1-1 alignment |
| 6 | Gate 2 PASS if ALL metrics ≥ threshold; else FAIL |

framework 已落盘: `products/task392_issue99_formula_based_sid_diversity/gate2_acceptance_framework.md`

---

## 5. 关键产物 (R21 强制具体 hash)

- **commit hash**: pending push (see gh issue comment)
- **verdict**: verdicts/task392_issue99_formula_based_sid_diversity_v2.md (本文件)
- **实施**: scripts/task392_issue99_formula_based_sid_diversity.py
- **evidence**: products/task392_issue99_formula_based_sid_diversity/evidence_package.json
- **framework**: products/task392_issue99_formula_based_sid_diversity/gate2_acceptance_framework.md

---

result: Issue #99 [方向C Gate2 SID/metadata 多样性准入] Gate 2 FAIL (依赖 #97/#98 修复, 当前 baseline 确认坍缩 8/9 指标 FAIL). framework + acceptance criteria 完整落盘, 等 #97/#98 修复后跑 post-fix framework → 写 Gate 2 verdict → commit+push+comment+close. 实施 commit + push + issue comment + close 进行中.