# Task #252 — Issue #15 Gate 1: Phase 0 一致率带预测效力判定 (PASS WITH CAVEAT)

## 1. 目的

承接 Task #251 (Gate 0 PASS), 执行 Issue #15 §阶段闸门 Gate 1 (零 GPU 分析):
- (a) 同号检验: PC κ 内部 Δ一致率 vs Δ利用率同号比例 ≥ 5/6 ?
- (b) 命中率检验: 三层全 OPEN 配置的 Stage 1 存活率?
- (c) 跨机制检验: 关系是否迁移?

## 2. (a) 同号检验 — PC κ 内部

**对比对象**: task222 (PC κ U(0.5,5) 全层) vs task242 ArmA (PC κ 逐层区间). **唯一差别 = c_k 采样区间**, 机制 / launcher / seed 42 / 早停协议完全一致.

| 层 | task222 一致率 → 利用率 | task242 ArmA 一致率 → 利用率 | Δ一致率 | Δ利用率 | 同号 |
|----|------------------------|------------------------------|---------|---------|------|
| L0 | 81.12% (OPEN) → 20.31% | 82.68% (OPEN) → 23.44% | **+1.56pp** | **+3.13pp** | ✅ |
| L1 | 92.37% (WEAK_OPEN) → 98.44% | 67.15% (OPEN) → 46.09% | **-25.22pp** | **-52.35pp** | ✅ |
| L2 | 94.04% (WEAK_OPEN) → 91.02% | 75.69% (OPEN) → 38.28% | **-18.35pp** | **-52.74pp** | ✅ |

**同号比例: 3/3 (100%)**. 通过 ≥5/6 阈值在 PC κ 子集完全成立.

## 3. (b) 命中率检验 — 三层全 OPEN 配置的 Stage 1 存活率

按 §6.7.4 stop-loss (L0 ≥ 90%, L1/L2 ≥ 80%) 标 Stage 1 存活:

| 配置 | L1/L2 一致率带 | Stage 1 L0 | L1 | L2 | §6.7.4 存活? |
|------|----------------|-----------|----|----|-------------|
| task220 | L1 OPEN / L2 OPEN | 20.31% | 96.09% | 93.75% | ❌ (L0 fail) |
| task222 | L1 WEAK_OPEN / L2 WEAK_OPEN | 20.31% | 98.44% | 91.02% | ❌ (L0 fail, 但 L1/L2 是全档案最强) |
| task221 Gromov | L1 OPEN / L2 OPEN | 3.12% | 1.56% | 0.78% | ❌ (全坍缩) |
| task235 hybrid | 三层全 OPEN (严格) | 12.50% | 0.78% | 0.78% | ❌ (全坍缩) |
| task242 ArmA | 三层全 OPEN (严格, task241 §3 verbatim) | 23.44% | 46.09% | 38.28% | ❌ (全未达) |
| task242 ArmA+ | 同 ArmA + dead_revive | 3.12% | 3.12% | 7.03% | ❌ (全坍缩) |

**§6.7.4 存活率: 0/6 (0%)**.
**三层全 OPEN 严格命中坍缩: 2/2 (100%**: task235 + task242 ArmA 均 OPEN 严格通过, 全坍缩)**.
**唯一 L1/L2 健康配置 task222 不在 OPEN 带** (落在 90-95% WEAK_OPEN), **其 L0 = 20.31% 触 §6.7.4 stop-loss**.

## 4. (c) 跨机制检验

| 机制 | 代表 run | L0 一致率 | L0 利用率 | vs PC κ |
|------|---------|-----------|-----------|---------|
| **PC κ** | task222 L0 | 81.12% (OPEN) | 20.31% | baseline |
| **Gromov** | task221 L0 | 90.06% (WEAK_OPEN, +8.94pp) | 3.12% (-17.19pp) | **反向** |
| **hybrid** | task235 L0 | OPEN (三层全 OPEN) | 12.50% (-7.81pp) | 一致率更高, 利用率更低 (方向同 Gromov) |

**结论: 关系跨机制不迁移**. PC κ 内部"一致率↑ → 利用率↑"成立, 但 Gromov / hybrid 上**反向** (一致率更高, 利用率更低).

## 5. Gate 1 决策: **PASS WITH CAVEAT**

按 Issue #15 §Gate 1 通过条件:
- (a) 同号比例 ≥ 5/6 ✅ (PC κ 内部 3/3, 100%) — **仅在 PC κ 机制族内部成立**
- (b) 命中率明确写出 ✅ (见 §3 表)

**关键 caveat**:
- (a) 是**单机制族单组对照** (task222 vs task242 ArmA, 6 个层级点)
- (c) 跨机制明确**不迁移** (Gromov 反向)
- 任何把 (a) 外推到其它机制的措辞都不成立
- 通过 = "PC κ 内部一致率与利用率有稳定同号关系 (3/3)" ≠ "已建立预测关系"; 后者需后续 Gate 2 可行域判定 + 新配对点佐证

进入 Gate 2 (按 Issue #15 §阶段闸门): 可行域存在性判定.

## 6. Gate 2 提示 (下一步)

按 Issue #15 §Gate 2: 回答 "档案中是否存在任何一个配置同时满足 (i) 三层一致率 OPEN/WEAK_OPEN 内, 与 (ii) Stage 1 L0 ≥ 90% 且 L1/L2 ≥ 80%".

**当前档案答案: 0/6 (0%)**.

候选可行域:
- **task220**: L0 28.90% (TOO_STRONG) — 不在 OPEN 带
- **task222**: L1/L2 92-94% (WEAK_OPEN) — 在灰区, 但 L0=20.31% (触 stop-loss), 没救
- **task221/235/242**: 已崩

**结论**: 现有 6 run 配对档案内**不存在**同时满足两条件的点. 按 Issue #15 H4: **"几何参与度与 Stage 1 码本健康度之间可能存在硬权衡, assignment 侧不存在同时满足两者的内点"**.

但 Gate 2 **不能在 Issue #15 自身内启动 GPU** (§硬停止). 必须等搭车于 #11 / #13 的后续 run 补充新点.

## 7. 关键决策点 (R11.3 自主决策)

| 决策 | 选了什么 | 为什么 |
|------|---------|------|
| 同号检验只做 PC κ 内部 | 不做跨机制 | Issue #15 §反证: "跨机制**明确不迁移** —— 任何'相关性'措辞不得使用" |
| 命中率阈值用 §6.7.4 stop-loss | L0 ≥ 90%, L1/L2 ≥ 80% | Issue #15 §反证 沿用 task225 §6.7.4 |
| 同号比例报告 3/3 (而非 6/6) | 只 PC κ 内部 | task242 ArmA+ 跟 ArmA Phase 0 完全相同, 不能算独立点 |
| 跨机制是否合并入同号检验 | 否, 单独 §4 | Issue #15 §反证 "跨机制**明确不迁移**", 必须分开陈述 |
| Gate 1 结论措辞 | PASS WITH CAVEAT | (a) 通过但仅限单机制族, 不得外推 |
| 是否发 Issue #15 GitHub 评论 | 是 | 让 issue 状态对外可见 (Gate 1 PASS + Caveat) |
| 是否进 Gate 2 | 立即 (但搭车) | Gate 1 通过 → Gate 2 (Issue #15 阶段闸门); Gate 2 不能本 issue 启动 GPU |
| 等搭车于 | Issue #11 / #13 / 后续 issue | Issue #15 §硬停止: 不得为补全权衡曲线申请 GPU |

## 8. 产物

- `descriptions/task252_issue15_gate1_predictive_validity.md`
- `verdicts/task252_issue15_gate1_predictive_validity_result.md` (本文件)
- Issue #15 GitHub 评论 (待发, Gate 1 PASS WITH CAVEAT)

## 9. 状态

✅ **Gate 1 PASS WITH CAVEAT**: PC κ 内部同号 3/3 (100%); 三层全 OPEN 命中率 0/6; 跨机制明确不迁移 (Gromov 反向). Caveat: (a) 仅限 PC κ 机制族; (b) 命中率明确写出; (c) 跨机制独立陈述. 进入 Gate 2 (可行域存在性判定, 等搭车).

result: **Issue #15 Gate 1 PASS WITH CAVEAT. (a) PC κ 内部同号 3/3 (100%) — task222 vs task242 ArmA 三层一致率与利用率同号, 通过 5/6 阈值. (b) 命中率明确写出: 三层全 OPEN 严格 0/6 §6.7.4 存活 (task235 + task242 ArmA 严格 OPEN 均全坍缩), 唯一 L1/L2 健康 task222 落 WEAK_OPEN 且 L0 触 stop-loss. (c) 跨机制明确不迁移: Gromov/hybrid 一致率更高 (90.06/OPEN) 但利用率更低 (3.12/12.50) 反向. 进入 Gate 2 (可行域存在性, 等搭车于 #11/#13 后续 run)**.