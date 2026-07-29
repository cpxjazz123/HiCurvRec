# Task #298 / Issue #26 Conflict Report verdict

**日期**: 2026-07-29
**状态**: 📋 **Issue #26 Conflict Report 透明记录 — 等候 owner 决定 (a/b/c)**
**R11.3 决策**: 不主动开启违反 NORTH STAR §1 的 housekeeping 单独 issue; 不替 owner 做架构层方向选择; 维持 cron tick 透明报告冲突

---

## 1. 冲突点整理 (R11.3 复述)

| 规则 | 来源 | 触发条件 | 行动 |
|------|------|---------|------|
| **Step 0.B** | loop.md §每轮执行步骤 | open 实验 issue 列表空 | 必须起草新 issue |
| **task296 §5** | owner R11.3 自主决策 | R10 backlog 真空 | housekeeping 推进, 不强启动 ROI 极低实验 |
| **task297 K3** | owner verdict | baseline recipe 内部 R@10 杠杆穷尽 | 后续必须在架构层 (Gumbel-Softmax / 多样 hash / per-item soft-assign) |
| **NORTH STAR §1** | loop.md 不可降级 | 每个 issue 都必须直接服务于 per-layer 可变曲率 | 不允许纯治理/纯审计/纯量化基建议题 |
| **NORTH STAR §4** | loop.md 不可降级 | 不可降级: 换 metric / 换码本空间 | 纯路线替换一律不发, 但 per-layer 异构机制豁免 |

**三者矛盾**:
- Step 0.B 要求强启动 issue
- task296 §5 要求 housekeeping
- task297 K3 暗示 architecture-level (新方向)

**NORTH STAR §1 锁死**: 任何 housekeeping 单独 issue 都违反 §1.

---

## 2. Issue #26 三选项分析 (R11.3 自主决策)

### 选项 1: 修订 loop.md, Step 0.B 在 R10 backlog 真空时改走 housekeeping
- **优点**: 解决 Step 0.B vs task296 §5 冲突, 不违反 NORTH STAR §1 (housekeeping 不发单独 issue)
- **缺点**: house-keeping 实际可在 task296 / task297 / verdicts 整理中完成, 不需要新 issue 触发
- **R11.3 决策**: ⚠️ **需 owner 决定**, AI 不可改 loop.md (R11.4 critical decision)

### 选项 2: owner override 启动架构层新方向 issue (per-item soft-assign / per-layer diverse hash)
- **优点**: 跟 task297 K3 owner 暗示一致, 不违反 NORTH STAR §4 (per-layer 异构机制豁免)
- **缺点**: 跟 task296 §5 R10 backlog 真空不主动启动冲突; 且 owner override 不可由 AI 替做
- **R11.3 决策**: ⚠️ **需 owner 决定**

### 选项 3: 暂停 cron tick
- **优点**: 完全透明停止 Step 0.B 强制起草, 避免重复 noise
- **缺点**: 完全停止任务推进, R10 backlog 真空 → 0 推进
- **R11.3 决策**: ⚠️ **需 owner 决定**

### R11.3 自主决策 (本任务行为)
- **不替 owner 选 1/2/3**
- **不启动违反 NORTH STAR §1 的 housekeeping 单独 issue**
- **不替 owner 启动架构层 issue**
- **维持 cron tick 透明报告**: 每个 loop tick 报告 backlog 真空, 等候 owner 决定

---

## 3. R10 backlog 真空实况 (2026-07-29)

| # | 任务 | 方向 | 状态 |
|---|------|------|------|
| 1 | task287 + #284 | κ-decouple K=128/256 | NO-GO (-16% / -17%) |
| 2 | task290 | FSQ + κ-decouple | NO-GO (-45.8%) |
| 3 | task291 | EMA + κ-decouple | NO-GO (-25.0%) |
| 4 | task292 | Restoration + κ-decouple | NO-GO (-21.7%) |
| 5 | task293 | per-layer per-epoch c_k curriculum | NO-GO Gate 0 (0/81 OPEN) |
| 6 | task294 | 跨任务 8 方向 13 verdict | NO-GO 综合 |
| 7 | task295 | R14 规则 promotion | ✅ done |
| 8 | task296 | paper.md §6.7.4 联动 | ✅ done |
| 9 | task297 / Issue #25 | Phase A + B 联合 | NO-GO (-16.5%) |
| 10 | task298 / Issue #26 | Conflict Report (本次) | 📋 等候 owner |

**9 个方向全部 NO-GO 收口**, baseline Stage 1 recipe 内部 R@10 杠杆穷尽 (task294 + task296 跨任务确认).

---

## 4. 架构层候选方向 (R11.3 整理, 备 owner 选项 2 用)

按 task297 K3 owner 暗示 + Issue #27 5 篇 partial-truth 候选:

1. **per-item soft-assign (Diverse Semantic IDs)**: 每层每个 item 分配到多个 codeword (probabilistic), 突破 K cap
2. **per-layer diverse hash**: 每层独立 hash function, codebook 容量 8x → 2048+ (vs 现状 256)
3. **Gumbel-Softmax 替代 argmax**: 训练时 soft-quantization, 推理时 hard-quantization, 缓解坍缩
4. **TIGER-style multi-codeword**: Rajput 2023 NeurIPS 实际用过 (Google Scholar `8228948348343790806`)
5. **Codebook Transforms**: 2026 Google Scholar `14663627898199463552`, Differentiable Vector Quantization

**R11.3 评估**:
- 选项 1+2+3 都属于 per-layer 异构机制 (NORTH STAR §4 豁免)
- 选项 4+5 是文献参考, 不可直接复用但提供方向

**但**: 这些都是新方向, 需要 owner override 才能启动, R11.3 不可替 owner 做决定.

---

## 5. 后续 cron tick 行为 (R10 + R11.3)

按 R10 "§16 空时主动推进" + R11 "禁止阻塞等待" + R11.4 "不可替 owner 做关键决策":

1. **每个 cron tick**: 检查 open issues → 0 (Issue #26 等候, 26 待 owner) → 透明报告 backlog 真空
2. **不启动任何 housekeeping 单独 issue** (NORTH STAR §1 锁死)
3. **不替 owner 启动架构层新方向** (R11.4 critical decision)
4. **Issue #26 维持 OPEN**: 等候 owner 决定
5. **paper.md / verdicts/ 整理**: 在 task296 / task297 已经在做, 不需要新 issue
6. **R10 真正可推进的低成本动作**:
   - 持续整理已写 verdict (合并到 TASKS_INDEX.md)
   - 持续 paper.md §6.7.4 联动段追加 (task296 模式)
   - 持续 R14 监控 + 透明报告 (Issue #25/26/27 模式)

---

## 6. 物理产物

- `verdicts/task298_issue26_conflict_report.md` (本 verdict, 透明记录)
- 不申请任何 GPU / Stage 1 / Stage 2 / Stage 3 / Stage 4 预算

---

## 7. 关键决策点 (R11.3)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | Issue #26 决策 | ⚠️ 等候 owner 决定 1/2/3 | 替 owner 选 | R11.4 critical decision 不可替做 |
| 2 | housekeeping 单独 issue | ❌ 不发 (违反 NORTH STAR §1) | 发 | 锁死 |
| 3 | 架构层新方向 | ❌ 不替 owner 启动 (R11.4) | 启动 | 锁死 |
| 4 | 后续 cron tick 行为 | ✅ 透明报告 + 不主动强启动 | 强启动 | R10 + R11 综合 |
| 5 | 本 verdict 写入 | ✅ 写 + commit | 不写 | 透明化 |

---

result: Task #298 / Issue #26 Conflict Report 透明记录. 9 方向 × 14 verdict 全 NO-GO 收口, R10 backlog 真空. Step 0.B 强启动 vs task296 §5 housekeeping vs task297 K3 architecture-level 三者矛盾, 需 owner 决定 (a 修订 loop.md / b 启动架构层 / c 暂停 cron tick). R11.3 自主决策: 不替 owner 选 1/2/3, 不发违反 NORTH STAR §1 的 housekeeping, 不替 owner 启动架构层新方向, 维持 cron tick 透明报告.
