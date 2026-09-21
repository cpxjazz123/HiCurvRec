# iter8 Direction Decision（Agent B，亲自产出）

- **数据集**：Amazon-2023 Instruments
- **迭代**：iter8（targeted bottleneck search，沿 BACKUP 路径）
- **上轮失败**：iter7 P3 Stage3-aware Distillation Term REFUSE-LAUNCH（来源：git history `097c573:stage2_RQ-VAE/curvature_RQ-VAE_iter7/logs/gate_decision_iter7.md`）
- **iter8 主线**：Agent G 在 `iteration_bridge.md` 给出 BACKUP P1 M2 Intrinsic Residual Reference Point 重构
- **审计基线 commit**：`097c573a11dea2613f69306efbb44ba2ab8ccab5`
- **审计日期**：2026-09-21

## 0. P1/P2/P3 三态硬资格预审

| 维度 | P1 (M2 codeword centroid ref) | P2 (batch-centroid 投影) | P3 (M2+M3 reference point 一致化) |
|---|---|---|---|
| 是否显式回答"如何修复 iter6 dominant bottleneck（Stage2→Stage3 几何传导路径失效）" | **是（强）** — M2 reference point 从 origin 改为 selected codeword Lorentz centroid，Stage2 残差几何 = Stage3 codeword 加权分布；与 v321 memory §"必须跳出的方向"完全契合 | **是（弱）** — batch-centroid 是 Stage 2 内部几何对象，不直接对齐 Stage 3 codeword 加权分布 | **是（强但风险高）** — M2+M3 reference point 一致化对齐跨曲率 transport 与 codeword 加权分布；但 M2+M3 双改与 v334 DDP 卡死同族 |
| 是否能让 test_R@10 > 0.065 突破 | 有机理路径（d=3）；+0.005~+0.010 上限 | 仅修复 Stage 2 内部几何，不修传导路径（d=5） | 有机理路径（d=2）；+0.005~+0.015 上限但风险高 |
| iter8 forbidden directions 命中 | 不命中（不写 ε-anneal、不加 attention、不 per-item 路由、不 cyclic_factor.detach()、不用 +0.25 bias、不 P2 manifold 替换、不跨 stage 蒸馏、不改 Stage3 trainer） | 不命中 | **命中（边界）** — M2+M3 双改与 v334 DDP 卡死历史同族；但 lit_search 已明确"不替换 transport 公式，仅 reference point" |
| Stage2 step ≤ 5000 是否可打印 DE-1/DE-2/DE-3 数值 | **可以** — DE-1 codebook centroid norm vs c(t)、DE-2 残差 vs codeword angle、DE-3 step5000 unique | **可以** — DE-1 batch-centroid norm vs c(t)、DE-2 残差 vs batch-centroid angle、DE-3 step5000 unique | 可以但需 M2+M3 同步 |
| (e) Gap-closing relevance（强制） | **PASS（强）** | **PASS（弱）** | **PASS（强但与 v334 历史冲突）** |

**预审结论**：
- P1 三态全部 PASS，进入正式评分
- P2 (e) PASS 但修复路径弱，列入 BACKUP
- P3 (e) PASS 但与 v334 DDP 卡死历史同族，列入 BACKUP（仅当 P1 失败时启用）

## 1. P1/P2/P3 评分表

| 维度 | P1 | P2 | P3 |
|---|---:|---:|---:|
| (a) 明确修复 iter6 dominant bottleneck 的路径 | **4** | **2** | **4** |
| (b) 2023+ 论文实证支持 | **3** | **3** | **3** |
| (c) 与 Poincare+Sinkhorn+M2/M3+cyclic c(t) 栈兼容性 | **4** | **5** | **2** |
| (d) Stage 1 端纯曲率变更 ceiling 风险（5=低） | **3** | **5** | **2** |
| (f) 创新与曲率相关性合规 | PASS | PASS | PASS |
| (e) Gap-closing relevance（强制） | PASS（强） | PASS（弱） | PASS（强） |
| **总评** | 14/20 | 15/20 | 11/20 |

P2 评分看似最高但 (a) 弱：batch-centroid 不直接对齐 Stage 3 codeword 加权分布，仅修 Stage 2 内部几何；P1 在 (a) 与 (e) 维度更强。

## 2. 唯一推荐：**P1 M2 Intrinsic Residual Reference Point 重构**

**为什么是 P1**：

1. **唯一直接对齐 Stage2↔Stage3 表征空间**：P1 把 M2 reference point 从 Poincaré origin 改为 selected codebook codeword 的 Lorentz centroid（按当前 batch assignment 加权），让 Stage2 残差向量直接对应 Stage3 codeword 加权分布，理论上能修复 Stage2→Stage3 几何传导路径失效。
2. **与 v321 memory 完全契合**：v321 论证 Stage 3 T5 SID 表征空间对 Stage 1 端纯几何变更强 lock；P1 不改 Stage 1 端几何变更，而是改 Stage 2 残差 reference point 让 Stage 2 geometric 输出与 codebook 加权分布几何对齐，绕开 v321 lock 论证的传导瓶颈。
3. **in-loop 无 Stage3 依赖**：P1 完全 in-loop（只改 `_step4_m2_residual`），不需要加载 Stage3 frozen model；MVG 4 层可立即验证；Stage2 训练 16 min 即可；总耗时约 1 小时（远低于 iter7 P3 跨 stage ~2 小时）。

## 3. 三行理由（P1 为唯一推荐）

1. P1 是 3 条候选中唯一直接对齐 Stage2↔Stage3 表征空间且显式回答"如何修复 Stage2→Stage3 几何传导路径失效"的机制；P2 仅修 Stage 2 内部几何，P3 与 v334 DDP 卡死历史同族。
2. P1 完全跳出 R36n 6 大方向（不写新 loss、不改 optimizer、不换 manifold）；命中 v321 memory §"必须跳出的方向"第 2 条"跨 stage 联合机制"子方向（虽然没有跨 stage 蒸馏，但是让 Stage 2 geometric 残差携带 Stage 3 codeword 几何信息）。
3. iter32 valid→test drift 风险可控（P1 实施复杂度低，不需要新 loss）；iter8 forbidden directions 全部不命中（P1 仅改 M2 reference point 选择，不修改 M2 transport 公式）。

## 4. BACKUP：**P2 batch-centroid Lorentz 切空间投影**

**保留 P2 作为 BACKUP 的原因**：
- 若 P1 codeword centroid 在 DDP 4 卡 + R51+ 6 确定性约束下出现 batch 内 centroid 计算不一致（不同 rank 的 codeword centroid 不一致），可立即切回 P2
- P2 是 in-loop 操作（不引入 Lorentz centroid 的 batch dependency），DDP 同步更简单
- P2 的 (a) 修复路径有部分道理（batch-centroid 在 Lorentz 切空间与 codebook 的 local frame 对齐）

**切回条件**：
- P1 implementation FAIL（grad_fn missing / MVG counterfactual 不通过）
- P1 codeword centroid 在 DDP 多卡下不一致导致 Stage 2 几何变化不一致
- Stage3 test_R@10 在 step5000 已显著低于 iter6 baseline 0.0534

## 5. iter8 forbidden directions 自检

| 禁止项 | P1 | P2 | P3 |
|---|---|---|---|
| 再做 Sinkhorn linear ε-anneal | 不命中 | 不命中 | 不命中 |
| 再加 attention | 不命中 | 不命中 | 不命中 |
| per-item 路由 π(c\|item) | 不命中 | 不命中 | 不命中 |
| cyclic_factor.detach() | 不命中 | 不命中 | 不命中 |
| +0.25 bias | 不命中 | 不命中 | 不命中 |
| P2 manifold 完整替换 | 不命中 | 不命中 | 不命中 |
| 跨 stage 蒸馏 | 不命中 | 不命中 | 不命中 |
| Stage3 trainer 修改 | 不命中 | 不命中 | 不命中 |

3 条候选全部 PASS forbidden 自检；最终推荐 P1，次选 P2，P3 因 v334 历史冲突不作为常规推荐（仅当 P1+P2 都 FAIL 时启用）。

## 6. 关键证据来源

1. iter7 审计 commit: `097c573a11dea2613f69306efbb44ba2ab8ccab5`（含 gate_decision_iter7.md）
2. iter8 candidate pool: `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter8/logs/lit_search_iter8.md`
3. iter8 dominant bottleneck + forbidden directions: `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter8/logs/iteration_bridge.md`
4. mechanism pool: `/home/wlia0047/.claude/skills/curvature-rqvae-iter/references/mechanism_pool.md`
5. baseline context: `/home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/MEMORY.md` — `v321-r37-fail-sid-locks-baseline.md`、`v318-r36h-ceiling-lock-56.md`、`iter32-r37-fail-valid-test-drift.md`

只读取，未修改 Python；未触发 web_search / WebFetch；未让 (e) 不通过的候选成为推荐。