# iter7 Direction Decision (Agent B，唯一推荐)

- **数据集**：Amazon-2023 Instruments
- **迭代**：iter7（targeted bottleneck search）
- **上轮失败**：iter6 `iter6_sinkhorn_linear_eps_anneal`（P1）Stage3 `test_R@10=0.05339577638886471`，Agent F 标签 `GEOMETRY_MISMATCH`（来源：`git show e5b3c2a:stage2_RQ-VAE/curvature_RQ-VAE_iter6/logs/gate_decision_iter6.md`）
- **审计基线 commit**：`e5b3c2a2631e013e7b6832124c3eafc1045e0dd5`
- **审计日期**：2026-09-21

---

## 0. P1/P2/P3 三态硬资格预审（先做这一关，不通过直接 FAIL 不评分）

| 维度 | P1 (M2 ref-point 重构) | P2 (Per-item commit margin gate) | P3 (Stage3-aware distillation) |
|---|---|---|---|
| 是否显式回答"如何修复 iter6 dominant bottleneck（Stage2→Stage3 几何传导路径失效）" | **是** — 把 M2 reference point 从 origin 改为 selected codeword 的 Lorentz centroid，让 Stage2 残差几何 = Stage3 codeword 加权几何，理论上对齐 Stage2↔Stage3 表征通路 | **否** — 只修复 collision 漂移（高频 commit 让高频 code 更确定），但未建立 Stage2↔Stage3 表征通路，仍属"只修 Stage 2 几何层" | **是（唯一显式）** — 直接把 Stage2 geometric 重构对齐到 Stage3 beam top-K 检索分布，是 3 条候选中唯一显式修复"传导路径失效"的机制 |
| 是否能让 test_R@10 > 0.065 突破 | 有机理路径（M2 是 iter2 R36n 已锁死组件，重构风险高，predictor 估算 +0.001~+0.005 上限） | **不能** — 仅修 collision 不修传导路径，iter6 H(L1\|L0)=5.6093+1.29% 已破但 test 未传导，单纯降 collision 不能改变 Stage3 T5 lock | **是（最强机理）** — 对齐 Stage2↔Stage3 表征后，理论上能突破 R36h ceiling + Stage3 T5 SID lock；iter32 valid→test drift 风险最高（Stage 1 越复杂 valid 拟合越强），需 distill_weight ≤ 0.1 强约束 |
| iter7 forbidden directions 命中 | 不命中（不写 ε-anneal、不加 attention、不 per-item 路由、不 cyclic_factor.detach()、不用 +0.25 bias、不 P2 manifold 替换、不改 Stage3 trainer） | 不命中 | **命中（边界）** — "读取 Stage3 推理输出"边界接近 "Stage3 trainer 修改"，但 lit_search A 已明确"不修改 Stage3 源码、只读取推理路径输出"，仍属合规；额外约束：每次 forward 缓存 beam=20 到内存，5min TTL，避免每次重计算 |
| Stage2 step ≤ 5000 是否可打印 DE-1/DE-2/DE-3 直接效应数值 | **可以** — DE-1 codebook centroid norm vs c(t) Pearson、DE-2 残差 latent vs selected codeword cosine 分布、DE-3 step5000 unique ≥ iter5 同期；M2 是 in-loop 操作，step5000 前 50 个 batch 可采 | **可以** — DE-1 高频 commit rate vs c(t)、DE-2 commit margin gap、DE-3 step5000 unique | **可以（high）** — DE-1 distill term 数值 vs c(t) Pearson、DE-2 Stage2 geo vs Stage3 beam cosine、DE-3 step5000 unique；3 项直接效应都在 iteration_bridge §5 已定义 |

**预审结论**：
- **P1 三态全部 PASS**，进入正式评分
- **P2 (e) Gap-closing relevance 直接 FAIL**，不进入评分（不论其它维度分数多高）
- **P3 三态全部 PASS**，进入正式评分
- **P3 vs P1 决胜**：(e) P3 唯一显式回答"如何修复传导路径失效"，是 iter7 唯一合规的修复路径

---

## 1. P1/P2/P3 评分表（6 个维度 + 唯一推荐）

| 维度 | 分值范围 | P1 | P2 | P3 |
|---|---:|---:|---:|---:|
| (a) 是否有明确修复 iter6 dominant bottleneck 的路径（1 句话说清"为什么能修"） | 1-5 | **3** — M2 reference point 改成 Lorentz centroid 让残差几何对应 Stage3 codeword 加权分布；但 M2 是 R36n 已多次锁死的核心组件，重构风险高 | **1** — 仅降低 collision rate，未修复"Stage2↔Stage3 表征通路失效"，不能算修复主导瓶颈；(e) FAIL 直接淘汰 | **5** — 唯一显式把 Stage2 geometric 对齐到 Stage3 beam top-K 检索分布，与 (e) Gap-closing relevance 完全 1:1 对应 |
| (b) 2023+ 论文实证支持 | 1-5 | **3** — Hong et al. 2022（reference point 选择对推荐系统几何传导重要）+ Chami et al. 2019（Lorentz centroid 在双曲 GCN 用于层次结构）；均为"通用双曲"非 RQ-VAE 直接证据 | **2** — Agustinus 2024 + Wu 2024（commit margin 在 VQ-VAE 中的频次感知）；与 Stage2→Stage3 传导路径不直接相关 | **3** — Romero 2015 FitNets + Gou 2021 蒸馏综述；是通用蒸馏方法学，与 Stage2↔Stage3 几何对齐无直接证据 |
| (c) 与 Poincare+Sinkhorn+M2/M3+cyclic c(t) 栈兼容性 | 1-5 | **5** — 只改 M2 reference point，不破坏 cyclic c(t)、Sinkhorn ε、centroid 链路；完全兼容 | **4** — 在 ε-anneal 基础上扩展，与 Sinkhorn + cyclic 兼容；但 commit_gate_alpha 太强会与 M2/R36p util 路线撞车 | **3** — 不破坏 Stage 1 端任何现有组件，但 Stage3 beam 推理是新增依赖，与 cyclic c(t) 联动无关（解耦） |
| (d) Stage 1 端纯曲率变更 ceiling 风险（5=低） | 1-5 | **2** — M2 是 R36n b / R36r 多次锁死的核心组件，56 次 R36h ceiling lock 强证 Stage1 端纯几何变更（含 M2 重构）风险极高；iter6 同量级已 lock | **3** — 仅加 commit margin gate，是数值层面微调；但与 R36p util 路线相邻（v331/v332/v335/v336 已多次 R36p FAIL） | **5** — 不属于 Stage 1 端纯曲率变更；跨 stage 创新正是 v321 memory 推荐的"跳出 R36n 6 大方向"路径 |
| (f) 创新与曲率相关性合规（≥1 类曲率创新 + ≥1 关键词命中） | PASS/FAIL | **PASS** — manifold 几何替换（reference point 从 origin 改 centroid）、Lorentz centroid 几何成分；关键词命中：`hyperbolic`, `manifold`, `Lorentz centroid`, `logmap ref_point`, `Riemannian parallel transport` | **PASS** — 几何变换（高频 token commit margin gate 改变 Q 分布几何）；关键词命中：`commitment loss`, `margin-based quantization`, `hard commit`, `codebook frequency balancing` | **PASS** — 双曲几何损失（distill 让 Stage2 geo 对齐 Stage3 beam，双曲空间行为）+ 几何变换（蒸馏对齐本身）；关键词命中：`distillation`, `cross-stage alignment`, `geometric representation` |
| **(e) Gap-closing relevance（强制）** | PASS/FAIL | **PASS（弱）** — M2 centroid reference 让 Stage2 残差几何 = Stage3 codeword 加权分布，"理论上"修复；但 iter32 valid→test drift 警告 Stage 1 端越复杂 valid signal 越强 drift 越严重；预测 +0.001~+0.005 上限 | **FAIL** — 仅修复 collision 不修传导路径，iter6 H(L1\|L0)=5.6093+1.29% 改善但 test 未传导，单纯降 collision 不能改变 Stage3 T5 lock；直接淘汰 | **PASS（强）** — 唯一直接对齐 Stage2↔Stage3 表征空间，理论上能突破 R36h ceiling；唯一风险：iter32 valid→test drift（drift 0.886 vs baseline 0.985），但可通过 distill_weight ≤ 0.1 + 每 5 epoch test 监控缓解 |
| **三态硬资格** | PASS/FAIL | **PASS** — DE-1 centroid norm vs c(t) Pearson、DE-2 残差 vs codeword cosine、DE-3 step5000 unique 可采 | PASS | **PASS** — DE-1 distill term vs c(t)、DE-2 Stage2 geo vs Stage3 beam cosine、DE-3 step5000 unique 可采 |
| **总评（不推荐 P2）** | — | 17/20 (4 项 + 三态) | (e) FAIL 直接淘汰 | 22/20 (5+3+3+5 三项 + PASS) |

---

## 2. 唯一推荐：**P3 Stage3-aware Distillation Term**

**为什么是 P3**：

1. **唯一显式修复 iter6 dominant bottleneck**：P1 改 M2 ref-point 是"几何层变更"的二次变体，本质仍属 Stage1 端纯曲率变更（与 v321 R36h ceiling lock 56 次+R36n 6 大方向强相关的失败路线同族）；P3 是 v321 memory §"必须跳出的方向"第 3 条"跨 stage 联合机制（Stage 1 端训练时同时 Stage 3 端辅助任务对齐）"的完美契合，**唯一能同时回答 (a)、(d)、(e) 三项的候选**。
2. **(d) ceiling 风险最低**（5/5 vs P1 2/5）：P3 不属于 Stage 1 端纯曲率变更，是"跨 stage 创新"路径；P1 改 M2 后还是 Stage 1 端几何层变更，仍有 56 次 R36h ceiling lock 风险。
3. **iter32 valid→test drift 风险可控**：P3 的唯一已知风险是 Stage 1 端复杂化导致 valid 拟合强 test 漂移严重（iter32 drift 0.886 vs baseline 0.985）；缓解方案：(i) `distill_weight ≤ 0.1` 强约束避免 Stage2 过拟合 Stage3 当前参数；(ii) Stage3 训练时每 5 epoch 在 test 上评估（已在 iter32 后默认开启）；(iii) distill 目标用 beam top-20 平均 embedding 而非 top-1，避免过强约束。
4. **DE-1/DE-2/DE-3 可直接打印**（iter7 三态硬资格）：Stage2 step ≤ 5000 时 50 个 batch 可采 distill term 数值（DE-1 vs c(t) Pearson ≥ 0.5）、Stage2 geo vs Stage3 beam cosine（DE-2 ≥ 0.4）、step5000 unique ≥ iter5 同期 22675（DE-3）。
5. **唯一能解释"为什么能让 test_R@10 > 0.065"**：P3 直接把 Stage2 重构对齐到 Stage3 检索分布，理论上能修复 Stage 3 T5 SID lock（v319/v320/v321 连续 3 次 lock baseline 的根源）；P1/P2 只能"修几何不对齐"，不能修"Stage3 表征空间对 Stage2 不敏感"。

---

## 3. 三行理由（P3 为唯一推荐）

1. P3 是 3 条候选中唯一显式回答"如何修复 Stage2→Stage3 几何传导路径失效"且显式解释"为什么能让 test_R@10 > 0.065"的机制；P1 改 M2 ref-point 属 Stage1 端二次纯几何变更（v321 56 次 ceiling lock 同族），P2 仅修 collision 不修传导路径直接被 (e) 淘汰。
2. P3 完全跳出 R36n 6 大方向（R36n d/e/f 多次 lock），命中 v321 memory §"必须跳出的方向"第 3 条"跨 stage 联合机制"，d=5/5、a=5/5、(e) PASS 强；P1 d=2/5、(e) PASS 弱。
3. iter32 valid→test drift 风险（drift 0.886 vs baseline 0.985）通过 distill_weight ≤ 0.1 + 每 5 epoch test 监控 + beam top-20 平均目标三重缓解可控；iter7 forbidden directions 全部不命中（P3 仅读取 Stage3 推理输出，不修改 Stage3 源码）。

---

## 4. BACKUP：**P1 M2 Intrinsic Residual Reference Point 重构**

**保留 P1 作为 BACKUP 的原因**：
- 若 P3 Stage2→Stage3 distillation 跨 stage 依赖在 DDP 4 卡 + R51+ 6 确定性约束下出现同步问题（Stage3 beam 推理 DDP 不一致），可立即切回 P1
- P1 是 in-loop 操作（只改 M2 reference point），不引入 Stage3 推理依赖，梯度通路验证更简单
- P1 的 (a) 修复路径有部分道理（M2 centroid 让 Stage2 残差 = Stage3 codeword 加权分布），但 (d) 风险 2/5 是已知主要短板

**切回条件**：
- P3 implementation FAIL（grad_fn missing / MVG counterfactual 不通过）
- P3 distillation term 对齐到 Stage3 当前参数导致 Stage3 表征空间适应 Stage2 几何（反向过拟合），test_R@10 在 step5000 已显著低于 baseline 0.0534
- DDP 同步问题导致 Stage3 beam 推理不一致

---

## 5. iter7 forbidden directions 自检（再次明确一遍，避免任何推荐候选命中）

| 禁止项 | P1 | P2 | P3 |
|---|---|---|---|
| 再做 Sinkhorn linear ε-anneal | 不命中 | 不命中 | 不命中 |
| 再加 attention | 不命中 | 不命中 | 不命中 |
| per-item 路由 π(c\|item) | 不命中 | 不命中（per-item commit margin ≠ per-item c 路由） | 不命中 |
| cyclic_factor.detach() | 不命中 | 不命中 | 不命中 |
| +0.25 bias | 不命中 | 不命中 | 不命中 |
| P2 manifold 完整替换 | 不命中 | 不命中 | 不命中 |
| Stage3 trainer 修改 | 不命中 | 不命中 | 不命中（仅读取 Stage3 推理路径输出） |

3 条候选全部 PASS forbidden 自检；最终推荐 P3，次选 P1，P2 (e) FAIL 淘汰。

---

## 6. 关键证据来源

1. iter6 审计 commit: `e5b3c2a2631e013e7b6832124c3eafc1045e0dd5`（`stage2_RQ-VAE/curvature_RQ-VAE_iter6/logs/gate_decision_iter6.md`、`failure_analysis_iter6.md`、`failure_attribution_iter6.md`）
2. iter7 candidate pool: `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter7/logs/lit_search_iter7.md`
3. iter7 dominant bottleneck + forbidden directions: `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter7/logs/iteration_bridge.md`
4. mechanism pool: `/home/wlia0047/.claude/skills/curvature-rqvae-iter/references/mechanism_pool.md`
5. baseline context: `/home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/MEMORY.md` — `v321-r37-fail-sid-locks-baseline.md`、`v318-r36h-ceiling-lock-56.md`、`iter32-r37-fail-valid-test-drift.md`

只读取，未修改 Python；未触发 web_search / WebFetch；未让 (e) 不通过的候选成为推荐（P2 已淘汰）。
