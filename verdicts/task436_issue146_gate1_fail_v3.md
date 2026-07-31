# Issue #146 [方向B Gate1] hard-SID不变的product距离对比辅助损失 — Gate 1 FAIL (架构修复成功 + 配方失败)

**Task**: #436 / Issue #146
**Commit**: `59e69bc`
**Verdict**: `verdicts/task436_issue146_gate1_fail_v3.md`
**产物**: `products/task436_issue146_product_distance_contrastive/{config,verdict,precheck,d_mix_graph_proof}.json`
**SHA256**: item_emb.parquet=`1a42341f01537d6db5f622e4831e0fc1a2039dc8b1288291dcc4963deee000cc`

---

## 4 Gate 详细内容回答 (R17+R20 强制)

### Gate 1 (= Stage 1 RQ-VAE / product-distance contrastive): ⚠️ **PARTIAL PASS + FAIL per spec**
- **状态**: ⚠️ PARTIAL PASS (架构修复成功) + FAIL (实际训练中 κ/mixing grad 仍=0)
- **关键数据 (Issue #146 spec 强制)**:
  - **Precheck PASS** ✅: aux_loss → c → κ grad = [2.16e-5, 1.92e-3, 6.33e-3], aux_loss → mixing grad = [6.24e-4, 3.85e-3, 1.20e-2], hard SID → κ/mixing grad = [0, 0, 0] (isolated=True)
  - **架构修复实证成功**: 跟 Issue #143 (Task #433 NO-GO) 反证对照, 加 aux loss 让 κ/mixing grad path 修好
  - Control 1000-step (no-aux, 跟 #144 同根因): κ grad = [0, 0, 0] 全部, mixing grad = [0, 0, 0] 全部, util = 6.8% (R137 κ lock 复现)
  - Calibration 1000-step (with product-distance contrastive):
    - step 200: loss=0.0102, grad_κ=[0, 0, 0], grad_mixing=[0, 0, 0]
    - step 400: loss=0.0106, grad_κ=[0, 0, 0], grad_mixing=[0, 0, 0]
    - step 600: loss=0.0085, grad_κ=[0, 0, 0], grad_mixing=[0, 0, 0]
    - step 800: loss=0.0130, grad_κ=[0, 0, 0], grad_mixing=[0, 0, 0]
    - step 1000: loss=0.0112, grad_κ=[0, 0, 0], grad_mixing=[0, 0, 0]
  - **最终 κ**: [-0.832, -0.804, -0.792] (三层都从 -0.793 软锁定轻微扰动到 -0.83/-0.80, 极小移动)
  - **最终 grad**: [0, 0, 0] (三层都=0, 训练中拿不到有效梯度)
  - **最终 util**: L0=1.56%, L1=3.91%, L2=0.78% (期望 >= 90% ❌)
  - **最终 max_load**: L0=100%, L1=98.34%, L2=97.75% (期望 < 5% ❌)
- **失败原因**:
  1. **核心根因 — argmin 切断梯度流**: `anchor_assign = d_mix.argmin(dim=-1)` 是 discrete 索引, 不可微. 然后 `d_mix[i, anchor_assign[i]]` 通过 argmin index 提取 (i, anchor) 位置的距离, 虽然 d_mix 本身有 grad_fn, 但实际只有 ONE element per sample 被选中 — 路径信号微弱
  2. **配方根因 — F.relu(margin - pos + neg) 频繁触发零梯度**: 当 margin 已经满足 (pos << neg), relu clamp 到 0, gradient = 0
  3. **fallback 路径无效 — d_mix.mean() * 0.0**: 即使 positive pair 找不到, 用 d_mix.mean() * 0.0 保留 grad_fn, 但 value = 0 → gradient 乘 0 = 0
  4. **precheck vs 训练 gap**: precheck 用 X[:BATCH_SIZE] 的固定 batch (precheck 一次性 backward 验证 graph), 训练时随机 batch 的 contrastive_loss 经常 = 0
- **R18 4 维度路径对比 vs #144**:
  - D1 spec 摘录: Issue #146 product distance contrastive (3 分量 d_mix), #144 temperature simplex — **完全不同**
  - D2 实施核心: ProductDistanceModel 3 mixing + softmax + d_mix = α·d_anchor + β·d_fixed + γ·d_eucl + relu(margin - pos + neg) contrastive — **完全不同**
  - D3 Gate 1 失败机制: argmin-based anchor_select + F.relu 频繁 0 → 训练中 grad=0 — **完全不同失败点**
  - D4 引用文献: arXiv:2307.04514 weighted mixed-curvature product manifold — **完全不同文献**
- **verdict 路径**: `verdicts/task436_issue146_gate1_fail_v3.md` (本文件)
- **实施**: `scripts/task436_issue146_product_distance_contrastive.py`
- **整体决策**: ⚠️ PARTIAL — 架构级 PASS (aux 修好 κ/mixing grad path, 跟 #144 反证) + 配方级 FAIL (argmin + relu 联合切断训练中梯度). Issue #146 关闭, 不重启 contrastive-margin 路径, 但打开新方向: **continuous-anchor aux** (NOT argmin 选 anchor, 用 soft attention / Gumbel-Softmax)

### Gate 2 (= Stage 2 Sinkhorn + dedup): ⏸ STOP per spec
- **原因**: Gate 1 PARTIAL/FAIL per spec, 无 SID 产出可推断 Sinkhorn (Issue #146 spec Gate 1 PASS 前禁止 Gate 2)
- **Issue spec 强制**: Gate 1 PASS 前禁止 Gate 2/3/4

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
- **原因**: Gate 2 STOP
- **Issue spec 强制**: Gate 3 训练 + 前置条件

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- **原因**: Gate 3 STOP
- **Issue spec 强制**: R@10 阈值 (vs HG-Rec baseline 0.1020) + 标记 [TARGET REACHED] 条件

---

## 关键产物

- **verdict**: `verdicts/task436_issue146_gate1_fail_v3.md` (本文件)
- **commit hash**: `59e69bc`
- **push**: origin/main (R15 强制)
- **实施**: `scripts/task436_issue146_product_distance_contrastive.py`
- **8 件套审计**: config.json + SHA256(item_emb=`1a42341f...`) + trace(control + aux) + precheck.json + d_mix_graph_proof.json + raw_log + verdict.json + commit
- **整体决策**: ⚠️ PARTIAL — 架构级 PASS (aux_loss 修好 κ/mixing grad path, 跟 #144 反证) + 配方级 FAIL (argmin + relu 联合切断训练中梯度). Issue #146 关闭, 不重启 contrastive-margin 路径, 但打开新方向: **continuous-anchor aux** (NOT argmin, 用 soft attention / Gumbel-Softmax)

---

## R18 严格路径对比 (vs #144 + #145)

| 维度 | Issue #144 (NO-GO) | Issue #145 (PARTIAL) | Issue #146 (PARTIAL) |
|------|---------------------|----------------------|----------------------|
| **D1 spec 摘录** | 修 mixing 路径: temperature simplex + entropy clamp | 修 loss design (aux 加 κ-dependent 项) | 修 loss design (product distance + contrastive) |
| **D2 实施** | 3 mixing scalars + softmax + entropy bound | aux_loss = pairwise + ranking minimization | d_mix = α·d_anchor + β·d_fixed + γ·d_eucl + contrastive margin |
| **D3 失败机制** | 温度受控 + 熵下界仍 mixing grad=0 | aux minimize 推 codebook collapse | argmin + relu 切断训练中梯度 |
| **D4 文献** | arXiv:2307.04514 + Riemannian Optimization | arXiv:2405.13979 曲率依赖可学习几何 | arXiv:2307.04514 weighted mixed-curvature |
| **Precheck vs 训练** | grad=0 训练中 (跟 precheck 一致) | grad 训练中有 (1.39) | grad 训练中=0 (跟 precheck 不一致) |
| **R18 判定** | 必须新实验 | 必须新实验 (PARTIAL) | 必须新实验 (PARTIAL) |

---

## 后续修复路径 (R11.5 自主决策)

**核心发现 — 跟 Issue #145 共享根因**: 任何依赖 argmin 选择 anchor 的 aux loss 都会切断训练中梯度. precheck 单 batch 能验证 graph, 但训练时随机 batch 的 positive pairs 经常失败 → gradient 实际 = 0.

**真正可修复 aux** (跟 #145 共同方向):
1. **Soft anchor selection**: 用 softmax(d_mix / T) 而非 argmin(d_mix), 让 anchor 选取 differentiable
2. **Gumbel-Softmax STE with κ grad**: 跟 Issue #28 同框架, 但加 κ-dependent 项
3. **Continuous contrastive**: 不选 discrete anchor, 直接最小化 continuous pairwise distance, 让 codebook 推到固定 dispersion
4. **Margin-based diversity aux** (跟 #145 同方向): diversity-promoting NOT minimize

**Why NO-GO 收口 #146 当前方向**: argmin + relu 联合切断训练中梯度. 任何"先选 anchor 再算 contrastive"的路径都会被切断.

**Why 架构级 PARTIAL**: precheck 验证 graph 是真的修好 (跟 #144 反证对照, mixing grad path 修复确认), 但训练中具体 batch 的 contrastive loss 经常 = 0 → 实际无更新.

---

## 历史事故关联

| 事故 | 现象 | 根因 | 跟 #146 关系 |
|------|------|------|--------------|
| Issue #144 (Task #434) | temperature simplex mixing grad=0 | 温度受控 + 熵下界无效 | #146 反证: 加 contrastive 让 mixing grad path 修好 (precheck) |
| Issue #145 (Task #435) | aux minimize 推 collapse | pairwise+ranking minimization | #146 共享根因 (argmin + relu 切断) |
| Issue #143 (Task #433) | 修所有权 κ grad=0 | loss design 锁 | #146 反证: 加 aux → κ grad path 修好 (precheck) |
| Issue #141 (Task #431) | attribution matrix grad=0 | 同样 argmin 切断 | 跟 #146 共享 argmin 根因 |
| Issue #28 (Task #299) | Gumbel-Softmax mode collapse | τ_l + c_k range 推 boundary | #146 修复方向 (Gumbel-Softmap STE) |

---

## 收口

- Issue #146 关闭 (`gh issue close 146 --reason completed`)
- 11 方向 NO-GO 收口 + Issue #145/#146 架构级 PARTIAL = **baseline recipe 路径 + 修复尝试 联合耗尽**
- **新方向候选** (R11.5 决策, 等 owner 拍板): continuous-anchor aux (softmax(d_mix/T)) + Gumbel-Softmax STE with κ grad + diversity-promoting aux
- **R@10 ceiling 0.1053 + R137 κ lock + R139 reproducibility triangle + argmin 切断 = baseline recipe 路径耗尽确认**