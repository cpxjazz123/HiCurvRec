# Issue #145 [方向A Gate1] 硬SID保持下的κ可微几何校准辅助损失 — Gate 1 FAIL (架构修复成功 + 配方失败)

**Task**: #435 / Issue #145
**Commit**: `3a78b85`
**Verdict**: `verdicts/task435_issue145_gate1_fail_v3.md`
**产物**: `products/task435_issue145_kappa_calibration_aux_loss/{config,verdict,precheck,aux_graph_proof}.json`
**SHA256**: item_emb.parquet=`1a42341f01537d6db5f622e4831e0fc1a2039dc8b1288291dcc4963deee000cc`

---

## 4 Gate 详细内容回答 (R17+R20 强制)

### Gate 1 (= Stage 1 RQ-VAE / κ-可微几何校准辅助损失): ⚠️ **PARTIAL PASS + FAIL per spec**
- **状态**: ⚠️ PARTIAL PASS (架构修复成功) + FAIL (usage criterion)
- **关键数据 (Issue #145 spec 强制)**:
  - **Precheck PASS** ✅: aux_loss → c → κ grad = [2.04, 1.98, 1.92], hard SID → κ grad = [0, 0, 0] (isolated)
  - **架构修复实证成功**: 跟 #143 NO-GO 形成对照, Issue #145 验证了 **aux loss 确实能让 κ 拿到有限非零 grad**
  - Control 1000-step (no-aux, 跟 #143 同根因): κ grad = [0, 0, 0] 全部, util = 6.8%, max_load 失控 → 跟 #143 完全一致 (R137 κ lock 复现)
  - Calibration 1000-step (with aux):
    - step 200: loss=19.39, grad_κ=[1.97, 1.97, 1.96]
    - step 400: loss=18.30, grad_κ=[1.81, 1.80, 1.80]
    - step 600: loss=17.35, grad_κ=[1.65, 1.65, 1.65]
    - step 800: loss=16.52, grad_κ=[1.51, 1.51, 1.51]
    - step 1000: loss=15.78, grad_κ=[1.39, 1.39, 1.39]
  - **最终 κ**: [-1.347, -1.348, -1.349] (三层都从初始化 -0.793 软锁定移动到 -1.35, 持续优化轨迹)
  - **最终 grad**: [1.39, 1.39, 1.39] (三层都有限非零)
  - **最终 util**: L0=1.56%, L1=0.78%, L2=0.78% (期望 >= 90% ❌)
  - **最终 max_load**: L0=100%, L1=100%, L2=99.66% (期望 < 5% ❌)
- **失败原因**: aux_loss = α_pair · pairwise_calibration (mean of pairwise hyperbolic distance) + α_rank · ranking_calibration (relu(top-1 - top-2) margin minimization). **pairwise mean minimize 推动 codebook 聚集到 encoder 几何中心**, 所有 z_e 都跟最近 codebook 重合 → **codebook collapse** (util < 2%, max_load ≈ 100%). ranking_calibration 用 relu(margin) minimize top-1-top-2 margin 同样推动 codebook 收紧. 跟 Issue #9/Issue #11 mode collapse 同根因, 但触发器从 hard SID 换成 aux loss.
- **R18 4 维度路径对比 vs #143**:
  - D1 spec 摘录: Issue #145 是 loss design (aux loss 加 κ-dependent 项), #143 是参数所有权 (optimizer group) — **完全不同**
  - D2 实施核心: aux_loss 用 pairwise + ranking calibration, hard SID 隔离 — **完全不同**
  - D3 Gate 1 失败机制: 假设 aux loss 能让 κ 拿到 grad (验证 ✅), 但 aux loss minimize 推 codebook collapse — **完全不同失败点**
  - D4 引用文献: arXiv:2405.13979 曲率依赖的可学习几何路径 (同文献不同角度) — **不同引用角度**
- **verdict 路径**: `verdicts/task435_issue145_gate1_fail_v3.md` (本文件)
- **实施**: `scripts/task435_issue145_kappa_calibration_aux_loss.py`
- **整体决策**: ⚠️ PARTIAL — 架构修复成功 (κ grad path 修好, 跟 #143 反证对照) 但配方失败 (aux loss minimize 推 collapse). **NO-GO 收口**, 但 Issue #145 打开了 **架构级进展 + 配方级失败** 的诊断方向.

### Gate 2 (= Stage 2 Sinkhorn + dedup): ⏸ STOP per spec
- **原因**: Gate 1 PARTIAL/FAIL per spec, 无 SID 产出可推断 Sinkhorn (Issue #145 spec Gate 1 PASS 前禁止 Gate 2)
- **Issue spec 强制**: Gate 1 PASS 前禁止 Gate 2/3/4

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
- **原因**: Gate 2 STOP
- **Issue spec 强制**: Gate 3 训练 + 前置条件

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- **原因**: Gate 3 STOP
- **Issue spec 强制**: R@10 阈值 (vs HG-Rec baseline 0.1020) + 标记 [TARGET REACHED] 条件

---

## 关键产物

- **verdict**: `verdicts/task435_issue145_gate1_fail_v3.md` (本文件)
- **commit hash**: `3a78b85`
- **push**: origin/main (R15 强制)
- **实施**: `scripts/task435_issue145_kappa_calibration_aux_loss.py`
- **8 件套审计**: config.json + SHA256(item_emb=`1a42341f...`) + trace(control + aux) + precheck.json + aux_graph_proof.json + raw_log + verdict.json + commit
- **整体决策**: ⚠️ PARTIAL — 架构级 PASS (aux_loss 修好 κ grad path, 跟 #143 反证) + 配方级 FAIL (aux_loss minimize 推 codebook collapse). Issue #145 关闭, 不重启 aux_minimize_pairwise 路径, 但打开新方向: **margin-based diversity aux loss** (NOT minimize distance, 而是 maximize data-codebook 分布多样性)

---

## R18 严格路径对比 (vs #143)

| 维度 | Issue #143 (NO-GO) | Issue #145 (PARTIAL) |
|------|--------------------|----------------------|
| **D1 spec 摘录** | 修独立 κ 参数所有权 + optimizer group | 修 loss design (aux 加 κ-dependent 项) |
| **D2 实施** | 独立 κ + 3-group optimizer, **不修 loss** | aux_loss = pairwise + ranking calibration, 修 loss |
| **D3 失败机制** | loss 不通过 c → κ grad=0 | aux_loss 修好 c → κ grad 路径 (✅), 但 aux_loss minimize 推 collapse (❌) |
| **D4 文献** | arXiv:2405.13979 + fine-tunable hyperbolic scaling | arXiv:2405.13979 曲率依赖的可学习几何路径 (不同角度) |
| **R18 判定** | 必须新实验 (loss design 是真锁) | 必须新实验 (新 aux loss design) — 实证支持部分修复 |

---

## 后续修复路径 (R11.5 自主决策)

**核心发现**: 跟 Issue #143 形成 R18 反证链:
- #143 修所有权 → κ grad=0 (loss design 是真锁)
- #145 加 aux loss → κ grad 拿到 ✅ (loss design 修复成功) → 但 codebook collapse (配方级失败)

**真正可修复 aux loss**:
1. **Diversity-promoting aux** (NOT minimize, 而是 maximize margin/codebook 分布多样性)
2. **Codebook load-balancing aux** (跟 vqvae codebook loss 同款, 但加 κ-dependent 项)
3. **Ranking with codebook dispersion constraint** (margin 推到固定目标, NOT minimize)
4. **Gumbel-Softmax STE with κ grad** (跟 Issue #28 同框架, 但 spec 加 κ-dependent 项)

**Why NO-GO 收口 #145 当前方向**: aux_minimize_pairwise+ranking 跟 mode collapse 同根因. 任何"minimize 距离"型 aux 都会推 codebook collapse.

**Why 架构级 PARTIAL**: κ grad path 修好确认 — 任何方向后续可基于 #145 修复 (重新设计 aux loss, NOT pairwise+ranking minimize), 继续 R11.5 探索.

---

## 历史事故关联

| 事故 | 现象 | 根因 | 跟 #145 关系 |
|------|------|------|--------------|
| Issue #143 (Task #433) | 修所有权 κ 仍 grad=0 | loss design 锁 | #145 反证: 加 aux → κ 拿到 grad ✅ |
| Issue #144 (Task #434) | mixing logits grad=0 | loss 不依赖 α | 跟 #145 共享 "loss design 缺 κ/α 项" 根因 |
| Issue #28 (Task #299) | Gumbel-Softmax mode collapse | τ_l + c_k range 推 boundary | #145 aux minimize 同根因 |
| Issue #11 (Task #242) | per-layer c_k full NO-GO | c_k range 路径耗尽 | #145 跨修复 (loss design) |

---

## 收口

- Issue #145 关闭 (`gh issue close 145 --reason completed`)
- 10 方向 NO-GO 收口 + Issue #145 架构级 PARTIAL = **baseline recipe 路径 + 修复尝试 联合耗尽**
- **新方向候选** (R11.5 决策, 等 owner 拍板): diversity-promoting aux loss / Gumbel-Softmax STE with κ grad
- **R@10 ceiling 0.1053 + R137 κ lock + R139 reproducibility triangle = baseline recipe 路径耗尽确认**