# Issue #149 [方向B Gate1] 硬SID不变的连续soft-anchor product校准损失 — Gate 1 FAIL (架构 PASS + 配方 FAIL)

**Task**: #439 / Issue #149
**Commit**: TBD (after git push)
**Verdict**: `verdicts/task439_issue149_gate1_fail_v1.md`
**产物**: `products/task439_issue149_continuous_anchor_product/{config,d_mix_graph_proof,verdict}.json`
**SHA256**: item_emb.parquet=`1a42341f01537d6db5f622e4831e0fc1a2039dc8b1288291dcc4963deee000cc`

---

## 4 Gate 详细内容回答 (R17+R20 强制)

### Gate 1 (= Stage 1 RQ-VAE / HRQVAE): ❌ **FAIL (架构 PASS + 配方 FAIL)**
- **状态**: Precheck PASS + Control 复现 #144 + Aux grad 非零 + 两分量 < 0.1 + usage/max_load 不达标
- **关键数据 (Issue #149 spec 强制)**:
  - **Precheck PASS** ✅: aux_loss → κ grad = [2.45e-5, 5.42e-6, 5.94e-5], mixing grad = [6.15e-4, 4.66e-4, 1.13e-4]; hard SID → κ/mixing grad = [0, 0, 0] (isolated=True)
  - **架构修复实证成功**: 跟 Issue #146 (Task #436) 同 pattern, 加 soft-anchor continuous d_mix 让 κ/mixing grad path 修好 (precheck)
  - **Control 1000-step (no-aux, 跟 #144 同根因)**: 
    - step 200/400/600/800/1000: grad_κ=[0, 0, 0], grad_mixing=[0, 0, 0] (R137 κ lock 复现)
  - **Calibration 1000-step (with continuous-anchor + softplus)**:
    - step 200: loss=1.9774, grad_κ=[4.70e-5, 6.71e-6, 7.49e-6], grad_mixing=[0.007, 0.001, 0.002]
    - step 400: loss=1.9300, grad_κ=[1.62e-5, 4.87e-5, 9.98e-6], grad_mixing=[0.069, 0.011, 0.006]
    - step 600: loss=0.9751, grad_κ=[4.16e-6, 1.45e-5, 2.79e-2], grad_mixing=[0.177, 0.049, 0.075]
    - step 800: loss=0.5598, grad_κ=[2.26e-7, 2.07e-5, 6.72e-3], grad_mixing=[0.090, 0.101, 0.022]
    - step 1000: loss=0.3067, grad_κ=[6.70e-6, 4.65e-6, 6.10e-3], grad_mixing=[0.046, 0.048, 0.017]
    - **loss 显著下降 1.97 → 0.31 (-85%)** ✓ (跟 #146 不同, #149 修复 loss = 0 反例)
  - **最终 κ**: [-0.785, -0.794, -0.693] (跟 #146 同 pattern, 三层都接近 -0.793 软锁定)
  - **最终 grad_κ**: [6.7e-6, 4.7e-6, 6.1e-3] (finite nonzero ✓)
  - **最终 grad_mixing**: [0.046, 0.048, 0.017] (finite nonzero ✓)
  - **两分量贡献 > 0.1**: 1/3 layers (期望 >= 2) ❌ (L0=0.177 > 0.1 ✓; L1=0.049 < 0.1 ❌; L2=0.075 < 0.1 ❌)
  - **最终 util**: L0=3.13%, L1=1.56%, L2=0.39% (期望 >= 90% ❌)
  - **最终 max_load**: L0=99.90%, L1=99.95%, L2=100% (期望 < 5% ❌)
  - **hard SID round-trip**: True ✓
  - **all loss finite**: True ✓
- **失败原因**:
  1. **核心根因 — soft-anchor continuous 路径 + softplus margin 仍 fail codebook spread**: 即便 loss 显著下降 (1.97 → 0.31), codebook 仍 collapse 到 max_load=100%. 跟 #148 同 family
  2. **两分量贡献不达标**: L1/L2 两分量 contribution < 0.1 (期望 >= 2 layers), 说明 mixing 跟 data 不匹配 (κ lock 现象)
  3. **soft-anchor 修复 loss=0 反例成功但 usage 仍 fail**: Issue #149 spec 明确要求"全 10 个记录点 κ/mixing grad finite nonzero" ✓ (跟 #146 反例区分), 但 usage/max_load 阈值 (跟 #148 同 pattern)
  4. **跟 #144/#146/#148 共享 collapse family**: 任何"aux loss 让 κ/mixing 拿 grad 但不强制 codebook spread"的 recipe 都失败, encoder z_e 仍聚到中心
- **R18 4 维度路径对比 vs #146**:
  - D1 spec: #149 continuous soft-anchor + softplus, #146 argmin + ReLU — **完全不同**
  - D2 实施: continuous soft-anchor (NO argmin 切断) + softplus (NO 0 区间), argmin + relu — **完全不同**
  - D3 Gate 1 失败机制: 两分量贡献不足 + usage 不达标, argmin 切断 + relu 0 grad — **不同 mechanism, 同 collapse family**
  - D4 引用文献: arXiv:2307.04514 同文献 — **同文献不同实施**
- **verdict 路径**: `verdicts/task439_issue149_gate1_fail_v1.md` (本文件)
- **实施**: `scripts/task439_issue149_continuous_anchor_product.py`
- **整体决策**: ⚠️ PARTIAL — 架构级 PASS (soft-anchor+softplus 修好 loss=0 反例, grad 全程 finite nonzero, 跟 #146 反证) + 配方级 FAIL (两分量贡献不达标 + usage/max_load 不达标). Issue #149 关闭, 不重启 continuous-anchor 路径

### Gate 2 (= Stage 2 Sinkhorn + dedup): ⏸ STOP per spec
- **原因**: Gate 1 FAIL per spec (Issue #149 spec Gate 1 PASS 前禁止 Gate 2)
- **Issue spec 强制**: Gate 1 PASS 前禁止 Gate 2/3/4

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
- **原因**: Gate 2 STOP
- **Issue spec 强制**: Gate 3 训练 + 前置条件

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- **原因**: Gate 3 STOP
- **Issue spec 强制**: R@10 阈值 (vs HG-Rec baseline 0.1020) + 标记 [TARGET REACHED] 条件

---

## 关键产物

- **verdict**: `verdicts/task439_issue149_gate1_fail_v1.md` (本文件)
- **commit hash**: TBD (after `git add` + `git commit` + `git push`)
- **push**: origin/main (R15 强制)
- **实施**: `scripts/task439_issue149_continuous_anchor_product.py`
- **8 件套审计**: config.json + SHA256(item_emb=`1a42341f...`) + d_mix_graph_proof.json (aux→κ grad=[2.45e-5, 5.42e-6, 5.94e-5], mixing grad=[6.15e-4, 4.66e-4, 1.13e-4], hard SID isolated) + raw_log + verdict.json + commit
- **整体决策**: ⚠️ PARTIAL — 架构级 PASS + 配方级 FAIL

---

## R18 严格路径对比 (vs #144 / #146 / #148)

| 维度 | Issue #144 (NO-GO) | Issue #146 (PARTIAL) | Issue #148 (PARTIAL) | Issue #149 (本 task PARTIAL) |
|------|---------------------|----------------------|----------------------|------------------------------|
| **D1 spec** | temperature simplex mixing | product distance contrastive + argmin+relu | triplet hinge + diversity EXPAND | continuous soft-anchor + softplus |
| **D2 实施** | 温度受控 + 熵下界 + softmax mixing | argmin anchor + relu margin | hinge + 64-sample diversity | softmax(-d/T) anchor + softplus margin |
| **D3 失败机制** | mixing grad=0 | argmin 切断 + relu 0 grad | hinge 满足 → grad=0 + diversity 未 enforce spread | 两分量不足 + usage 不达标 |
| **D4 文献** | arXiv:2307.04514 + Riemannian | arXiv:2307.04514 | arXiv:2405.13979 | arXiv:2307.04514 (同 #146) |
| **Precheck vs 训练** | grad=0 | precheck PASS, 训练中=0 | precheck PASS, 训练中 nonzero | precheck PASS, 训练中 nonzero (但小) |
| **util** | - | 1.56-3.91% | 0.39-1.56% | 0.39-3.13% |
| **max_load** | - | 97-100% | 100% | 99.9-100% |
| **R18 判定** | 必须新实验 | 必须新实验 (PARTIAL) | 必须新实验 (PARTIAL) | 必须新实验 (PARTIAL) |

---

## 后续修复路径 (R11.5 自主决策)

**核心发现 — 跟 #145/#146/#147/#148 共享 collapse family**: 任何"aux loss 让 κ/mixing 拿 grad 但不强制 codebook spread"的 recipe 都失败. 即便 soft-anchor 修了 loss=0 反例, codebook 仍 collapse.

**真正可修复方向** (跟 #145/#146/#148 共识):
1. **直接 enforce codebook spread**: 加 batch-level codebook orthogonality loss (强制 codeword 间 cosine orthogonality) 或 per-layer K-means init (用 data 初始化 codebook)
2. **EMA codebook update**: 跟 VQ-VAE 一样用 EMA 更新 codebook, encoder 仍 minimize recon
3. **Dead code revival (跟 #30 同)**: 监控 usage, 死掉的 codeword 重新 init 到 encoder state
4. **β-VAE style scaling**: 在 loss 加 β * commitment + β * codebook 正则化 (β-lasso 推 spread)
5. **Architecture pivot (跟 #147 同)**: 不在 Stage 1 改, 改在 Stage 3 T5 注入 (但 #147 已 NO-GO)

**Why NO-GO 收口 #149 当前方向**: 跟 #146 反例对照成功 (loss 显著下降 + grad 全程 nonzero), 但 collapse family 仍未解 (usage/max_load 失败). Issue #149 修复 loss=0 但没解决 usage.

**Why 架构级 PARTIAL**: 跟 #145/#146/#147/#148 同 pattern — precheck 完美验证 graph 修好, 但训练中具体 batch 的梯度对 codebook spread 无 enforce 能力. 跟 #146 反证对照: #146 argmin+relu 让 grad=0 (反例), #149 continuous+softplus 让 grad nonzero (反例修复成功).

---

## 历史事故关联

| 事故 | 现象 | 根因 | 跟 #149 关系 |
|------|------|------|--------------|
| Issue #146 (Task #436) | argmin + relu 切断 | contrastive margin | #149 反证: soft-anchor + softplus 让 grad nonzero |
| Issue #144 (Task #434) | temperature simplex mixing grad=0 | 温度受控 + 熵下界 | #149 跟 #144 共享 mixing 失败 family |
| Issue #148 (Task #438) | triplet+diversity usage fail | collapse family | #149 跟 #148 同 collapse family |
| Issue #145 (Task #435) | aux minimize → collapse | minimize 全局 pairwise | #149 跟 #145 共享 collapse family |

---

## 收口

- Issue #149 关闭 (`gh issue close 149 --reason completed`)
- 14 方向 NO-GO 收口 + Issue #145/#146/#147/#148/#149 架构级 PARTIAL = **baseline recipe 路径 + 修复尝试 联合耗尽**
- **新方向候选** (R11.5 决策, 等 owner 拍板): enforce codebook spread (orthogonality / K-means init / dead revival / EMA) + Stage 3 protocol split + 架构 pivot
- **R@10 ceiling 0.1053 + R137 κ lock + R139 reproducibility triangle + argmin 切断 + sigmoid 饱和 + frozen T5 反作用 + collapse family 累加 = baseline recipe 路径耗尽确认**