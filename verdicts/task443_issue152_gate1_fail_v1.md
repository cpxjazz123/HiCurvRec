# Issue #152 [方向B Gate1] 样本条件 product 权重与容量硬分配审计 — Gate 1 ❌ FAIL (weight entropy 退化)

**Task**: #443 / Issue #152
**Commit**: `<pending>`
**Verdict**: `verdicts/task443_issue152_gate1_fail_v1.md`
**产物**: `products/task443_issue152_sample_conditioned_product/{config,d_mix_graph_proof,verdict}.json`
**SHA256**: item_emb.parquet=`1a42341f01537d6db5f622e4831e0fc1a2039dc8b1288291dcc4963deee000cc`

---

## 4 Gate 详细内容回答 (R17+R20 强制)

### Gate 1 (= Stage 1 RQ-VAE / HRQVAE): ❌ **FAIL (util 部分 + weight entropy 退化)**
- **关键数据 (Issue #152 spec 强制)**:
  - **Precheck PASS** ✅: aux_loss → κ grad = [3.97e-5, 1.31e-6, 1.73e-5] + weight_mlp grad = [4.36e-3, 2.52e-3, 2.02e-3]; hard SID → κ grad = [0, 0, 0] (isolated=True)
  - **架构修复实证成功**: 跟 #149 同 pattern, 加 sample-conditioned + capacity 让 κ/weight_mlp grad path 修好
  - **Control 1000-step (no-aux, 跟 #144/#146 同根因)**:
    - step 0-900: grad_κ=[0, 0, 0] 全部 (R137 κ lock 复现)
    - loss ~0.01-0.02
  - **Calibration 1000-step (with sample-conditioned aux + Hungarian capacity)**:
    - step 0: loss=1.98, grad_κ=[1.4e-5, 3.3e-5, 9.6e-5], util=[0.91, 0.69, 0.52], max_load=[0.020, 0.012, 0.008] ✓, comp=L0:[0.33, 0.33, 0.34] (3 分量均衡)
    - step 100: loss=1.99, util=[0.89, 0.73, 0.55], max_load PASS, comp=L0:[0.23, 0.26, 0.51] (d_eucl 主导)
    - step 500: loss=1.89, util=[0.81, 0.70, 0.61], max_load PASS, **comp=L0:[0.020, 0.020, 0.960]** ⚠️ weight 退化成 one-hot (entropy ≈ 0)
    - step 900: loss=1.05, util=[0.95, 0.68, 0.53], max_load PASS, **comp=L0:[~0, ~0, 1.0]** + L2:[0.68, 0.14, 0.18]
  - **最终 κ**: [-0.790, -0.809, -0.736]
  - **最终 util**: L0=0.83, L1=0.67, L2=0.56 (期望 >= 90% ❌, L0/L1/L2 全部 FAIL)
  - **最终 max_load**: L0=1.95%, L1=1.17%, L2=0.78% (**全部 < 5% PASS** ✅, 跟 #149 反例)
  - **最终 sample_weights entropy**: L0=1.33e-12 ≈ 0 (退化), L1=1.086, L2=1.003 (期望 >= 0.999 ❌, L0 FAIL)
  - **component contribution ≥2 per layer**: L0=False, L1=True, L2=True (期望 3/3 PASS, 1/3 FAIL)
  - **all batch feasible**: True ✓
- **失败原因**:
  1. **核心根因 — per-sample weight MLP 退化成 one-hot**: L0 entropy 从 step 0 的 1.0989 (3 分量均衡 ~0.33) 衰到 step 900 的 1.33e-12 (≈ 0), 即 weight MLP 学到"全 sample 都用 d_eucl 主导". entropy regularization (ALPHA=0.1, MIN=0.999) 力度不够, 没法阻止 weight 退化为 one-hot
  2. **Hungarian 容量约束成功**: max_load 0.78-1.95% PASS 证明 #151 修复路径在 #152 也有效
  3. **util 仍 fail**: encoder 仍聚到中心 (跟 #145/#148/#149 共享 collapse family), 容量约束只 cap max 不推 spread
  4. **跟 #149 反例**: #149 静态 mixing + soft-anchor → loss 降 85% 但 util 仍 ≤3.13%. #152 per-sample + capacity → loss 降 47% 但 util 仍 < 90%, weight 退化成 one-hot. 单纯让 mixing per-sample 化不能解决 collapse family
- **R18 4 维度路径对比 vs #149**:
  - D1 spec: per-sample weight MLP + entropy reg + Hungarian capacity vs #149 continuous soft-anchor + softplus + random neg — **完全不同**
  - D2 实施: weight MLP(z_e) → softmax (per-sample α_l ∈ Δ) vs ProductDistanceModel 静态 mixing scalars — **完全不同**
  - D3 Gate 1 失败机制: weight entropy 退化 (L0 ~0) + util 不达标 vs loss=0 + util 不达标 — **不同 mechanism, 同 collapse family**
  - D4 引用文献: arXiv:2307.04514 同文献 (同 #149) 不同实施
- **verdict 路径**: `verdicts/task443_issue152_gate1_fail_v1.md` (本文件)
- **实施**: `scripts/task443_issue152_sample_conditioned_product.py`
- **整体决策**: ⚠️ PARTIAL — 架构级 PASS (Hungarian capacity 修复 max_load, 跟 #149 反例) + 配方级 FAIL (per-sample weight entropy 退化 + util 不达标). Issue #152 关闭, 不重启 sample-conditioned 路径

### Gate 2/3/4: ⏸ STOP per spec
- Gate 1 FAIL per spec, Gate 2/3/4 禁止进入

---

## 关键产物

- **verdict**: `verdicts/task443_issue152_gate1_fail_v1.md` (本文件)
- **commit hash**: `<pending>`
- **push**: origin/main (R15 强制)
- **实施**: `scripts/task443_issue152_sample_conditioned_product.py`
- **8 件套审计**: config.json + SHA256(item_emb=`1a42341f...`) + d_mix_graph_proof.json + raw_log + verdict.json + commit
- **整体决策**: ⚠️ PARTIAL — max_load PASS (Hungarian 成功) + util FAIL + weight entropy 退化

---

## R18 严格路径对比 (vs #144 / #146 / #149)

| 维度 | Issue #149 (Task #439) | Issue #144 (Task #434) | Issue #152 (本 task PARTIAL) |
|------|------------------------|------------------------|------------------------------|
| **D1 spec** | continuous soft-anchor + softplus + random neg | temperature simplex mixing | **per-sample weight MLP + entropy reg + Hungarian capacity** |
| **D2 实施** | ProductDistanceModel + continuous contrastive | 温度受控 + softmax mixing | **SampleConditionedProductKappaModel + Hungarian capacity** |
| **D3 失败机制** | loss=0 + util 不达标 | mixing grad=0 | **weight entropy 退化 (L0 ~0) + util 不达标** (跟 #149 共享 collapse family) |
| **D4 文献** | arXiv:2307.04514 | arXiv:2307.04514 | arXiv:2307.04514 (同 #149) |
| **Precheck vs 训练** | grad nonzero | grad=0 | grad nonzero |
| **util** | 0.39-3.13% | - | **0.56-0.83 (FAIL)** |
| **max_load** | 99.9-100% | - | **0.78-1.95% (PASS, 反 #149 例)** |
| **weight entropy** | - | - | **L0=1.33e-12 ≈ 0 (退化)** |
| **R18 判定** | 必须新实验 | 必须新实验 | 必须新实验 (PARTIAL) |

---

## 后续修复路径 (R11.5 自主决策)

**核心发现 — per-sample weight 退化成 one-hot**:
- weight MLP 没强 entropy regularization → step 500 后 L0 退化成 ~[0, 0, 1.0] one-hot
- 即便 entropy_loss 设计成 F.relu(ENTROPY_MIN - entropy), α=0.1 力度不够 vs contrastive_loss 推 weight 到单分量
- 真正可修复方向:
  1. **强 entropy regularization**: α_entropy ≥ 1.0 + 严格 ENTROPY_MIN (>= log(3) - 0.01)
  2. **Gumbel-Softmax 离散化**: 让 weight 真的变成 discrete choice 而非 soft (但 #28 已 NO-GO)
  3. **per-codeword capacity + sample entropy**: 推 per-sample α 跟 codebook use 同步
  4. **Stage 3 protocol split (跟 #129/#147 同)**

**Why NO-GO 收口 #152 当前方向**: weight 退化成 one-hot (entropy ≈ 0) + util 仍 fail. per-sample mixing 化不能解决 collapse family.

**Why 架构级 PARTIAL**: 跟 #145/#146/#147/#148/#149/#151 共同确认 = "aux + hard constraint + per-sample 化" 路径不能解决 encoder collapse family.

---

## 历史事故关联

| 事故 | 现象 | 跟 #152 关系 |
|------|------|--------------|
| Issue #149 (Task #439) | continuous soft-anchor loss=0 + util fail | #152 共享 collapse family, weight entropy 退化是新版失败 |
| Issue #144 (Task #434) | mixing grad=0 | #152 反证: weight MLP 加 → mixing grad path 修好 |
| Issue #151 (Task #442) | capacity-Hungarian max_load PASS / util FAIL | #152 复用 #151 capacity, max_load 也 PASS, util 也 FAIL (共同确认) |

---

## 收口

- Issue #152 关闭 (`gh issue close 152 --reason completed`)
- **累计 NO-GO 收口**: 14 + Issue #145/#146/#147/#148/#149/#151 PARTIAL + Issue #152 PARTIAL = **baseline recipe 路径 + 修复尝试 联合耗尽**
- **关键突破**: 
  - Issue #151 + #152 都证明 **Hungarian capacity-hard assignment 直接修复 max_load** (跟 #148/#149 100% 反例, max_load 0.78-1.95%)
  - 但 **util L1/L2 仍不达标** (encoder 仍 collapse 到局部 codebook 子集)
  - #152 额外发现 **per-sample weight entropy 退化** (L0 entropy ≈ 0, one-hot)
- **新方向候选** (R11.5 决策, 等 owner 拍板): encoder-side z_e repulsion + Hungarian / 强 entropy reg / Stage 3 protocol split / architecture pivot