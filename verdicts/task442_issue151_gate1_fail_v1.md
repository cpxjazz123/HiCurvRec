# Issue #151 [方向A Gate1] 精确容量约束硬双曲分配与κ校准联合审计 — Gate 1 ❌ FAIL (max_load PASS, util 部分 FAIL)

**Task**: #442 / Issue #151
**Commit**: `<pending>`
**Verdict**: `verdicts/task442_issue151_gate1_fail_v1.md`
**产物**: `products/task442_issue151_capacity_hard_hyperbolic/{config,capacity_graph_proof,verdict}.json`
**SHA256**: item_emb.parquet=`1a42341f01537d6db5f622e4831e0fc1a2039dc8b1288291dcc4963deee000cc`

---

## 4 Gate 详细内容回答 (R17+R20 强制)

### Gate 1 (= Stage 1 RQ-VAE / HRQVAE): ❌ **FAIL (max_load PASS + util 部分 FAIL)**
- **关键数据 (Issue #151 spec 强制)**:
  - **Precheck PASS** ✅: aux_loss → κ grad = [3.70, 4.08, 4.08] (provides_grad=True); hard SID → κ grad = [0, 0, 0] (isolated=True)
  - **架构修复实证成功**: 跟 #145/#148 同 pattern, 加 aux loss 让 κ grad path 修好
  - **Control 1000-step (no-aux, 跟 #143 同根因)**: 
    - step 0-900: grad_κ=[0, 0, 0] 全部 (R137 κ lock 复现)
    - loss ~0.01-0.02
  - **Calibration 1000-step (with pairwise aux + Hungarian capacity)**:
    - step 0: loss=38.32, grad_κ=[3.70, 4.08, 4.08], util=[0.83, 0.80, 0.65], **max_load=[0.020, 0.012, 0.008]** ✓
    - step 100: loss=31.87, grad_κ=[2.04, 3.90, 3.89], util=[0.84, 0.73, 0.50], max_load=[0.020, 0.012, 0.008] ✓
    - step 500: loss=28.36, grad_κ=[1.74, 3.25, 3.20], util=[0.86, 0.68, 0.54], max_load=[0.020, 0.012, 0.008] ✓
    - step 900: loss=25.74, grad_κ=[1.44, 2.74, 2.68], util=[0.92, 0.74, 0.55], max_load=[0.020, 0.012, 0.008] ✓
  - **最终 κ**: [-1.327, -1.346, -1.338] (从 0 软锁定到 ~-0.793 软锁定外推到更大 magnitude)
  - **最终 util**: L0=0.9375, L1=0.6719, L2=0.5820 (期望 >= 90% ❌, L0 PASS, L1/L2 FAIL)
  - **最终 max_load**: L0=1.95%, L1=1.17%, L2=0.78% (**全部 < 5% PASS** ✅, 跟 #148/#149 的 100% 完全反例)
  - **all batch feasible**: True ✓ (Hungarian 100% 可行)
  - **all loss finite**: True ✓
- **失败原因**:
  1. **核心根因 — Hungarian 容量约束成功消除 max_load, 但 L1/L2 util 仍不达标**: max_load PASS 证明 Issue #151 修复路径部分有效 (Hungarian 直接 enforce 每码字 ≤ cap), 但 L1/L2 利用率仍 < 90% 说明 encoder 仍聚到局部 codebook 子集
  2. **cap = ceil(B/K) + 1 太松**: K=64 L0 cap=5, K=128 L1 cap=3, K=256 L2 cap=2 — L2 cap=2 太严格, 256 个码字只能用 512 个 slot 但 batch=256 实际只需 256 slot, 留 256 slot 给 encoder 仍把 data 推到少数码字
  3. **encoder 仍聚到中心**: pairwise aux minimize 推 spread 失败 (跟 #145 共享 family), Hungarian 仅约束 hard SID 不约束 encoder z_e
  4. **跟 #148/#149 共享 collapse family**: 任何 aux + hard SID constraint 路径, encoder z_e 仍聚到中心 → codebook 利用率 < 90%
- **R18 4 维度路径对比 vs #148**:
  - D1 spec: capacity-hard Hungarian + per-codeword cap vs #148 triplet+diversity EXPAND — **完全不同机制**
  - D2 实施: linear_sum_assignment (Hungarian) + cap slot expansion vs hinge+64-sample diversity — **完全不同**
  - D3 Gate 1 失败机制: **max_load PASS (0.020/0.012/0.008) vs #148 100%** — **反例对照成功**, 但 L1/L2 util 仍不达标 (跟 #148 共享 encoder collapse family)
  - D4 引用文献: arXiv:2405.13979 + CrossRef VQ optimization (新检索) — **新检索, 跟 #148 同文献**
- **verdict 路径**: `verdicts/task442_issue151_gate1_fail_v1.md` (本文件)
- **实施**: `scripts/task442_issue151_capacity_hard_hyperbolic.py`
- **整体决策**: ⚠️ PARTIAL — 架构级 PARTIAL PASS (max_load 直接修复, 跟 #148/#149 反例, 验证 Hungarian capacity 路径有效) + 配方级 FAIL (util L1/L2 不达标). Issue #151 关闭, 不重启 capacity-Hungarian 路径

### Gate 2/3/4: ⏸ STOP per spec
- Gate 1 FAIL per spec, Gate 2/3/4 禁止进入

---

## 关键产物

- **verdict**: `verdicts/task442_issue151_gate1_fail_v1.md` (本文件)
- **commit hash**: `<pending>`
- **push**: origin/main (R15 强制)
- **实施**: `scripts/task442_issue151_capacity_hard_hyperbolic.py`
- **8 件套审计**: config.json + SHA256(item_emb=`1a42341f...`) + capacity_graph_proof.json + raw_log + verdict.json + commit
- **整体决策**: ⚠️ PARTIAL — max_load PASS (Hungarian 成功) + util 部分 FAIL

---

## R18 严格路径对比 (vs #145 / #148)

| 维度 | Issue #145 (Task #435) | Issue #148 (Task #438) | Issue #151 (本 task PARTIAL) |
|------|------------------------|------------------------|------------------------------|
| **D1 spec** | pairwise minimize + hard argmin | triplet hinge + diversity EXPAND + hard argmin | **pairwise minimize + Hungarian capacity (cap = ceil(B/K)+1)** |
| **D2 实施** | CalibrationKappaModel | TripletKappaModel | **CapacityConstrainedKappaModel** |
| **D3 失败机制** | minimize 全局均值 → collapse | hinge 满足 → grad=0 + diversity 未 enforce | **max_load PASS (Hungarian 成功)**, 但 L1/L2 util < 90% (encoder 仍聚到中心) |
| **D4 文献** | arXiv:2405.13979 | arXiv:2405.13979 | arXiv:2405.13979 + CrossRef VQ |
| **Precheck vs 训练** | grad 训练中有 | grad 训练中有 | grad 训练中有 |
| **util** | 1.56% | 0.39-1.56% | **0.67-0.94 (L0 PASS, L1/L2 FAIL)** |
| **max_load** | 100% | 100% | **0.78-1.95% (PASS, 反 #148 例)** |
| **R18 判定** | 必须新实验 | 必须新实验 | 必须新实验 (PARTIAL — max_load 修复成功) |

---

## 后续修复路径 (R11.5 自主决策)

**核心发现 — Hungarian capacity 修复 max_load 但未修复 util**:
- max_load 0.78-1.95% PASS 证明 Issue #151 修复路径部分有效
- L1/L2 util 仍不达标说明 encoder z_e 仍聚到中心 (跟 #145/#148 共享 collapse family)
- 真正可修复方向 (跟 #145/#146 共识):
  1. **Hungarian capacity + encoder z_e repulsion**: 加 encoder-side constraint 推 z_e spread
  2. **EMA codebook update (跟 #140 同, 但跟 Hungarian 组合)**
  3. **Stage 3 protocol split (跟 #129 同)**

**Why NO-GO 收口 #151 当前方向**: max_load 修复但 util 仍 fail. 单纯 hard constraint 不够, encoder 仍 collapse 到局部 codebook 子集.

**Why 架构级 PARTIAL**: 跟 #145/#146/#147/#148/#149 共同确认 = "aux loss 让 κ 拿 grad + hard constraint" 路径不能解决 encoder collapse family.

---

## 历史事故关联

| 事故 | 现象 | 跟 #151 关系 |
|------|------|--------------|
| Issue #148 (Task #438) | triplet+diversity usage fail | #151 max_load PASS 反例, 但 util 共享 family |
| Issue #145 (Task #435) | aux minimize → collapse | #151 共享 encoder collapse family |
| Issue #143 (Task #433) | 所有权 κ grad=0 | #151 反证: aux 加 → κ grad path 修好 |

---

## 收口

- Issue #151 关闭 (`gh issue close 151 --reason completed`)
- **累计 NO-GO 收口**: 14 + Issue #145/#146/#147/#148/#149 PARTIAL + Issue #151 PARTIAL (max_load PASS / util FAIL) = **baseline recipe 路径 + 修复尝试 联合耗尽**
- **关键突破**: Issue #151 证明 Hungarian capacity-hard assignment 直接修复 max_load (跟 #148 100% 反例), 但 util L1/L2 仍 fail. Issue #151 关闭, 不重启 capacity-Hungarian 路径
- **新方向候选** (R11.5 决策, 等 owner 拍板): Hungarian + encoder-side z_e repulsion / EMA + Hungarian 组合 / Stage 3 protocol split / architecture pivot