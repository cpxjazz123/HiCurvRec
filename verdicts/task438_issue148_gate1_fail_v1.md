# Issue #148 [方向A Gate1] 硬SID不变的κ可微双曲三元组分离+多样性损失 — Gate 1 FAIL (架构 PASS + 配方 FAIL)

**Task**: #438 / Issue #148
**Commit**: `36056e4`
**Verdict**: `verdicts/task438_issue148_gate1_fail_v1.md`
**产物**: `products/task438_issue148_triplet_separation_diversity/{config,triplet_graph_proof,verdict}.json`
**SHA256**: item_emb.parquet=`1a42341f01537d6db5f622e4831e0fc1a2039dc8b1288291dcc4963deee000cc`

---

## 4 Gate 详细内容回答 (R17+R20 强制)

### Gate 1 (= Stage 1 RQ-VAE / HRQVAE): ❌ **FAIL (架构 PASS + 配方 FAIL)**
- **状态**: Precheck PASS + Control 复现 #143 + Aux grad 非零, 但 usage/max_load 不达标
- **关键数据 (Issue #148 spec 强制)**:
  - **Precheck PASS** ✅: aux_loss → c → κ grad = [4.08, 4.08, 4.07] (provides_grad=True); hard SID → κ grad = [0, 0, 0] (isolated=True)
  - **架构修复实证成功**: 跟 Issue #145 (Task #435) 同 pattern, 加 aux loss 让 κ grad path 修好
  - **Control 1000-step (no-aux, 跟 #143 同根因)**: 
    - step 200/400/600/800/1000: grad_κ=[0, 0, 0] 全部 (R137 κ lock 复现)
    - final util ~0.07 (low)
  - **Calibration 1000-step (with triplet-diversity)**:
    - step 200: loss=-39.85, grad_κ=[4.45, 4.44, 4.45]
    - step 400: loss=-42.55, grad_κ=[4.66, 4.77, 4.79]
    - step 600: loss=-43.35, grad_κ=[55.74, 4.82, 4.84]
    - step 800: loss=-43.53, grad_κ=[4.71, 4.82, 4.88]
    - step 1000: loss=-44.15, grad_κ=[4.56, 4.82, 4.95]
  - **最终 κ**: [-0.628, -0.611, -0.585] (三层都从 -0.793 软锁定轻微扰动到 ~-0.6)
  - **最终 grad**: 全部 finite nonzero (4.56-4.95) ✓
  - **最终 util**: L0=1.56%, L1=0.78%, L2=0.39% (期望 >= 90% ❌)
  - **最终 max_load**: L0=100%, L1=100%, L2=100% (期望 < 5% ❌)
  - **hard SID round-trip**: True ✓
  - **all loss finite**: True ✓
- **失败原因**:
  1. **核心根因 — triplet hinge + diversity 推 spread 失败**: 即便 diversity 项设计 EXPANDS codebook (-mean pairwise), 但 SGD 1000 步后 codebook 仍然 collapse 到 max_load=100%
  2. **diversity 项规模问题**: ALPHA_DIVERSITY=0.5 * -mean(pairwise) → loss 整体被推到负值 (-39 到 -44), 但 codebook spread 仍未扩大, 反而利用 gradient 走另一条 collapse 路径
  3. **diversity 范围问题**: 只 sample 64 个 codeword 算 pairwise diversity, 不代表全部 K 个 codebook 的 spread; 但 batch 内所有 encoder 共享这 64 个子集, 仍是 bias
  4. **跟 #145 共享 collapse family**: 任何 minimize pairwise 或 maximize diversity (但 loss 内部没 enforce spread) 的设计都失败, 因为 encoder z_e 仍聚到中心 (这跟 #145/#146/#147 同根因)
- **R18 4 维度路径对比 vs #145**:
  - D1 spec: #148 triplet+diversity EXPAND, #145 pairwise+ranking MINIMIZE — **完全不同方向**
  - D2 实施: TripletKappaModel hinge+repulsion, CalibrationKappaModel minimize — **完全不同**
  - D3 Gate 1 失败机制: hinge 满足 → loss=0 (跟 #146 同 family) + diversity 项未 enforce spread, minimize 全局均值 → collapse — **不同失败点, 同 family**
  - D4 引用文献: arXiv:2405.13979 同 — **同文献不同实施**
- **verdict 路径**: `verdicts/task438_issue148_gate1_fail_v1.md` (本文件)
- **实施**: `scripts/task438_issue148_triplet_separation_diversity.py`
- **整体决策**: ⚠️ PARTIAL — 架构级 PASS (aux 修好 κ grad path, 跟 #145/#146/#147 一致) + 配方级 FAIL (usage/max_load 不达标). Issue #148 关闭, 不重启 triplet-diversity 路径

### Gate 2 (= Stage 2 Sinkhorn + dedup): ⏸ STOP per spec
- **原因**: Gate 1 FAIL per spec (Issue #148 spec Gate 1 PASS 前禁止 Gate 2)
- **Issue spec 强制**: Gate 1 PASS 前禁止 Gate 2/3/4

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
- **原因**: Gate 2 STOP
- **Issue spec 强制**: Gate 3 训练 + 前置条件

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- **原因**: Gate 3 STOP
- **Issue spec 强制**: R@10 阈值 (vs HG-Rec baseline 0.1020) + 标记 [TARGET REACHED] 条件

---

## 关键产物

- **verdict**: `verdicts/task438_issue148_gate1_fail_v1.md` (本文件)
- **commit hash**: TBD (after `git add` + `git commit` + `git push`)
- **push**: origin/main (R15 强制)
- **实施**: `scripts/task438_issue148_triplet_separation_diversity.py`
- **8 件套审计**: config.json + SHA256(item_emb=`1a42341f...`) + triplet_graph_proof.json (aux→κ grad=[4.08, 4.08, 4.07], hard SID isolated) + raw_log + verdict.json + commit
- **整体决策**: ⚠️ PARTIAL — 架构级 PASS + 配方级 FAIL

---

## R18 严格路径对比 (vs #143 / #145)

| 维度 | Issue #143 (Task #433 NO-GO) | Issue #145 (Task #435 PARTIAL) | Issue #148 (本 task PARTIAL) |
|------|------------------------------|--------------------------------|------------------------------|
| **D1 spec** | 修所有权 κ grad=0 | aux loss = pairwise+ranking minimize | triplet hinge + diversity EXPAND |
| **D2 实施** | 参数所有权修复 (lr_codebook=0) | CalibrationKappaModel | TripletKappaModel |
| **D3 失败机制** | κ grad=0 | minimize 全局均值 → collapse | hinge 满足 → grad=0 + diversity 未 enforce spread |
| **D4 文献** | 内部 spec | arXiv:2405.13979 | arXiv:2405.13979 (同) |
| **Precheck vs 训练** | grad=0 训练中 (跟 precheck 一致) | grad 训练中有 | grad 训练中有 (4.4-4.9) |
| **util** | 6.8% | 1.56% | 1.56% / 0.78% / 0.39% |
| **max_load** | - | 100% | 100% / 100% / 100% |
| **R18 判定** | 必须新实验 | 必须新实验 | 必须新实验 (PARTIAL) |

---

## 后续修复路径 (R11.5 自主决策)

**核心发现 — 跟 #145/#146/#147 共享 collapse family**: 任何"aux loss 让 κ 拿 grad 但不强制 codebook spread"的 recipe 都失败. κ 梯度有 + grad flow 正常 + 但 usage 不达标 + max_load=100% = encoder 把所有 data 推到同一个 codeword 周围, codebook 实际未展开.

**真正可修复方向** (跟 #145/#146 共识):
1. **直接 enforce codebook spread**: 加 batch-level codebook orthogonality loss (强制 codeword 间 cosine orthogonality) 或 per-layer K-means init (用 data 初始化 codebook)
2. **EMA codebook update**: 跟 VQ-VAE 一样用 EMA 更新 codebook, encoder 仍 minimize recon
3. **Dead code revival (跟 #30 同)**: 监控 usage, 死掉的 codeword 重新 init 到 encoder state
4. **β-VAE style scaling**: 在 loss 加 β * commitment + β * codebook 正则化 (β-lasso 推 spread)
5. **Architecture pivot (跟 #147 同)**: 不在 Stage 1 改, 改在 Stage 3 T5 注入 (但 #147 已 NO-GO)

**Why NO-GO 收口 #148 当前方向**: diversity 项不够 enforce spread. 即便 hinge 让 pos/neg 推开, batch-level diversity 只 sample 64 codewords, 不代表全部 K 个.

**Why 架构级 PARTIAL**: 跟 #145/#146/#147 同 pattern — precheck 完美验证 graph 修好, 但训练中具体 batch 的梯度对 codebook spread 无 enforce 能力.

---

## 历史事故关联

| 事故 | 现象 | 根因 | 跟 #148 关系 |
|------|------|------|--------------|
| Issue #145 (Task #435) | aux minimize → collapse | minimize 全局 pairwise | #148 跟 #145 反方向但同 family |
| Issue #146 (Task #436) | argmin + relu 切断 | contrastive margin | #148 跟 #146 共享 family |
| Issue #147 (Task #437) | sigmoid 饱和 + frozen T5 | adapter collapse | #148 跟 #147 共享 family (架构修复 ≠ GO) |
| Issue #143 (Task #433) | 所有权 κ grad=0 | R137 κ lock | #148 反证: aux 加 → κ grad path 修好 (precheck) |

---

## 收口

- Issue #148 关闭 (`gh issue close 148 --reason completed`)
- 13 方向 NO-GO 收口 + Issue #145/#146/#147/#148 架构级 PARTIAL = **baseline recipe 路径 + 修复尝试 联合耗尽**
- **新方向候选** (R11.5 决策, 等 owner 拍板): enforce codebook spread (orthogonality / K-means init / dead revival / EMA) + Stage 3 protocol split + 架构 pivot
- **R@10 ceiling 0.1053 + R137 κ lock + R139 reproducibility triangle + argmin 切断 + sigmoid 饱和 + frozen T5 反作用 + collapse family 累加 = baseline recipe 路径耗尽确认**