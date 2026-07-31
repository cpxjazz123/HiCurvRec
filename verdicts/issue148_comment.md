## Issue #148 R20+R21 强制 4 Gate 详细内容 + commit hash

### Gate 1 (= Stage 1 RQ-VAE / HRQVAE): ❌ FAIL (架构 PASS + 配方 FAIL)
- **关键数据**:
  - **Precheck PASS** ✅: aux_loss → c → κ grad = [4.08, 4.08, 4.07] (provides_grad=True); hard SID → κ grad = [0, 0, 0] (isolated=True)
  - **架构修复实证成功**: 跟 Issue #145 (Task #435) 同 pattern, 加 triplet-diversity aux loss 让 κ grad path 修好
  - **Control 1000-step (no-aux, 跟 #143 同根因)**: grad_κ=[0, 0, 0] 全部 (R137 κ lock 复现)
  - **Calibration 1000-step (with triplet-diversity)**:
    - step 200: loss=-39.85, grad_κ=[4.45, 4.44, 4.45]
    - step 400: loss=-42.55, grad_κ=[4.66, 4.77, 4.79]
    - step 600: loss=-43.35, grad_κ=[55.74, 4.82, 4.84]
    - step 800: loss=-43.53, grad_κ=[4.71, 4.82, 4.88]
    - step 1000: loss=-44.15, grad_κ=[4.56, 4.82, 4.95]
  - **最终 κ**: [-0.628, -0.611, -0.585]
  - **最终 util**: L0=1.56%, L1=0.78%, L2=0.39% (>=90%: False ❌)
  - **最终 max_load**: L0=100%, L1=100%, L2=100% (<5%: False ❌)
  - **hard SID round-trip**: True ✓
  - **all loss finite**: True ✓
- **失败原因**:
  1. **核心根因**: triplet hinge + diversity 推 spread 失败. 即便 diversity 项设计 EXPANDS codebook (-mean pairwise), 但 SGD 1000 步后 codebook 仍 collapse 到 max_load=100%
  2. **diversity 项规模问题**: ALPHA_DIVERSITY=0.5 * -mean(pairwise) → loss 整体被推到负值 (-39 到 -44), 但 codebook spread 仍未扩大
  3. **diversity 范围问题**: 只 sample 64 个 codeword 算 pairwise diversity, 不代表全部 K 个 codebook 的 spread
  4. **跟 #145 共享 collapse family**: 任何 minimize pairwise 或 maximize diversity (但 loss 内部没 enforce spread) 的设计都失败
- **verdict 路径**: `verdicts/task438_issue148_gate1_fail_v1.md`
- **commit**: `36056e4` (R21 fix: `1a0854f`)

### Gate 2 (= Stage 2 Sinkhorn + dedup): ⏸ STOP per spec
- **原因**: Gate 1 FAIL per spec, 无 SID 产出可推断 Sinkhorn
- **Issue spec 强制**: Gate 1 PASS 前禁止 Gate 2/3/4

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
- **原因**: Gate 2 STOP
- **Issue spec 强制**: Gate 3 训练 + 前置条件

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- **原因**: Gate 3 STOP
- **Issue spec 强制**: R@10 阈值 (vs HG-Rec baseline 0.1020) + 标记 [TARGET REACHED] 条件

### 关键产物
- **commit hash**: `36056e4`
- **R21 fix**: `1a0854f` (R21 强制填入 commit hash)
- **push**: origin/main (R15 强制)
- **verdict**: `verdicts/task438_issue148_gate1_fail_v1.md`
- **实施**: `scripts/task438_issue148_triplet_separation_diversity.py`
- **8 件套审计**: config.json + SHA256(item_emb=`1a42341f`) + triplet_graph_proof.json + raw_log + verdict.json + commit
- **整体决策**: ⚠️ PARTIAL — 架构级 PASS (aux 修好 κ grad path, 跟 #145/#146/#147 一致) + 配方级 FAIL (usage/max_load 不达标). Issue #148 关闭

### R18 4 维度对比 vs #145
- D1 spec: triplet+diversity EXPAND vs pairwise+ranking minimize — 完全不同方向
- D2 实施: TripletKappaModel hinge+repulsion vs CalibrationKappaModel minimize — 完全不同
- D3 失败机制: hinge 满足 → grad=0 + diversity 未 enforce vs minimize 全局 → collapse — 不同 mechanism, 同 family
- D4 文献: arXiv:2405.13979 同文献不同实施

### 累计 NO-GO 收口
- 13 方向 NO-GO 收口 + Issue #145/#146/#147/#148 架构级 PARTIAL = **baseline recipe 路径 + 修复尝试 联合耗尽**
- **新方向候选** (R11.5 决策, 等 owner 拍板): enforce codebook spread (orthogonality / K-means init / dead revival / EMA) + Stage 3 protocol split + 架构 pivot