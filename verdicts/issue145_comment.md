## Issue #145 R20+R21 强制 4 Gate 详细内容 + commit hash

### Gate 1 (= Stage 1 RQ-VAE / κ-可微几何校准辅助损失): ⚠️ PARTIAL PASS + FAIL per spec
- **关键数据**:
  - Precheck PASS: aux_loss → c → κ grad = [2.04, 1.98, 1.92], hard SID → κ grad = [0, 0, 0] (isolated)
  - Architecture 修复实证成功: κ grad path 修好 (跟 Issue #143 NO-GO 反证对照)
  - Control 1000-step (no-aux, 跟 #143 同根因): κ grad = [0, 0, 0] 全部, util = 6.8% (R137 κ lock 复现)
  - Calibration 1000-step (with aux): step 200/400/600/800/1000 grad_κ=[1.97, 1.81, 1.65, 1.51, 1.39], 最终 κ=[-1.347, -1.348, -1.349]
  - 最终 util: L0=1.56%, L1=0.78%, L2=0.78% (期望 >= 90% ❌)
  - 最终 max_load: L0=100%, L1=100%, L2=99.66% (期望 < 5% ❌)
- **失败原因**: aux_loss = α_pair · pairwise_calibration (mean of pairwise hyperbolic distance) + α_rank · ranking_calibration (relu(top-1 - top-2) margin minimization). pairwise mean minimize 推动 codebook 聚集到 encoder 几何中心, 所有 z_e 都跟最近 codebook 重合 → **codebook collapse** (util < 2%, max_load ≈ 100%). 跟 Issue #9/Issue #11 mode collapse 同根因, 但触发器从 hard SID 换成 aux loss.
- **verdict 路径**: `verdicts/task435_issue145_gate1_fail_v3.md`
- **commit**: `3a78b85`
- **实施**: `scripts/task435_issue145_kappa_calibration_aux_loss.py`
- **后续**: ⏸ STOP per spec (Gate 1 PARTIAL/FAIL, Issue #145 spec Gate 1 PASS 前禁止 Gate 2/3/4)

### Gate 2 (= Stage 2 Sinkhorn + dedup): ⏸ STOP per spec
- **原因**: Gate 1 PARTIAL/FAIL per spec, 无 SID 产出可推断 Sinkhorn (Issue #145 spec Gate 1 PASS 前禁止 Gate 2)
- **Issue spec 强制**: Gate 1 PASS 前禁止 Gate 2/3/4

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
- **原因**: Gate 2 STOP
- **Issue spec 强制**: Gate 3 训练 + 前置条件

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- **原因**: Gate 3 STOP
- **Issue spec 强制**: R@10 阈值 (vs HG-Rec baseline 0.1020) + 标记 [TARGET REACHED] 条件

### 关键产物
- **commit hash**: `3a78b85`
- **push**: origin/main (R15 强制)
- **verdict**: `verdicts/task435_issue145_gate1_fail_v3.md`
- **实施**: `scripts/task435_issue145_kappa_calibration_aux_loss.py`
- **8 件套审计**: config.json + SHA256(item_emb=`1a42341f01537d6db5f622e4831e0fc1a2039dc8b1288291dcc4963deee000cc`) + trace(control + aux, 10 record points each) + precheck.json + aux_graph_proof.json + raw_log + verdict.json + commit
- **整体决策**: ⚠️ PARTIAL — 架构级 PASS (aux_loss 修好 κ grad path, 跟 #143 反证) + 配方级 FAIL (aux_loss minimize 推 codebook collapse). Issue #145 关闭, 不重启 aux_minimize_pairwise 路径, 但打开新方向: **margin-based diversity aux loss** (NOT minimize distance, 而是 maximize data-codebook 分布多样性)