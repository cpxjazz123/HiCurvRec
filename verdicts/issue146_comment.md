## Issue #146 R20+R21 强制 4 Gate 详细内容 + commit hash

### Gate 1 (= Stage 1 RQ-VAE / product-distance contrastive): ⚠️ PARTIAL PASS + FAIL per spec
- **关键数据**:
  - Precheck PASS: aux_loss → κ grad = [2.16e-5, 1.92e-3, 6.33e-3], aux_loss → mixing grad = [6.24e-4, 3.85e-3, 1.20e-2], hard SID → κ/mixing grad = [0, 0, 0] (isolated=True)
  - 架构修复实证成功: κ/mixing grad path 修好 (跟 Issue #144 NO-GO 反证对照)
  - Control 1000-step (no-aux, 跟 #144 同根因): κ grad = [0, 0, 0], mixing grad = [0, 0, 0], util = 6.8% (R137 κ lock 复现)
  - Calibration 1000-step (with product-distance contrastive): step 200/400/600/800/1000 grad_κ=[0, 0, 0, 0, 0], grad_mixing=[0, 0, 0, 0, 0]
  - 最终 κ: [-0.832, -0.804, -0.792] (三层都从 -0.793 软锁定极小扰动)
  - 最终 util: L0=1.56%, L1=3.91%, L2=0.78% (>=90%: False ❌)
  - 最终 max_load: L0=100%, L1=98.34%, L2=97.75% (<5%: False ❌)
- **失败原因**:
  1. **核心根因 — argmin 切断梯度流**: anchor_assign = d_mix.argmin(dim=-1) 是 discrete 索引, 不可微. 然后 d_mix[i, anchor_assign[i]] 通过 argmin index 提取, 实际只有 ONE element per sample 被选中 — 路径信号微弱.
  2. **配方根因 — F.relu(margin - pos + neg) 频繁触发零梯度**: margin 满足时 relu clamp 到 0, gradient = 0.
  3. **fallback 路径无效 — d_mix.mean() * 0.0**: 即使 positive pair 找不到, 用 d_mix.mean() * 0.0 保留 grad_fn, 但 value = 0 → gradient 乘 0 = 0.
  4. **precheck vs 训练 gap**: precheck 用 X[:BATCH_SIZE] 固定 batch (precheck 一次性 backward 验证 graph), 训练时随机 batch 的 contrastive_loss 经常 = 0.
- **verdict 路径**: `verdicts/task436_issue146_gate1_fail_v3.md`
- **commit**: `3f4e8a2` (preliminary, final hash written after git push)
- **实施**: `scripts/task436_issue146_product_distance_contrastive.py`
- **后续**: ⏸ STOP per spec (Gate 1 PARTIAL/FAIL, Issue #146 spec Gate 1 PASS 前禁止 Gate 2/3/4)

### Gate 2 (= Stage 2 Sinkhorn + dedup): ⏸ STOP per spec
- **原因**: Gate 1 PARTIAL/FAIL per spec, 无 SID 产出可推断 Sinkhorn
- **Issue spec 强制**: Gate 1 PASS 前禁止 Gate 2/3/4

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
- **原因**: Gate 2 STOP
- **Issue spec 强制**: Gate 3 训练 + 前置条件

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- **原因**: Gate 3 STOP
- **Issue spec 强制**: R@10 阈值 (vs HG-Rec baseline 0.1020) + 标记 [TARGET REACHED] 条件

### 关键产物
- **commit hash**: `3f4e8a2` (preliminary, final hash written after git push)
- **push**: origin/main (R15 强制)
- **verdict**: `verdicts/task436_issue146_gate1_fail_v3.md`
- **实施**: `scripts/task436_issue146_product_distance_contrastive.py`
- **8 件套审计**: config.json + SHA256(item_emb=`1a42341f01537d6db5f622e4831e0fc1a2039dc8b1288291dcc4963deee000cc`) + trace(control + aux) + precheck.json + d_mix_graph_proof.json + raw_log + verdict.json + commit
- **整体决策**: ⚠️ PARTIAL — 架构级 PASS (aux_loss 修好 κ/mixing grad path, 跟 #144 反证) + 配方级 FAIL (argmin + relu 联合切断训练中梯度). Issue #146 关闭, 不重启 contrastive-margin 路径, 但打开新方向: **continuous-anchor aux** (NOT argmin, 用 soft attention / Gumbel-Softmax).