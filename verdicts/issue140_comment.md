## Issue #140 R18+R20+R21 实证闭环 — 4 Gate 详细内容回答 (R17 + R20 + R21 强制)

### 6 件套审计 (Issue #140 spec 强制) — 全部已落地
1. **config**: `products/task430_issue140_update_order_audit/config.json`
2. **sha256**: item_emb.parquet `1a42341f01537d6db5f622e4831e0fc1a2039dc8b1288291dcc4963deee000cc`
3. **trace**: 1000 step × 10 record points (step 100/200/.../1000) 全落 verdict.json
4. **raw_log**: `logs/task430_issue140_update_order_audit.log` + `logs/task430_issue140_update_order_audit.launch.log`
5. **verdict**: `products/task430_issue140_update_order_audit/verdict.json` + `verdicts/task430_issue140_gate1_fail_v3.md`
6. **commit**: 048e96b (R21 v2: 不允许 "pending" 占位, commit hash 在 commit + push 后立即填入)

### Gate 0 (= 协议重建): ✅ PASS
- 关键数据: 1000-step 最小复现脚本 (`scripts/task430_issue140_update_order_audit.py`, 280 lines), data = item_emb.parquet (9922, 768), SHA256 `1a42341f01537d6db5f622e4831e0fc1a2039dc8b1288291dcc4963deee000cc`
- 配置: seed=42, codebook_size=[64,128,256], batch_size=256, lr=1e-4, ema_decay=0.99
- 实施: R18 强制 "两阶段更新" — Stage 1 encoder/backprop only (optimizer.step), Stage 2 detached hard-count EMA on full epoch (no_grad + geodesic EMA per codeword), 每 step #47 scale 同步在 kappa 更新后

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): ❌ FAIL (3/5 关键 check 失败)
- 关键数据:
  - **encoder→codebook 梯度泄漏 = 0** (3/3 layers codebook.grad max=0.0) ✅ R18 隔离更新顺序生效
  - **kappa #47 sync trace**: kappa_l_raw=0 → -0.1-softplus(0)=-0.793 全程锁定 ❌ (Issue #140 spec 要求 kappa 更新有同步 trace, 但 kappa_l_raw=0 constant — R137 κ lock 跟 #140 spec 互斥)
  - **all loss finite (no NaN/Inf)**: ✅ True
  - **final util**: 1.7% / 1.2% / 0.9% (≪90% 阈值) ❌ — EMA 每 100 step 触发但 codebook 仍 collapse 到 1-2 个码字
  - **final max_load**: 100% / 100% / 72.2% (≫5% 阈值) ❌ — 码字负载极度集中, EMA 投影到边界塌缩
  - **hard SID round-trip reproducible**: ✅ True
- 失败原因: **EMA detached hard-count 跟 encoder 优化的码字联合塌缩**. 即使 encoder 不偷渡梯度到 codebook (grad=0), detached geodesic EMA 仍把码字推到 boundary (sqrt(c)*‖x‖_E→1-eps). 这跟 Phase 0 mode collapse (task178/180/231/242) 共享根因 — codebook 推 boundary + EMA 沿边界聚集 = 100% single-codeword collapse. R18 spec 假设"隔离更新顺序能解决"假设 falsified.

### Gate 2 (= Stage 2 Sinkhorn): ⏸ STOP per spec
- 原因: Gate 1 FAIL, 无 SID 产出可推断 Sinkhorn

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
- 原因: Gate 2 STOP

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- 原因: Gate 3 STOP

### 关键产物
- commit hash: 048e96b
- push: origin/main
- verdict: verdicts/task430_issue140_gate1_fail_v3.md
- 实施: scripts/task430_issue140_update_order_audit.py
- products: products/task430_issue140_update_order_audit/verdict.json
- 整体决策: ❌ Gate 1 FAIL (R18 反证: 隔离更新顺序无法避开 EMA-driven codebook collapse)

### 联立 #426/#430 = hard-EMA 路径 NO-GO 收口
Task #426 (#134) 5-step audit grad_finite_nz=False (R137 κ lock) + Task #430 (#140) 1000-step 实证 EMA-driven collapse. 共同根因 = R137 κ lock + geodesic EMA → 码字推 boundary + 沿边界聚集 → 100% single-codeword collapse. Issue #140 closed (R16).