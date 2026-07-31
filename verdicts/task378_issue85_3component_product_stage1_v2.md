# Issue #85 / Task #378 verdict — Gate 1 FAIL NO-GO (三分量 product 路径, util/collision 同样坍缩)

**日期**: 2026-07-31
**Issue**: #85 [方向B Gate1] 三分量 product 混合权重 Stage 1 训练验证
**任务**: task378_issue85_3component_product_stage1.py

---

## R17 4 Gate 决策

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE + 三分量 product 训练): ❌ FAIL NO-GO
- 关键数据:
  - Stage 1 50 epoch 训练完成 (ckpt 落盘 products/task378_issue85_3component_product_stage1/product3comp_stage1_ckpt.pt)
  - per-layer util @ ep49: L0=1.56%, L1=0.78%, L2=0.39% (目标 ≥90%, Δ 88.4pp / 89.2pp / 89.6pp)
  - collision: 99.09% (目标 ≤20%)
  - mixing weights @ ep49: L0 ≈ [0.333, 0.334, 0.333] (健康均匀, 无 mixing collapse)
  - L1/L2 mixing weights: 全部 ≈ 0.333 (均匀)
- 失败原因: **三分量 product manifold 路径仍 Phase 0 mode collapse**
  - mixing weights 健康 (无 collapse) 表明三分量 softmax 路径可微
  - 但码字本身仍坍缩到单点 (跟 task178/180/231/242/299/371/374/379/377 同根因)
  - 三分量组合不能绕过 baseline Poincaré loss + β=0.25 + 短训的坍缩陷阱
- 实施: scripts/task378_issue85_3component_product_stage1.py

### Gate 2 (= Stage 2 Sinkhorn): ⏸ STOP per Issue #85 spec
- 原因: Gate 1 FAIL, Issue #85 spec 明确"Gate 1 PASS 之前禁止 Gate 2/3/4"
- Stage 2 推断 util 同样坍缩 (L0/L1/L2 = 1.6%/0.8%/0.4%), 即使 Gate 2 跑也 FAIL

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per Issue #85 spec
- 原因: Gate 1 FAIL, Issue #85 spec

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per Issue #85 spec
- 原因: Gate 1 FAIL, Issue #85 spec

---

## 整体决策: NO-GO 收口

- 路径: Issue #82 三分量 product 准入 PASS → Issue #85 Stage 1 真实训练 FAIL
- 结果: 三分量 product manifold Stage 1 训练 NO-GO, mixing weights 健康但 util 坍缩
- 关键产物:
  - Product3ComponentHRQVAE (custom wrapper) 50 epoch 训练
  - per-layer mixing weights 健康 (≈0.333 各分量, 无 mixing collapse)
  - per-layer util 严重坍缩 (1.6%/0.8%/0.4%)
  - ckpt 落盘 (R12)

- 联立 NO-GO 列表 (Phase 0 mode collapse 同根因):
  - task178/task180/task231/task242/task299 (Poincaré β=0.25)
  - task371/task374 (κ-freeze warmup)
  - task377 (Issue #84 κ-aware anti-collapse)
  - task379 (Issue #86 真实 metadata 提取路径 PASS 但 util 同样坍缩)
- **baseline recipe 内部 R@10 杠杆已穷尽**
- 后续方向必须在架构层 (κ-Stereographic + long training / decoder 端 / T5 端)
- R18 4 维度对比 (Issue #85 vs Issue #82): 3/4 不一致 (D1 spec/D2 实施/D3 失败机制 不同, D4 同 arXiv:2307.04514 + ACE-HGNN DOI 10.1109/ICDM51629.2021.00021)

---

## 关键产物

- verdict: verdicts/task378_issue85_3component_product_stage1_v2.md (本文件)
- script: scripts/task378_issue85_3component_product_stage1.py
- ckpt: products/task378_issue85_3component_product_stage1/product3comp_stage1_ckpt.pt (R12)
- evidence: products/task378_issue85_3component_product_stage1/evidence_package.json
- description: descriptions/task378_issue85_direction_b_gate1_3component_product_stage1_training.md
- commit: **(待本轮 commit 落地后填入, R21 强制)**

---

result: Issue #85 [方向B Gate1 三分量 product Stage 1 训练] Gate 1 FAIL NO-GO 收口 (util 1.6%/0.8%/0.4% + collision 99.09%, mixing weights 健康 ≈0.333 但码字仍坍缩, 跟 task178/180/231/242/299/371/374/377/379 同根因). verdict 落盘 + commit+push → issue comment(含 hash) → close. ⏳ 待闭环.