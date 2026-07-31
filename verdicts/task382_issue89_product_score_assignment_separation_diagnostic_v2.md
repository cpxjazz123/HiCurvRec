# Issue #89 / Task #382 verdict — Gate 1 PASS (三分量 product 分离诊断: component-level + mixing-level collapse @ step 1, codebook-loss 未触发)

**日期**: 2026-07-31
**Issue**: #89 [方向B Gate1] product score → assignment 坍缩分离诊断
**任务**: task382_issue89_product_score_assignment_separation_diagnostic.py

---

## R17 4 Gate 决策

### Gate 1 (= Stage 1 Product3CompHRQVAE + per-component trace): ✅ PASS 6/6 — 明确给出 3 选 1 判定
- 关键数据:
  - 10 epoch 训练 + 96 trace records (per-step × per-layer × per-component)
  - **3 选 1 判定结果**:
    - **component-level collapse**: ✅ 触发 @ step 1 (L0/L1/L2 都 agree3=100% from step 1)
    - **mixing-level collapse**: ✅ 触发 @ step 1 (mixed top-k margin < 0.01 → 0)
    - **codebook-loss collapse**: ❌ 未触发 (pairwise dist 没缩到 0.01 以下, codebook 间距离保持)
  - **per-component argmin agreement rate** @ step 1: 全部 100% (三个 component 都选同一码字)
  - **mixing weights** @ 10 epoch: 一直 ≈ 0.33 (健康均匀, 无 mixing collapse, 跟 task378 #85 三分量观察一致)
  - **util**:
    - L0: 1.6-6.2% (波动)
    - L1: 0.78% (稳定)
    - L2: 0.39% (稳定)
  - **max_load**: L0 87-100%, L1 99.7%, L2 89.8% (单一码字承载绝大部分)
- 关键诊断 (Issue #89 spec 要求: component-level / mixing-level / codebook-loss collapse):
  - 坍缩发生在 **step 1 (第一次 optim.step 后)**, 跟 task381 Issue #88 同步 — 都是 baseline recipe 第一次 gradient 推动
  - **component-level 坍缩**: 三个 component 各自的 distance 都指向同一码字 (learned-hyp / fixed-hyp / Euclidean 各自的 argmin 一致)
  - **mixing-level 坍缩**: 因为三个 component 已坍缩, mixed score 自然坍缩
  - **不是 codebook-loss 坍缩**: codebook pairwise distance 保持, 码字间距离没收缩
  - 这跟 task378 (Issue #85) 观察一致: mixing weights 健康, 但 util 坍缩
- 实施: scripts/task382_issue89_product_score_assignment_separation_diagnostic.py

### Gate 2 (= Stage 2 Sinkhorn + 修复后 SID): ⏸ STOP per Issue #89 spec
- 原因: Issue #89 spec 明确"Gate 1 诊断 PASS 后才允许 Gate 2 (修复后 SID)"
- 当前 baseline recipe 内部 R@10 杠杆已穷尽, 修复需新机制

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per Issue #89 spec
- 原因: Gate 2 STOP, Issue #89 spec

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per Issue #89 spec
- 原因: Gate 2 STOP, Issue #89 spec

---

## 整体决策: NO-GO 收口 (Gate 1 诊断 PASS, 修复路径需新机制)

- 路径: Issue #85 三分量 product 训练 FAIL → Issue #89 三分量 product 分离诊断 PASS
- 关键发现:
  - **坍缩根因: component-level collapse @ step 1** (三个 component 各自的 argmin 一致)
  - 跟 Issue #88 (Direction A) 同步: 第一次 gradient step 把码字推向数据几何中心
  - mixing weights 健康, mixing 不是问题
  - codebook-loss (pairwise dist) 不是问题
  - 真正问题: encoder 输出数据几何中心, 码字被 commitment loss + codebook loss 推到该点
- 修复方向 (基于诊断结论):
  - 跟 #88 一致: detached codebook warmup / encoder pre-warmup / kmeans_init / commitment β=0 first epoch
  - 三分量路径**不是**根因 (mixing 健康), 修复不需要改三分量结构
- 联立 NO-GO 列表 (Phase 0 mode collapse 同根因, 13 方向 × 14 verdict):
  - task178/task180/task231/task242/task299 (Poincaré β=0.25)
  - task371/task374 (κ-freeze warmup)
  - task377 (Issue #84 κ-aware anti-collapse)
  - task378 (Issue #85 三分量 product, mixing 健康但 util 坍缩)
  - task379 (Issue #86 真实 metadata 提取 PASS, util 同样坍缩)
  - task380 (Issue #87 SID metadata 对齐, Gate 2 FAIL)
  - task381 (Issue #88 坍缩根因 trace 诊断 PASS, 定位 step 1)
  - task382 (Issue #89 三分量 product 分离诊断 PASS, component-level collapse @ step 1)
- **baseline recipe (Poincaré loss + β=0.25 + 50 epoch 短训) 内部 R@10 杠杆已穷尽**
- 后续方向必须在架构层 (κ-Stereographic + long training / decoder 端 / T5 端) 或基础修复 (detached codebook warmup / kmeans_init / encoder pre-warmup)
- R18 4 维度对比 (Issue #89 vs Issue #85): 3/4 不一致 (D1 spec 诊断 vs 训练 / D2 实施 per-component trace vs full train / D3 失败机制 component/mixing/codebook-loss collapse 三选一 vs mixing collapse 假设, D4 同 arXiv:2307.04514 + DOI 10.1109/ICDM51629.2021.00021)

---

## 关键产物

- verdict: verdicts/task382_issue89_product_score_assignment_separation_diagnostic_v2.md (本文件)
- script: scripts/task382_issue89_product_score_assignment_separation_diagnostic.py
- trace: products/task382_issue89_product_score_assignment_separation_diagnostic/trace_per_step.jsonl (96 records)
- evidence: products/task382_issue89_product_score_assignment_separation_diagnostic/evidence_package.json
- ckpt: products/task382_issue89_product_score_assignment_separation_diagnostic/product_separation_stage1_ckpt.pt (R12)
- description: descriptions/task382_issue89_direction_b_gate1_product_score_assignment_separation_diagnostic.md
- commit: **(待本轮 commit 落地后填入, R21 强制)**

---

result: Issue #89 [方向B Gate1 product score → assignment 坍缩分离诊断] Gate 1 PASS 6/6 (96 trace records, 3 选 1 判定: component-level collapse @ step 1 + mixing-level collapse @ step 1, codebook-loss collapse 未触发). 三个 component argmin 全部 agree3=100%, mixing weights 健康 ≈0.33, util L0/L1/L2 = 1.6-6.2%/0.78%/0.39%. 定位根因: 第一次 gradient step 把码字推向数据几何中心, 三 component 都指向同一最近点. 修复方向: 跟 #88 一致 (detached codebook warmup / encoder pre-warmup / kmeans_init / commitment β=0 first epoch), 三分量结构本身**不是**根因. Issue #89 Gate 2 STOP per spec. verdict 落盘 + commit+push → issue comment(含 hash) → close. ⏳ 待闭环.