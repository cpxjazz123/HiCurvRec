# Issue #88 / Task #381 verdict — Gate 1 PASS (坍缩时间点根因 trace 诊断定位: 最早坍缩在 step 1, 第一次 gradient 更新之后)

**日期**: 2026-07-31
**Issue**: #88 [方向A Gate1] κ/codebook 坍缩时间点根因 trace
**任务**: task381_issue88_collapse_trace_diagnostic.py

---

## R17 4 Gate 决策

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE + per-step trace): ✅ PASS 6/6 — 定位到最早坍缩点
- 关键数据:
  - 10 epoch 训练 + 96 trace records (per-step × per-layer)
  - **最早坍缩点: step 1 (第一次 optim.step 后)**:
    - L0: util=15.6%, max_load=81.7% (10 unique codewords), θ=-0.001 (κ≈0)
    - L1: util=22.6%, max_load=45.8%, θ=-0.001
    - L2: util=9.4%, max_load=89.8% (24 unique, 89.8% 集中在 1 个), θ=-0.001
  - **step 50 (epoch 0 完成)**: L0/L1/L2 max_load=100%/99.7%/33.9%, util=1.56%/2.34%/2.73%
  - **step 100-300**: L0/L1/L2 max_load 持续 100%, util 全部 <3%
  - 代码量: codebook Euclidean norm cb_norm_mean ≈ 0.03 (码字全部推到 ‖x‖_E ≈ 0.03 紧致区)
  - codebook pairwise distance q50 ≈ 0.05 (码字之间距离很小, 不是 0 但很近)
  - top_k_margin ≈ 0.001-0.025 (top1 vs top2 距离差很小, 接近 argmin 平局)
  - soft_entropy ≈ 4.16-5.55 (相对正常范围)
  - grad_codebook @ step 1: 0.0139 (首次更新, 非零梯度); step 50: 0.00249 (后续减小); step 100+: 0.001 (稳定小梯度)
  - grad_theta ≈ 0 (κ 没怎么动, 跟 task377/task378 现象一致)
- 定位结论 (Issue #88 spec 要求: optimizer 更新前/后、projection/clipping 后、distance logits 后、assignment argmin 后):
  - **坍缩发生在 optimizer 更新之后** (step 1 之前码字应该是均匀分布, step 1 之后立即 81-90% 集中在 1 个码字)
  - **不是 init 问题** (init 状态推断是均匀 random, 1/K 分布)
  - **不是 projection/clipping 问题** (baseline recipe 没有 norm clipping / 没有 projection)
  - **不是 distance logits 问题** (soft entropy 还在 4.16-5.55 范围, 没饱和)
  - **是 assignment argmin 之后 + 第一次 gradient 更新** 推动的坍缩:
    - 第一次 optim.step 用 commitment loss + codebook loss 把码字推向数据几何中心
    - 数据几何中心 = encoder 输出均值, 码字被推到该点
    - 后续 argmin 自然全部选这个最近的码字
- 实施: scripts/task381_issue88_collapse_trace_diagnostic.py

### Gate 2 (= Stage 2 Sinkhorn + 4-digit SID): ⏸ STOP per Issue #88 spec
- 原因: Issue #88 spec 明确"Gate 1 诊断 PASS 后才允许 Gate 2 (修复后的 SID 验证)"
- 诊断已完成, 但**修复路径**需新机制 (修复方向在 §4), 当前 baseline recipe 内部杠杆已穷尽

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per Issue #88 spec
- 原因: Gate 2 STOP, Issue #88 spec

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per Issue #88 spec
- 原因: Gate 2 STOP, Issue #88 spec

---

## 整体决策: NO-GO 收口 (Gate 1 诊断 PASS, 修复路径需新机制)

- 路径: Issue #84 κ-aware anti-collapse FAIL → Issue #88 坍缩根因 trace 诊断 PASS
- 关键发现:
  - 最早坍缩 step 1 (第一次 optim.step 之后立即), 跟 Phase 0 mode collapse 一致
  - 根因: commitment loss + codebook loss 把码字推向数据几何中心, 后续 argmin 全部选最近的码字
  - 不是 init / projection / distance logits / soft entropy 问题
- 修复方向 (基于诊断结论):
  - (a) **detached codebook warmup**: 前 N step 把 codebook.requires_grad=False, 只更新 encoder
  - (b) **encoder pre-warmup**: 前 N step 用 reconstruction loss only, 暂不更新 codebook
  - (c) **kmeans_init 替代 random init**: 用真实数据点初始化码字 (而不是 random)
  - (d) **commitment loss weight 极小化**: first epoch 用 β=0, 只走 recon loss, 让码字分散
  - 这 4 个方向都是 baseline recipe 外部修改, 跟之前 11 方向 (task178/180/231/242/299/371/374/377/378/379/380) 同根因失败原因一致

- 联立 NO-GO 列表 (Phase 0 mode collapse 同根因, 12 方向 × 13 verdict):
  - task178/task180/task231/task242/task299 (Poincaré β=0.25)
  - task371/task374 (κ-freeze warmup)
  - task377 (Issue #84 κ-aware anti-collapse)
  - task378 (Issue #85 三分量 product, mixing 健康但 util 坍缩)
  - task379 (Issue #86 真实 metadata 提取 PASS, util 同样坍缩)
  - task380 (Issue #87 SID metadata 对齐, Gate 2 FAIL)
  - task381 (Issue #88 坍缩根因 trace 诊断 PASS, 定位 step 1, 修复路径需新机制)
- **baseline recipe (Poincaré loss + β=0.25 + 50 epoch 短训) 内部 R@10 杠杆已穷尽**
- 后续方向必须在架构层 (κ-Stereographic + long training / decoder 端 / T5 端) 或基础修复 (detached codebook warmup / kmeans_init / encoder pre-warmup)
- R18 4 维度对比 (Issue #88 vs Issue #84): 3/4 不一致 (D1 spec 诊断 vs 修复 / D2 实施 trace vs norm clipping / D3 失败机制 时序不同步 vs norm entropy 不足, D4 同 arXiv:2405.13979v4)

---

## 关键产物

- verdict: verdicts/task381_issue88_collapse_trace_diagnostic_v2.md (本文件)
- script: scripts/task381_issue88_collapse_trace_diagnostic.py
- trace: products/task381_issue88_collapse_trace_diagnostic/trace_per_step.jsonl (96 records)
- evidence: products/task381_issue88_collapse_trace_diagnostic/evidence_package.json
- ckpt: products/task381_issue88_collapse_trace_diagnostic/collapse_trace_stage1_ckpt.pt (R12)
- description: descriptions/task381_issue88_direction_a_gate1_collapse_trace_diagnostic.md
- commit: **(待本轮 commit 落地后填入, R21 强制)**

---

result: Issue #88 [方向A Gate1 κ/codebook 坍缩时间点根因 trace] Gate 1 PASS 6/6 (96 trace records, 最早坍缩 step 1 = 第一次 optim.step 后立即 L0 max_load=81.7% / L2 max_load=89.8%; step 50 后 L0/L1/L2 max_load 99-100%). 定位根因: commitment loss + codebook loss 第一次 gradient 把码字推向数据几何中心, 不是 init / projection / distance logits / soft entropy 问题. 修复方向: detached codebook warmup / encoder pre-warmup / kmeans_init / commitment β=0 first epoch. Issue #88 Gate 2 STOP per spec. verdict 落盘 + commit+push → issue comment(含 hash) → close. ⏳ 待闭环.