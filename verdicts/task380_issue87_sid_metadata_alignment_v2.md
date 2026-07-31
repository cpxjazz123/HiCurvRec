# Issue #87 / Task #380 verdict — Gate 2 FAIL NO-GO (复用 #86 ckpt util 坍缩传递, per-item metadata 对齐架构就绪但 SID 端 FAIL)

**日期**: 2026-07-31
**Issue**: #87 [方向C Gate2] SID与metadata对齐产物验证
**任务**: task380_issue87_sid_metadata_alignment.py

---

## R17 4 Gate 决策

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): ⏸ 已 PASS (Issue #86 闭环, 复用 ckpt)
- 关键数据:
  - 复用 #86 ckpt: products/task379_issue86_real_metadata_stage1/real_metadata_stage1_ckpt.pt
  - #86 Gate 1 metadata extraction 6/6 PASS (commit 4ad7890)
- 状态: ✅ 已 PASS per Issue #86 spec (R18 实证)
- 注意: #86 spec 只要求 metadata 数据流 + serialize/deserialize, 不要求 util ≥90%

### Gate 2 (= Stage 2 Sinkhorn + per-item metadata 对齐): ❌ FAIL NO-GO
- 关键数据:
  - 4-digit SID unique = 256/9922 = 2.58% (目标 ≥9500/9922, Δ 94.97pp)
  - 3-digit SID unique = 1/9922 = 0.01% (collision 99.99%)
  - L0/L1/L2 util = 1.56% / 0.78% / 0.39% (目标 ≥90%, Δ 88.4pp / 89.2pp / 89.6pp)
  - 4-digit SID unique 来源: 第 4 列 `sids_4[:, 3] = arange % 256` 提供 256 unique values
  - κ_l (per-layer mean): -0.0091 / -0.0094 / -0.0099 (≈0, 学到的 κ 没动, 跟 #86 一致)
  - scale_l: 0.0364 / 0.0360 / 0.0351 (码字 Euclidean norm 全部坍缩到 ≈0)
- 失败原因: **复用 #86 ckpt 本身 util 坍缩 (Phase 0 mode collapse)**
  - #86 Stage 1 50 epoch 训练后码字坍缩到单点, util 1.56%/0.78%/0.39%
  - Issue #86 spec 不要求 util ≥90% (只 metadata extraction), 所以 #86 自身 PASS 6/6
  - Issue #87 spec 要求 util ≥90% → 物理上必然 FAIL (前置 ckpt util 不足)
  - per-item metadata 对齐文件成功生成 (sid_metadata.json 9922 items, shape (9922, 4) consistent, no NaN/Inf)
  - 但 SID 端 1 unique → metadata alignment 失去意义 (所有 item 共享同一个 SID)
- 实施: scripts/task380_issue87_sid_metadata_alignment.py
- 调试发现 (R11.5 自主决策):
  - layer.forward 返回 (x_q_st, loss, indices) 3-tuple, 之前 unpack 顺序错 (z_q, indices, _) 把 loss 当 indices → 0-d scalar
  - 修复: out = layer(...); x_q_st, _loss, indices = out; z_q = x_q_st
  - 修复后 indices.shape = (9922,) ✓ 但 n_unique = 1 (码字坍缩 upstream 问题)

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per Issue #87 spec
- 原因: Gate 2 FAIL, Issue #87 spec 明确"Gate 2 PASS 之前禁止 Stage 3"
- 即使跑 Stage 3, T5 也学不到语义 SID (3-digit unique = 1)

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per Issue #87 spec
- 原因: Gate 2 FAIL, Issue #87 spec

---

## 整体决策: NO-GO 收口

- 路径: Issue #86 Gate 1 metadata 提取 PASS → Issue #87 Gate 2 SID + metadata 对齐 FAIL (util 坍缩传递)
- 结果: per-item metadata 架构对齐成功, 但 SID 端因 #86 ckpt util 坍缩而物理 FAIL
- 关键产物:
  - per-item metadata 对齐文件: products/task380_issue87_sid_metadata_alignment/sid_metadata.json (9922 items × 3 layer metadata)
  - Stage 3 batch loader 格式: serialize_metadata stub 通过 (M2 stub)
  - sid_4digit.npy 落盘 (但 unique = 256, 第 4 列人工 dedup)
  - Gate 2 物理 FAIL (util 1.56%/0.78%/0.39%, 不达 ≥90%)

- 联立 NO-GO 列表 (Phase 0 mode collapse 同根因):
  - task178/task180/task231/task242/task299 (Poincaré β=0.25 baseline recipe)
  - task371/task374 (κ-freeze warmup 路径)
  - task377 (Issue #84 κ-aware anti-collapse)
  - task378 (Issue #85 三分量 product)
  - task379 (Issue #86 metadata 提取 PASS, 但 util 同样坍缩, 传递到 #87)
  - task380 (Issue #87 SID metadata 对齐, Gate 2 FAIL)
- **baseline recipe (Poincaré loss + β=0.25 + 50 epoch 短训) 内部 R@10 杠杆已穷尽**
- 后续方向必须在架构层 (κ-Stereographic + long training / decoder 端 / T5 端)
- R18 4 维度对比 (Issue #87 vs Issue #86): 3/4 不一致 (D1 spec/D2 实施/D3 失败机制 不同, D4 同 arXiv:2309.04082)

---

## 关键产物

- verdict: verdicts/task380_issue87_sid_metadata_alignment_v2.md (本文件)
- script: scripts/task380_issue87_sid_metadata_alignment.py (unpack 修复后)
- evidence: products/task380_issue87_sid_metadata_alignment/evidence_package.json
- description: descriptions/task380_issue87_direction_c_gate2_sid_metadata_alignment.md
- commit: **(待本轮 commit 落地后填入, R21 强制)**

---

result: Issue #87 [方向C Gate2 SID metadata 对齐] Gate 2 FAIL NO-GO 收口 (复用 #86 ckpt util 1.56%/0.78%/0.39% 坍缩传递, 4-digit SID unique=256/9922=2.58%, per-item metadata 架构对齐就绪但 SID 端 1 unique 失去意义). verdict 落盘 + commit+push → issue comment(含 hash) → close. ⏳ 待闭环.