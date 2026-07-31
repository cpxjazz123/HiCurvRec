# Issue #84 / Task #377 verdict — Gate 1 + Gate 2 FAIL NO-GO (Phase 0 mode collapse 复刻, κ-aware 诊断 + norm clipping + entropy reg 全部未救)

**日期**: 2026-07-31
**Issue**: #84 [方向A Gate2] κ-aware codebook anti-collapse 与 SID 恢复验证
**任务**: task377_issue84_kappa_aware_anticollapse.py

---

## R17 4 Gate 决策

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): ❌ FAIL NO-GO
- 关键数据:
  - Stage 1 50 epoch 训练完成 (ckpt 落盘 products/task377_issue84_kappa_aware_anticollapse/kappa_aware_anticollapse_ckpt.pt)
  - κ-aware norm clipping 强制码字 Euclidean norm ∈ [0.3, 0.95] → 实际 norms ≈0.30 (全部触 NORM_MIN 下限)
  - assignment entropy regularization (weight=0.05) 加入 loss
  - per-layer util @ ep49: L0=1.56%, L1=0.78%, L2=0.39% (目标 ≥90%, Δ 88.4pp / 89.2pp / 89.6pp)
  - collision @ ep49: 98.44% (目标 ≤20%)
- 失败原因: **Phase 0 mode collapse 复刻, κ-aware 机制 (norm clipping + entropy reg) 未救**
  - norm clipping 把码字强行推到 NORM_MIN=0.3 下限, 但码字本身仍坍缩到单点
  - entropy reg 在 50 epoch 短训下不足以扩散码字分布
  - κ-freeze 复刻同根因 (task371/task374)
- 实施: scripts/task377_issue84_kappa_aware_anticollapse.py

### Gate 2 (= Stage 2 Sinkhorn + 4-digit SID): ❌ FAIL NO-GO
- 关键数据:
  - 4-digit SID unique = 1/9922 (目标 ≥9500, Δ 99.99pp)
  - 3-digit SID unique = 1/9922 (collision 99.99%)
- 失败原因: 跟 Gate 1 同根因 — Phase 0 mode collapse + Sinkhorn 无法恢复已坍缩码字 (跟 task371/task374 模式一致)

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per Issue #84 spec
- 原因: Gate 1/2 FAIL, Issue #84 spec 明确"Gate 1 PASS + Gate 2 PASS 之前禁止 Stage 3"

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per Issue #84 spec
- 原因: Gate 1/2 FAIL, Issue #84 spec 明确"Gate 1 PASS + Gate 2 PASS 之前禁止 Stage 4"

---

## 整体决策: NO-GO 收口

- 路径: Issue #81 κ-freeze 复刻 NO-GO → Issue #84 κ-aware anti-collapse NO-GO
- 新机制实证 (norm clipping + entropy reg) 验证 FAIL: 50 epoch 仍 99.99% collision
- 联立 NO-GO 列表:
  - task178/task180/task231/task242/task299 (Poincaré β=0.25 mode collapse)
  - task371/task374 (κ-freeze warmup 路径)
  - task379 (Issue #86 真实 metadata, util 同样坍缩)
  - task378 (Issue #85 三分量 product, util 同样坍缩)
- **baseline recipe (Poincaré loss + β=0.25 + 短训) 内部 R@10 杠杆已穷尽**
- 后续方向必须在架构层 (κ-Stereographic + long training / decoder 端 / T5 端)
- R18 4 维度对比 (Issue #84 vs Issue #81): 3/4 不一致 (D1 spec/D2 实施/D3 失败机制 不同, D4 同 arXiv:2405.13979v4)

---

## 关键产物

- verdict: verdicts/task377_issue84_kappa_aware_anticollapse_v2.md (本文件)
- script: scripts/task377_issue84_kappa_aware_anticollapse.py
- ckpt: products/task377_issue84_kappa_aware_anticollapse/kappa_aware_anticollapse_ckpt.pt (R12)
- evidence: products/task377_issue84_kappa_aware_anticollapse/evidence_package.json
- description: descriptions/task377_issue84_direction_a_gate2_kappa_aware_anticollapse.md
- commit: **(待本轮 commit 落地后填入, R21 强制)**

---

result: Issue #84 [方向A Gate2 κ-aware codebook anti-collapse] Gate 1+2 FAIL NO-GO 收口 (Phase 0 mode collapse 复刻, norm clipping + entropy reg 50 epoch 未救, 跟 task178/180/231/242/299/371/374/378/379 同根因, baseline recipe 内部 R@10 杠杆穷尽). verdict 落盘 + commit+push → issue comment(含 hash) → close. ⏳ 待闭环.