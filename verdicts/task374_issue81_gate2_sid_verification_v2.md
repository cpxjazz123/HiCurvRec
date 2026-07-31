# Issue #81 / Task #374 verdict — Gate 2 FAIL NO-GO (Phase 0 mode collapse 复刻)

**日期**: 2026-07-31
**Issue**: #81 [方向A Gate2] κ-freeze产物生成SID验证
**任务**: task374_issue81_gate2_sid_verification.py

---

## R17 4 Gate 决策

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): ⏸ 已 PASS (Issue #78 闭环)
- 关键数据: Issue #78 task371 1 warmup + 2 unfreeze epoch 实证 PASS, verdict 文件已修正 commit hash → 8a761f6
- 失败原因: 无
- verdict 路径: verdicts/task371_issue78_kappa_freeze_warmup_evidence_v2.md
- commit: 8a761f6

### Gate 2 (= Stage 2 Sinkhorn + 4-digit SID): ❌ FAIL NO-GO
- 状态: FAIL
- 关键数据:
  - 4-digit SID unique = **1/9922 = 0.01%** (目标 ≥9500/9922 ≈ 95.6%, 实际相差 99.99 pp)
  - 3-digit SID unique = 1/9922 (collision 99.99%)
  - L0/L1/L2 utilization = **1/64 / 1/128 / 1/256** (1.56% / 0.78% / 0.39%, 远低于 90% 阈值)
  - codebook **完全坍缩**: L0/L1/L2 各仅 1 个码字承载全部 9922 items
  - ckpt 已落盘: products/task374_issue81_gate2_sid_verification/free_curv_hrqvae_ckpt.pt (R12 强制)
- 失败原因: **Phase 0 mode collapse 复刻** (跟 task178/task180/task231/task242/task299 同模式)
  - 即使延长 EPOCHS_UNFREEZE 1+2 → 1+8 仍未恢复
  - κ-freeze warmup + Poincaré loss + β=0.25 + 短训不足以把码字从 ‖x‖_E ≈ 0.85 正常区逃出
  - Sinkhorn-Knopp 解码无法把已坍缩的码字恢复 (unique SID 制造假象, 但实际 100% 码字集中)
- 实施: scripts/task374_issue81_gate2_sid_verification.py (Stage 1 训练 + Stage 2 Sinkhorn + 4-digit dedup)

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
- 原因: Gate 2 FAIL, Issue #81 spec 明确"Gate 2 PASS 之前禁止 Stage 3"

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- 原因: Gate 2 FAIL, Issue #81 spec 明确"Gate 2 PASS 之前禁止 Stage 4"

---

## 整体决策: NO-GO 收口

- 路径: Issue #78 κ-freeze warmup → Issue #81 Gate 2 Sinkhorn SID 验证
- 结果: κ-freeze + Sinkhorn 路径 NO-GO, 跟 Phase 0 mode collapse 系列同根因
- 后续: 联立 task178/task180/task231/task242/task299 共同锁死 **baseline recipe (Poincaré loss + β=0.25 + κ-decouple) 内部 R@10 杠杆已穷尽**, 后续方向必须在架构层 (κ-Stereographic / T5 端 / decoder 端)
- R18 4 维度对比 (Issue #81 vs Issue #78): 3/4 不一致 (D1/D2/D3 不同, D4 同 arXiv:2405.13979v4)
- Issue #81 spec 强制: "Gate 2 不通过禁止 Stage 3/4" → 已 STOP

---

## 关键产物

- verdict: verdicts/task374_issue81_gate2_sid_verification_v2.md (本文件)
- script: scripts/task374_issue81_gate2_sid_verification.py
- ckpt: products/task374_issue81_gate2_sid_verification/free_curv_hrqvae_ckpt.pt
- evidence: products/task374_issue81_gate2_sid_verification/evidence_package.json
- description: descriptions/task374_issue81_direction_a_gate2_sid_verification.md
- Issue #78/#79/#80 commit hash: **8a761f6** (R21 强制明示, 无 pending)

---

result: Issue #81 [方向A Gate2 κ-freeze产物SID验证] Gate 2 FAIL NO-GO 收口 (Phase 0 mode collapse 1+8 epoch 仍 99.99% collision, 跟 task178/180/231/242/299 同根因, baseline recipe 内部 R@10 杠杆穷尽). verdict 落盘 + commit+push → issue comment(含 8a761f6) → close. ⏳ 待闭环.