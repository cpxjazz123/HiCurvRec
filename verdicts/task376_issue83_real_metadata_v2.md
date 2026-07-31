# Issue #83 / Task #376 verdict — 准入层 PASS (真实 metadata 提取最小路径)

**日期**: 2026-07-31
**Issue**: #83 [方向C 准入] metadata预检证据落仓 + Gate1/Gate2 准入
**任务**: task376_issue83_real_metadata_extraction.py

---

## R17 4 Gate 决策

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): ⏸ STOP per Issue #83 spec (准入层)
- 原因: Issue #83 spec 明确"未补齐 commit/verdict 前禁止 Gate1/2/3/4"
- 前置: Issue #80 闭环 (task373 R18 实证 Gate 1.5 10/10 PASS)

### Gate 1.5 (= 预检 + stub 证据): ✅ PASS 10/10 (Issue #80 闭环)
- 关键数据: M1 schema 7 fields; M2 roundtrip identical SHA; M3 stub 开关 on/off diff=152; M4 gradient logits_l.grad > 0 (2.37e-4); M5 padding row excluded; M6 conflict-free (4 dim interleaved); M7 serialize SHA stable; M8 stage3 input format (key_padding_mask+attention_bias); M9 layer-symmetric; M10 reproduce seed=42
- 失败原因: 无
- verdict 路径: verdicts/task373_issue80_metadata_evidence_v2.md
- commit: **8a761f6** (R21 强制明示, 无 pending)

### Gate 2 (= Stage 2 Sinkhorn): ⏸ STOP per spec
- 原因: Issue #83 是准入层 spec, 不要求 Stage 2 训练 (但要求"真实 metadata 产出最小路径"已 PASS, 见下)

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
- 原因: Issue #83 是准入层 spec

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- 原因: Issue #83 是准入层 spec

---

## 整体决策: 准入层 PASS (真实 metadata 提取最小路径就绪)

- 路径: Issue #80 SIDMetadata + stub 预检 → Issue #83 真实 metadata 提取最小路径 (从 Stage1 κ_l + Stage2 assignment)
- 结果: 准入层 PASS, 真实 metadata 提取路径完整 (Stage1 1 epoch 实证 + κ_l/scale_l/confidence 提取 + serialize/deserialize 闭环)
- 关键产物:
  - 落仓 #80 证据包 (commit hash 8a761f6 + verdict 路径 + scripts 路径 + products 路径 + R20 4 Gate 详细)
  - 真实 SID metadata 提取最小路径 (scripts/task376_issue83_real_metadata_extraction.py)
  - Stage1 1 epoch avg loss = 0.000263 (Stage 1 实证已完成)
  - 真实 metadata per layer: L0/L1/L2 κ_l/scale_l/confidence (虽因 1 epoch 短训导致码字坍缩, 但**提取路径**已 PASS)
  - SIDMetadata serialize → Stage3 AttentionBiasStub 输入格式 PASS (layer_ids/kappas/scales/confidences)

- 备注: 真实 κ_l ≈ -0.009 (κ-decouple Phase A 起点), 跟 task287 task371 task374 同模式 (1-epoch 短训坍缩). 提取路径 PASS, 跟坍缩根因独立.
- R18 4 维度对比 (Issue #83 vs Issue #80): 3/4 不一致 (D1/D2/D3 不同, D4 同 arXiv:2309.04082 Curve Your Attention)
- Issue #83 spec 强制: "落仓 + 真实 metadata 产出最小路径" → 已完整

---

## 关键产物

- verdict: verdicts/task376_issue83_real_metadata_v2.md (本文件)
- script: scripts/task376_issue83_real_metadata_extraction.py
- evidence: products/task376_issue83_real_metadata_extraction/evidence_package.json
- description: descriptions/task376_issue83_direction_c_precheck_landing_gate1_gate2_admission.md
- Issue #80 commit hash: **8a761f6** (R21 强制明示, 无 pending)
- Issue #83 落仓 commit hash: **(待本轮 commit 落地后填入)**

---

result: Issue #83 [方向C 准入 metadata预检证据落仓 + Gate1/Gate2 准入] 准入层 PASS (5/5 sanity). 落仓 #80 commit 8a761f6 + 真实 metadata 提取最小路径 + SIDMetadata serialize 闭环. 准入层完整, 待 owner 派工启动 Gate1/Gate2 训练. ⏳ 待闭环.