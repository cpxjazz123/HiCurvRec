# Task #373 / Issue #80 [方向C 预检] Stage3 metadata 接口证据包 — R17 PASS 10/10

**日期**: 2026-07-31
**触发**: Issue #80 [方向C 预检] Stage3 metadata 接口证据包 — R16 强制 GitHub OPEN 处理
**前置**: Issue #77 闭环 (task370 R18 实证 10/10, SIDMetadata + AttentionBiasStub)
**实施**: scripts/task373_issue80_evidence_package.py
**结果**: **R17 Gate 1.5 (预检) PASS 10/10** — 全部 10 个证据 markers 全部 PASS

---

## 1. R18 4 维度对比 (Issue #80 vs Issue #77)

| 维度 | Issue #77 | Issue #80 (本次) | 是否一致 |
|------|------|------|------|
| **D1 spec** | 完整 schema + 多层独立 + 关闭等价 + 开启不同 logits | 证据包 = schema + 序列化/反序列化 + stub 开关 + 响应 | ❌ 不同 (Issue #80 要求完整证据集合) |
| **D2 实施** | SIDMetadata + serialize/deserialize + AttentionBiasStub | 同 + structured schema 输出 + roundtrip + stub 开关对比 | ❌ 不同 |
| **D3 失败机制** | Stage3 vanilla T5 缺 curvature-conditioned embedding | 同 + 证据包可审计可复现 | ❌ 不同 |
| **D4 文献** | arXiv:2309.04082 (Curve Your Attention) | 同 | ✅ 相同 |

→ **R18 3/4 维度不一致, 必须做实验 (R18 强制). 本次实施完整证据包**.

## 2. 证据包内容 (6 类)

| 证据类型 | 内容 | 文件 |
|------|------|------|
| **metadata schema** | layer_id (int), kappa_l (float), scale_l (float), assignment_confidence (float), mask (Tensor [T]) | evidence_package.json → schema |
| **batch 序列化** | list[SIDMetadata] → dict {layer_id, kappas, scales, confidences, mask} shape [3]/[3]/[3]/[3]/[3,16] | 同 → serialized |
| **batch 反序列化 roundtrip** | dict → list[SIDMetadata] 一致 (layer_id/kappa_l/scale_l/confidence/mask 全部 allclose) | 同 → roundtrip_ok=True |
| **stub 关闭等价** | stub.enabled=False → out - hidden diff = 0.0 (atol=1e-6) | 同 → stub_disabled_diff=0.0 |
| **stub 开启响应** | stub.enabled=True + κ_l[0]=1.0→5.0 → out diff = 2.29 (atol=1e-3) | 同 → stub_enabled_diff=2.29 |
| **多层 metadata 独立** | κ diff=1.0 (1.0/1.5/2.0), scale diff=0.6 (0.5/0.8/1.1), conf diff=0.2 (0.8/0.7/0.6) | 同 → per_layer_metadata_diff |
| **gradient 路径** | stub params grad.max()=1.99e-2 > 1e-8 | 同 → stub_grad_max |
| **padding mask 对齐** | padding 位置 out - hidden diff = 0.0 (atol=1e-6) | 同 → padding_mask_diff=0.0 |

## 3. R17 Gate 决策 (4-Gate)

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE) — ⏸ STOP per Issue #80 spec
- 状态: STOP (Issue #80 是预检任务, spec 不要求 Stage 1 训练)

### Gate 1.5 (= 预检 + stub 证据包) — **PASS 10/10**
- 状态: PASS
- 关键数据: 10/10 markers PASS:
  - schema_complete (5 fields 全部存在)
  - multi_layer_3 (3 个 SIDMetadata)
  - roundtrip_ok (deserialize 一致)
  - stub_disabled_eq (diff=0.0)
  - stub_enabled_responsive (diff=2.29)
  - per_layer_independent (κ diff=1.0, scale diff=0.6)
  - gradient_path (1.99e-2 > 1e-8)
  - padding_aligned (diff=0.0)
  - serialized_has_keys (5 keys 全在)
  - deserialized_len (3 == 3)
- 失败原因: 无失败
- verdict 路径: verdicts/task373_issue80_evidence_package_v2.md (本次)
- commit: pending
- 后续: 如果 owner 想要 Stage 3 实际 T5 训练集成, 可启动 task373_ext (R19 立即推进备选)

### Gate 2 (= Stage 2 Sinkhorn) — ⏸ STOP per spec
- 状态: STOP
- 失败原因: Issue #80 是预检 spec, 不要求 Stage 2

### Gate 3 (= Stage 3 T5-mini) — ⏸ STOP per spec
- 状态: STOP
- 失败原因: Issue #80 是预检 spec, 不要求 Stage 3

### Gate 4 (= Stage 4 R@K eval) — ⏸ STOP per spec
- 状态: STOP
- 失败原因: Issue #80 是预检 spec, 不要求 Stage 4

## 4. 整体决策

**Gate 1.5 预检 PASS 10/10** — Issue #80 证据包完整闭环 (6 类证据齐全 + 10/10 sanity PASS). Issue #80 可关闭.

整体决策: **GO 闭环** (Issue #80 证据包完整交付)

---

result: Issue #80 [方向C 预检 Stage3 metadata 接口证据包] R17 Gate 1.5 PASS 10/10 (R18 4 维度 vs #77 3/4 不一致已实证). 6 类证据齐全: schema + 序列化 + roundtrip + stub 开关 + 响应 + gradient + padding. 后续 Gate 1/2/3/4 ⏸ STOP per Issue #80 预检 spec.
