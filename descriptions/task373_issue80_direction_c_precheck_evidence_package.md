# Task #373 / Issue #80 [方向C 预检] Stage3 metadata 接口证据包

**日期**: 2026-07-31
**触发**: Issue #80 [方向C 预检] Stage3 metadata 接口证据包 — R16 强制 GitHub OPEN 处理
**前置**: Issue #77 闭环 (task370 R18 实证 10/10, SIDMetadata + AttentionBiasStub + 10 markers)
**任务**: R18 4 维度 vs #77 对比 + 完整证据包 (metadata schema + batch 序列化/反序列化 + stub 开关 + 响应证据)
**结果**: ⏳ 进行中

---

## 1. R18 4 维度对比 (Issue #80 vs Issue #77)

| 维度 | Issue #77 (sid_metadata → attention stub 证据) | Issue #80 (Stage3 metadata 接口证据包) | 是否一致 |
|------|------|------|------|
| **D1 spec 摘录** | 完整 schema 字段 (layer_id/kappa_l/scale_l/confidence/mask) + 多层 metadata 独立 + 关闭等价 + 开启不同 logits | 证据包 = metadata schema + batch 序列化/反序列化 + stub 开关 + 响应证据 | ❌ 不同 (Issue #80 要求完整证据集合) |
| **D2 实施核心** | SIDMetadata dataclass + serialize/deserialize + AttentionBiasStub | 同 + structured schema 输出 + 完整序列化/反序列化 roundtrip + stub 开关对比 | ❌ 不同 |
| **D3 Gate 1 失败机制** | Stage3 vanilla T5 wrapper + 缺 curvature-conditioned embedding | 同 + 证据包要可审计可复现 | ❌ 不同 |
| **D4 引用文献** | arXiv:2309.04082 (Curve Your Attention) | 同 | ✅ 相同 |

→ **R18 3/4 维度不一致, 必须做实验 (R18 强制). Issue #80 要求"证据包" = 完整证据集合 (schema + 序列化 + stub 开关 + 响应)**.

## 2. Issue #80 证据包要求

| 证据类型 | 内容 | 来源 |
|------|------|------|
| **metadata schema** | layer_id (int), kappa_l (float), scale_l (float), assignment_confidence (float), mask (Tensor [T]) | SIDMetadata dataclass |
| **batch 序列化** | list[SIDMetadata] → dict {layer_id, kappas, scales, confidences, mask} | serialize_metadata 函数 |
| **batch 反序列化** | dict → list[SIDMetadata] roundtrip 一致 | deserialize_metadata 函数 |
| **stub 关闭等价** | stub.enabled=False → out == hidden (atol=1e-6) | AttentionBiasStub.forward |
| **stub 开启响应** | stub.enabled=True + 同一 token 不同 metadata → 不同 logits (atol=1e-3) | AttentionBiasStub.forward |
| **多层 metadata 独立** | per-layer bias diff > 1e-3 | AttentionBiasStub.proj |
| **gradient 路径** | stub.proj.weight.grad.abs().max() > 1e-8 | backward() |
| **padding mask 对齐** | padding 位置 bias=0, out = hidden | mask * bias |
| **verdict 路径** | verdicts/task373_issue80_evidence_package_v2.md | 本次 |

## 3. Issue #80 Gate 决策 (R17/§19)

| R17 Gate | 决策 | 状态 |
|------|------|------|
| **Gate 1 (= Stage 1 RQ-VAE/HRQVAE)** | ⏸ STOP (Issue #80 是预检任务) | Issue #80 spec 是预检, 不要求 Stage 1 训练 |
| **Gate 1.5 = 预检 + stub 证据包** | ⏳ 进行中 | 完整证据包实施 + 10 markers + 结构化输出 |
| Gate 2 (= Stage 2 Sinkhorn) | ⏸ STOP | per spec |
| Gate 3 (= Stage 3 T5-mini) | ⏸ STOP | per spec |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP | per spec |

---

result: Issue #80 [方向C 预检 Stage3 metadata 接口证据包] R18 3/4 维度 vs #77 不一致必须做实验. 要求完整证据集合 (schema + 序列化 + stub 开关 + 响应). ⏳ 进行中.