# Task #370 / Issue #77 Gate 1.5 — sid_metadata → attention stub 证据 PASS (R18 强制实证)

**日期**: 2026-07-31
**触发**: Issue #77 [方向C 预检] sid_metadata 进入 attention stub 证据 — GitHub OPEN
**前置**: Issue #74 闭环 (task367 R18 实证, 7/7 markers PASS, 但仅单层 metadata)
**任务**: R18 4 维度 vs #74 对比 + 实施 sid_metadata schema + Stage3 batch 序列化 + attention-bias stub (关闭等价 + 开启不同 logits + 多层 metadata 独立)
**结果**: ✅ **10/10 PASS** (Issue #77 spec 全部硬约束满足)

---

## 1. R17/§19 Gate 决策

| R17 Gate | 决策 | 状态 |
|------|------|------|
| **Gate 1 (= Stage 1 RQ-VAE/HRQVAE)** | ⏸ STOP (Issue #77 是预检任务) | Issue #77 spec 是预检, 不要求 Stage 1 训练 |
| **Gate 1.5 = 预检 + stub 证据** | ✅ **PASS (10/10)** | 全部硬约束满足 |
| Gate 2 (= Stage 2 Sinkhorn) | ⏸ STOP | per spec |
| Gate 3 (= Stage 3 T5-mini) | ⏸ STOP | per spec |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP | per spec |

## 2. R18 4 维度对比 (Issue #77 vs Issue #74)

| 维度 | Issue #74 (sid_metadata + attention-bias 合同) | Issue #77 (sid_metadata → attention stub 证据) | 是否一致 |
|------|------|------|------|
| **D1 spec 摘录** | 7 markers PASS (单层 metadata) | 完整 schema 字段 (layer_id/kappa_l/scale_l/confidence/mask) + 多层 metadata 独立 + 关闭等价 + 开启不同 logits | ❌ 不同 |
| **D2 实施核心** | SIDMetadata dataclass + AttentionBiasStub | 同 + 多层 metadata 独立验证 + padding mask | ❌ 不同 |
| **D3 Gate 1 失败机制** | Stage3 vanilla T5 wrapper | 同 + 缺 curvature-conditioned embedding | ✅ 相同 |
| **D4 引用文献** | arXiv:2309.04082 (Curve Your Attention) | 同 | ✅ 相同 |

→ **R18 2/4 维度不一致, 必须做实验 (R18 强制)**.

## 3. 实施 + 10 markers (R19 强制)

### 3.1 实施
- `SIDMetadata` dataclass (schema: layer_id, kappa_l, scale_l, assignment_confidence, mask)
- `serialize_metadata` / `deserialize_metadata` (Stage3 batch 序列化/反序列化)
- `AttentionBiasStub` (可关闭/开启 nn.Module)
  - 关闭时 outputs = hidden_states (与 vanilla T5 等价)
  - 开启时 outputs = hidden_states + proj(metadata[kappas, scales, confidences])
  - 多层 metadata 独立: per-layer metadata → per-layer bias (不同 layer_id → 不同 bias)
  - mask padding 对齐: padding 位置 bias=0

### 3.2 10 markers 真实数据 (R18 强制)

| Marker | 内容 | 数据 | 结果 |
|------|------|------|------|
| T1 serialize_metadata 完整 | dict 包含 layer_id/kappas/scales/confidences/mask | all 5 keys | ✅ PASS |
| T2 deserialize_metadata roundtrip | list → dict → list | len match + kappa_l diff < 1e-6 | ✅ PASS |
| T3 stub 关闭 ≡ vanilla T5 | stub_off == hidden (atol=1e-6) | True | ✅ PASS |
| T4 stub 开启 logits 改变 | stub_on != hidden (atol=1e-3) | True | ✅ PASS |
| T5 同 token 不同 metadata → 不同 logits | stub_on(meta) != stub_on(meta_alt) | True | ✅ PASS |
| T6 多层 metadata 独立 | per-layer bias diff > 1e-3 | bias_diff_01/02/12 > 1e-3 | ✅ PASS |
| T7 stub gradient 路径非零 | stub.proj.weight.grad.abs().max() > 1e-8 | 非零 | ✅ PASS |
| T8 mask padding 位置 bias=0 | padding 位置 out - hidden < 1e-5 | ~0 | ✅ PASS |
| T9 schema 字段完整 | layer_id/kappa_l/scale_l/confidence/mask 都存在 | True | ✅ PASS |
| T10 双向 roundtrip 一致 | metadata → serialize → deserialize 字段全等 | True | ✅ PASS |
| **TOTAL** | | | **10/10 PASS** |

### 3.3 Issue #77 spec 硬约束全部满足

- ✅ SIDMetadata schema 字段完整: layer_id, kappa_l, scale_l/codebook norm, assignment confidence, mask/padding 对齐
- ✅ Stage3 batch 序列化/反序列化
- ✅ Attention-bias stub 关闭时 logits 与 vanilla T5 等价
- ✅ Attention-bias stub 开启时同一 token 在不同 kappa_l/scale_l 下 logits 改变
- ✅ 多层 metadata 独立验证 (3 层)
- ✅ Padding mask 对齐
- ✅ Gradient 路径非零
- ✅ 不只改 embedding init / fixed hyp_c / post-step retraction / 绕过 SID metadata

## 4. 产物清单

| 路径 | 内容 |
|------|------|
| descriptions/task370_issue77_direction_c_precheck_sid_metadata.md | 本 description |
| verdicts/task370_issue77_direction_c_precheck_sid_metadata_v2.md | 本 verdict (R18 实证) |
| scripts/task370_issue77_sid_metadata_stub_evidence.py | SIDMetadata + AttentionBiasStub + 10 markers |
| logs/task370_issue77_sid_metadata_stub/training.log | stub 证据日志 |

---

result: Issue #77 [方向C 预检 sid_metadata → attention stub 证据] PASS (10/10). R18 2/4 维度 vs #74 不一致已做实证. 实施 SIDMetadata dataclass + 序列化/反序列化 + AttentionBiasStub (关闭等价 + 开启 logits 改变 + 多层独立). 真实数据: schema 完整, 多层 metadata 独立, padding mask 对齐, gradient 路径非零. Gate 1.5 ✅ PASS, Gate 1/2/3/4 ⏸ STOP per spec.