# Task #370 / Issue #77 [方向C 预检] sid_metadata 进入 attention stub 证据

**日期**: 2026-07-31
**触发**: Issue #77 [方向C 预检] sid_metadata 进入 attention stub 证据 — R16 强制 GitHub OPEN 处理
**前置**: Issue #74 闭环完成 (task367 旧 verdict, R18 实证, 7/7 markers PASS)
**任务**: R18 4 维度 vs #74 对比 + 实施 sid_metadata schema + Stage3 batch 序列化 + attention-bias stub (关闭等价 + 开启不同 logits)
**结果**: ⏳ 进行中 (task383 跟踪)

---

## 1. R18 4 维度对比 (Issue #77 vs Issue #74)

| 维度 | Issue #74 (sid_metadata + attention-bias 合同) | Issue #77 (sid_metadata → attention stub 证据) | 是否一致 |
|------|------|------|------|
| **D1 spec 摘录** | sid_metadata schema + Stage3 batch 序列化 + attention-bias stub + 7 markers PASS | sid_metadata schema (layer_id, kappa_l, scale_l/codebook norm, assignment confidence, mask) + Stage3 batch 序列化 + attention-bias stub (关闭等价 + 开启 logits 改变) | ❌ 不同 (Issue #77 明确 schema 字段 + 关闭/开启等价/不同 logits) |
| **D2 实施核心** | SIDMetadata dataclass + serialize/deserialize + AttentionBiasStub | 同上 + explicit schema field check + 多层 metadata 独立验证 | ❌ 不同 |
| **D3 Gate 1 失败机制** | Stage3 仍 vanilla T5 wrapper | Stage3 缺 sid_metadata + curvature-conditioned embedding + curvature-aware attention markers | ✅ 相同 |
| **D4 引用文献** | arXiv:2309.04082 (Curve Your Attention) | 同 | ✅ 相同 |

→ **R18 2/4 维度不一致, 必须做实验 (R18 强制). Issue #77 明确 schema 字段 (layer_id/kappa_l/scale_l/confidence/mask) + 多层 metadata 独立验证 (vs #74 仅单层)**.

## 2. Issue #77 关键字段 (per spec)

- 定义并物化 sid_metadata schema: `layer_id`, `kappa_l`, `scale_l` 或 codebook norm, `assignment confidence`, `mask/padding 对齐`
- Stage3 batch 必须能序列化/反序列化 metadata
- attention-bias stub 必须读取 metadata:
  - **关闭时 logits 与 vanilla T5 等价**
  - **开启时同一 token 在不同 kappa_l/scale_l 下 logits 改变**
- 禁止: 只改 embedding init / fixed hyp_c / post-step retraction / 绕过 SID metadata
- Gate 2 同步产出 metadata 文件
- Gate 3 C0/C1/C2 证明 conditioning 非零 + 梯度有限 + 打乱 metadata 产生可证伪响应

## 3. Issue #77 Gate 决策 (R17/§19)

| R17 Gate | 决策 | 状态 |
|------|------|------|
| **Gate 1 (= Stage 1 RQ-VAE/HRQVAE)** | ⏸ STOP (Issue #77 是预检任务) | Issue #77 spec 是预检, 不要求 Stage 1 训练 |
| **Gate 1.5 = 预检 + stub 证据** | ⏳ 进行中 | sid_metadata schema + 序列化 + attention stub |
| Gate 2 (= Stage 2 Sinkhorn) | ⏸ STOP | per spec |
| Gate 3 (= Stage 3 T5-mini) | ⏸ STOP | per spec |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP | per spec |

---

result: Issue #77 [方向C 预检 sid_metadata → attention stub] R18 2/4 维度 vs #74 不一致必须做实验. 明确 schema 字段 + 多层 metadata 独立验证. ⏳ 进行中 (task383 跟踪).