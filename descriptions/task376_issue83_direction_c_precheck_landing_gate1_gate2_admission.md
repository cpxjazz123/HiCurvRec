# Task #376 / Issue #83 [方向C 准入] metadata预检证据落仓 + Gate1/Gate2 准入

**日期**: 2026-07-31
**触发**: Issue #83 [方向C 准入] metadata预检证据落仓后进Gate1/Gate2 — R16 强制 GitHub OPEN 处理
**前置**: Issue #80 闭环 (task373 R18 实证 Gate 1.5 10/10 PASS, **verdict 文件 pending 需修正 → 8a761f6**)
**任务**: 1. 修正 verdict/task373 commit pending → 8a761f6; 2. 落仓证据包 (commit hash + 关键路径 + R20 4 Gate 详细); 3. Gate1/Gate2 准入最小路径 (从 Stage1/2 产出真实 SID metadata)
**结果**: ⏳ 进行中

---

## 1. R18 4 维度对比 (Issue #83 vs Issue #80)

| 维度 | Issue #80 (预检 10/10 PASS) | Issue #83 (落仓 + Gate1/Gate2 准入) | 是否一致 |
|------|------|------|------|
| **D1 spec 摘录** | 10/10 markers PASS, schema/roundtrip/stub 开关/gradient/padding | 落仓 commit/verdict + 从 Stage1/2 产出真实 SID metadata 的最小路径 | ❌ 不同 (Issue #83 是"准入"层) |
| **D2 实施核心** | SIDMetadata + serialize/deserialize + AttentionBiasStub | 静态配置冻结 + 真实 metadata 产出路径 | ❌ 不同 |
| **D3 失败机制** | Stage3 vanilla T5 缺 curvature-conditioned embedding | 真实 SID metadata 从 Stage1 κ_l + scale_l + assignment 提取 | ❌ 不同 |
| **D4 引用文献** | arXiv:2309.04082 (Curve Your Attention) | 同 | ✅ 相同 |

→ **R18 3/4 维度不一致, 必须做实验 (R18 强制). Issue #83 是"准入"层, 要求落仓 + 真实 metadata 产出路径**.

## 2. Issue #83 spec 要求

| 要求 | 内容 | 来源 |
|------|------|------|
| **修正 #80 verdict/commit 链接** | verdict/task373 pending → 8a761f6, commit hash 必须具体 (R21 强制) | Issue #83 spec |
| **预检证据落仓** | commit hash + verdict 路径 + scripts 路径 + products 路径 + R20 4 Gate 详细 | Issue #83 spec |
| **Gate1/Gate2 准入最小路径** | scripts/task376_issue83_real_metadata_extraction.py + 从 Stage1 κ_l + Stage2 assignment 提取真实 metadata | Issue #83 spec |
| **真实 SID metadata 产出** | 每层 (layer_id, kappa_l, scale_l, assignment_confidence, mask) → SIDMetadata → serialize → Stage3 AttentionBiasStub 输入 | Issue #83 spec |

## 3. Issue #83 Gate 决策 (R17/§19)

| R17 Gate | 决策 | 状态 |
|------|------|------|
| Gate 1 (= Stage 1 RQ-VAE/HRQVAE) | ⏸ STOP (Issue #83 是准入层) | Issue #83 spec "未补齐 commit/verdict 前禁止 Gate1/2/3" |
| Gate 1.5 (= 预检 + stub) | ⏸ 已 PASS (Issue #80 闭环, verdict 已修正 commit hash) | per Issue #83 spec |
| Gate 2 (= Stage 2 Sinkhorn) | ⏸ STOP (Issue #83 准入层) | Issue #83 spec |
| Gate 3 (= Stage 3 T5-mini) | ⏸ STOP per spec | Issue #83 spec 不要求 Stage 3 训练 |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP per spec | Issue #83 spec 不要求 |

---

result: Issue #83 [方向C 准入 metadata预检证据落仓 + Gate1/Gate2 准入] R22 + R21 立即开工. 3 步: 1) 修正 verdict/task373 pending → 8a761f6; 2) 落仓证据 (commit hash + 关键路径 + R20 4 Gate 详细); 3) Gate1/Gate2 真实 metadata 产出最小路径 + commit + push + comment(含 hash) + close. ⏳ 进行中.