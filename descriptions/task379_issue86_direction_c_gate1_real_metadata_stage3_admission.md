# Task #379 / Issue #86 [方向C Gate1] 真实SID metadata数据流进入Stage3准入

**日期**: 2026-07-31
**触发**: Issue #86 [方向C Gate1] — R16 强制 GitHub OPEN 处理
**前置**: Issue #83 准入层 PASS (commit a748503, 真实 metadata 提取最小路径就绪)
**任务**: 1. R18 4 维度 vs #83 对比; 2. Stage 1 真实训练产出 metadata 数据流 (per-layer κ/scale/confidence/mask); 3. batch serialize/deserialize 稳定 + Stage3 AttentionBiasStub 输入格式验证
**结果**: ⏳ 进行中

---

## 1. R18 4 维度对比 (Issue #86 vs Issue #83)

| 维度 | Issue #83 (准入层 PASS) | Issue #86 (Stage 1 metadata 数据流) | 是否一致 |
|------|------|------|------|
| **D1 spec 摘录** | 落仓 + 真实 metadata 提取最小路径 | Stage 1 真实训练 + metadata 数据流 + batch 序列化 | ❌ 不同 (Issue #86 是**真实 Stage1 数据流**) |
| **D2 实施核心** | 1 epoch 短跑 + serialize/deserialize 闭环 | Stage 1 50+ epoch + 稳定 metadata 产出 + Stage3 输入格式 | ❌ 不同 |
| **D3 失败机制** | N/A (1 epoch) | 必须保证 Stage1/2 真实 κ/scale/confidence/mask 稳定进入 T5 表征 | ❌ 不同 |
| **D4 引用文献** | arXiv:2309.04082 (Curve Your Attention) | 同 | ✅ 相同 |

→ **R18 3/4 维度不一致, 必须做 Stage 1 真实训练 (R18 强制). 不能把 metadata stub PASS 外推成 T5 训练成功**.

## 2. Issue #86 spec 要求

| 要求 | 内容 | 来源 |
|------|------|------|
| **Stage 1/2 SID 流程不变** | L0 K64, L1 K128, L2 K256, 三层 learnable θ_l→κ_l | Issue #86 spec 框架合规 |
| **保留 #83 sid_metadata schema** | layer_id, kappa_l, scale_l/codebook norm, assignment confidence, mask/padding | Issue #86 spec |
| **真实 Stage 1 产物** | per-layer κ/scale/confidence/mask, batch serialize/deserialize 稳定 | Issue #86 spec Gate1 |
| **禁止 embedding init 改 / fixed hyp_c / post-step retraction / 绕过 SID metadata** | Issue #86 spec 框架合规 |
| **Gate 1 PASS** | L0/L1/L2 metadata 字段齐全, shape 与 SID token 对齐, padding mask 正确, kappa/scale 非全常数, 无 NaN/Inf | Issue #86 spec Gate1 |

## 3. Issue #86 Gate 决策 (R17/§19)

| R17 Gate | 决策 | 状态 |
|------|------|------|
| **Gate 1 (= Stage 1 RQ-VAE/HRQVAE)** | ⏳ 进行中 | 真实 Stage 1 + metadata 数据流 + serialize/deserialize 稳定 |
| Gate 2 (= Stage 2 Sinkhorn) | ⏸ STOP per spec | Issue #86 spec "Gate1 未 PASS 前" |
| Gate 3 (= Stage 3 T5-mini) | ⏸ STOP per spec | Issue #86 spec |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP per spec | Issue #86 spec |

---

result: Issue #86 [方向C Gate1 真实 SID metadata 数据流] R22 + R19 立即并行启动 (GPU 2). 3 步: 1) 真实 Stage 1 50 epoch + per-layer κ/scale/confidence/mask 提取; 2) batch serialize/deserialize 稳定 + Stage3 AttentionBiasStub 输入格式; 3) Gate1 PASS 验证 + commit + push + comment(含 hash) + close. ⏳ 进行中.