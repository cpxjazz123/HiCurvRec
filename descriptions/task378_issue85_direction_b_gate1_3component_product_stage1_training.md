# Task #378 / Issue #85 [方向B Gate1] 三分量product混合权重Stage1训练验证

**日期**: 2026-07-31
**触发**: Issue #85 [方向B Gate1] — R16 强制 GitHub OPEN 处理
**前置**: Issue #82 准入层 PASS (commit a748503, Product3ComponentVQ autograd 10/10 PASS)
**任务**: 1. R18 4 维度 vs #82 对比; 2. 三分量 product Stage 1 真实训练 (200 epoch); 3. mixing weights + component score 联合 loss + L0/L1/L2 util/collision 验证
**结果**: ⏳ 进行中

---

## 1. R18 4 维度对比 (Issue #85 vs Issue #82)

| 维度 | Issue #82 (准入层 PASS) | Issue #85 (Stage 1 训练) | 是否一致 |
|------|------|------|------|
| **D1 spec 摘录** | 落仓 + 配置冻结 + Gate1 入口 | Stage 1 200 epoch 训练 + mixing weights + 6 项指标 | ❌ 不同 (Issue #85 是**真实训练**) |
| **D2 实施核心** | forward shape 验证 | 训练 loop + per-layer mixing softmax + component score 组合 | ❌ 不同 |
| **D3 失败机制** | N/A (未训练) | Stage 1 必须避免 mixing collapse / 单分量 >0.98 | ❌ 不同 |
| **D4 引用文献** | arXiv:2307.04514 + ACE-HGNN DOI 10.1109/ICDM51629.2021.00021 | 同 | ✅ 相同 |

→ **R18 3/4 维度不一致, 必须做 Stage 1 训练 (R18 强制). 不能把预检 autograd PASS 外推成训练成功**.

## 2. Issue #85 spec 要求

| 要求 | 内容 | 来源 |
|------|------|------|
| **保留三层 learnable θ_l→κ_l** | L0 K64, L1 K128, L2 K256 | Issue #85 spec 框架合规 |
| **三分量 product manifold** | learned-κ + fixed-hyp(κ=1.0) + Euclidean | Issue #85 spec |
| **learnable mixing weights** | per-layer softmax(logits_l) over 3 components, 进入 optimizer | Issue #85 spec |
| **commitment/codebook loss 可微路径** | 三分量 score 在 assignment 前组合 + commitment/codebook loss 可微 | Issue #85 spec |
| **禁止 fixed-only / global shared / pure Euclidean** | Issue #85 spec 框架合规 |
| **训练预算** | seed=42, K=[64,128,256], e_dim=32, M=3, 由外部执行者按项目规范决定 | Issue #85 spec |
| **Gate 1 PASS** | L0/L1/L2 util ≥90%, collision ≤0.20, mixing 不长期单分量 >0.98, 无 NaN/Inf | Issue #85 spec Gate1 |

## 3. Issue #85 Gate 决策 (R17/§19)

| R17 Gate | 决策 | 状态 |
|------|------|------|
| **Gate 1 (= Stage 1 RQ-VAE/HRQVAE)** | ⏳ 进行中 | Product3ComponentVQ + 200 epoch 训练 |
| Gate 2 (= Stage 2 Sinkhorn) | ⏸ STOP per spec | Issue #85 spec "Gate1 未 PASS 前" |
| Gate 3 (= Stage 3 T5-mini) | ⏸ STOP per spec | Issue #85 spec |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP per spec | Issue #85 spec |

---

result: Issue #85 [方向B Gate1 三分量 product Stage 1 训练] R22 + R19 立即并行启动 (GPU 1). 3 步: 1) Product3ComponentVQ 训练 loop + mixing softmax; 2) Stage 1 200 epoch + per-layer util/collision/mixing 监控; 3) Gate1 PASS 验证 + commit + push + comment(含 hash) + close. ⏳ 进行中.