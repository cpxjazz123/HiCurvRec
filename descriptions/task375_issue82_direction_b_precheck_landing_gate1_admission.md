# Task #375 / Issue #82 [方向B 准入] product预检证据落仓 + Gate1 准入

**日期**: 2026-07-31
**触发**: Issue #82 [方向B 准入] product预检证据落仓后进Gate1 — R16 强制 GitHub OPEN 处理
**前置**: Issue #79 闭环 (task372 R18 实证 Gate 1.5 10/10 PASS, **verdict 文件 pending 需修正 → 8a761f6**)
**任务**: 1. 修正 verdict/task372 commit pending → 8a761f6; 2. 落仓证据包 (commit hash + 关键路径 + R20 4 Gate 详细); 3. Gate1 训练入口 + 配置冻结清单
**结果**: ⏳ 进行中

---

## 1. R18 4 维度对比 (Issue #82 vs Issue #79)

| 维度 | Issue #79 (预检 10/10 PASS) | Issue #82 (落仓 + Gate1 准入) | 是否一致 |
|------|------|------|------|
| **D1 spec 摘录** | 10/10 markers PASS, logit/θ/embedding 扰动 + 梯度证据 | 落仓 commit/verdict 链接 + Gate1 训练入口 + 配置冻结清单 | ❌ 不同 (Issue #82 是"准入"层, 要求落仓 + 配置冻结) |
| **D2 实施核心** | Product3ComponentVectorQuantization + autograd 证据 | 静态配置冻结 (n_e/e_dim/M/κ_max/三分量/optimizer) + Gate1 入口 | ❌ 不同 |
| **D3 失败机制** | 三分量 product 合同未落到可微路径 | 配置冻结 + Gate1 入口定义 | ❌ 不同 |
| **D4 引用文献** | arXiv:2307.04514 + ACE-HGNN DOI 10.1109/ICDM51629.2021.00021 | 同 | ✅ 相同 |

→ **R18 3/4 维度不一致, 必须做实验 (R18 强制). Issue #82 是"准入"层, 要求落仓证据 + 配置冻结清单**.

## 2. Issue #82 spec 要求

| 要求 | 内容 | 来源 |
|------|------|------|
| **修正 #79 verdict/commit 链接** | verdict/task372 pending → 8a761f6, commit hash 必须具体 (R21 强制) | Issue #82 spec |
| **预检证据落仓** | commit hash + verdict 路径 + scripts 路径 + products 路径 + R20 4 Gate 详细 | Issue #82 spec |
| **Gate1 训练入口** | scripts/task375_issue82_gate1_entry.py + 配置冻结 | Issue #82 spec |
| **配置冻结清单** | n_e=[64,128,256], e_dim=32, M=3, κ_max=2.0, fixed_hyp_kappa=1.0, three_components weights, Adam lr=1e-3, epochs | Issue #82 spec |

## 3. Issue #82 Gate 决策 (R17/§19)

| R17 Gate | 决策 | 状态 |
|------|------|------|
| Gate 1 (= Stage 1 RQ-VAE/HRQVAE) | ⏸ STOP (Issue #82 是准入层, 不要求 Stage 1 训练) | Issue #82 spec "不是启动训练本身" |
| Gate 1.5 (= 预检 + autograd) | ⏸ 已 PASS (Issue #79 闭环, verdict 已修正 commit hash) | per Issue #82 spec |
| Gate 2 (= Stage 2 Sinkhorn) | ⏸ STOP per spec | Issue #82 spec "未补齐 commit/verdict 前禁止 Gate1" (但 Gate 1 也非 spec) |
| Gate 3 (= Stage 3 T5-mini) | ⏸ STOP per spec | Issue #82 spec 不要求 |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP per spec | Issue #82 spec 不要求 |

---

result: Issue #82 [方向B 准入 product预检证据落仓 + Gate1 准入] R22 + R21 立即开工. 3 步: 1) 修正 verdict/task372 pending → 8a761f6; 2) 落仓证据 (commit hash + 关键路径 + R20 4 Gate 详细); 3) Gate1 训练入口 + 配置冻结清单 + commit + push + comment(含 hash) + close. ⏳ 进行中.