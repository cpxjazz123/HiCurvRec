# Task #374 / Issue #81 [方向A Gate2] κ-freeze产物生成SID验证

**日期**: 2026-07-31
**触发**: Issue #81 [方向A Gate2] κ-freeze产物生成SID验证 — R16 强制 GitHub OPEN 处理
**前置**: Issue #78 闭环 (task371 R18 实证 Gate 1 5/5 PASS, 实际 Stage 1 1 warmup + 2 unfreeze epoch 短跑, **verdict 文件 pending 需修正 → 8a761f6**)
**任务**: 1. 修正 verdict/task371 commit pending → 8a761f6; 2. R18 4 维度 vs #78 对比; 3. 实施 Gate 2 = Stage 2 Sinkhorn/SID 生成验证 (4-digit SID unique ≥9500/9922)
**结果**: ⏳ 进行中

---

## 1. R18 4 维度对比 (Issue #81 vs Issue #78)

| 维度 | Issue #78 (Gate1 κ-freeze warmup 证据包) | Issue #81 (Gate2 κ-freeze产物SID验证) | 是否一致 |
|------|------|------|------|
| **D1 spec 摘录** | Gate1: warmup/unfreeze 机制证据包 | Gate2: 跑 Stage2 Sinkhorn + 4-digit SID unique ≥9500/9922 | ❌ 不同 (Gate 维度不同, #81 是 Stage 2) |
| **D2 实施核心** | LockableVQ + 1+2 epoch 短跑 | Stage 2 Sinkhorn-Knopp + 4-digit dedup | ❌ 不同 |
| **D3 失败机制** | 1-3 epoch 短跑, 验证机制真实运行 | 4-digit SID 唯一性 / collision / utilization 偏差 | ❌ 不同 |
| **D4 引用文献** | arXiv:2405.13979v4 | 同 | ✅ 相同 |

→ **R18 3/4 维度不一致, 必须做实验 (R18 强制). Issue #81 要求 Gate 2 Sinkhorn/SID 实证, 必须重跑训练留 ckpt**.

## 2. Issue #81 spec 要求

| 要求 | 内容 | 来源 |
|------|------|------|
| **修正 #78 verdict/commit 链接** | verdict/task371 pending → 8a761f6, commit hash 必须具体 (R21 强制) | Issue #81 spec |
| **Gate2 = Stage 2 Sinkhorn/SID 验证** | 跑 Stage 2 Sinkhorn-Knopp 解码 (max_iters=30) + 4-digit dedup, 输出 (9922, 4) int array | Issue #81 spec |
| **4-digit SID unique ≥9500/9922** | 验证 SID 唯一性 ≥95.6% | Issue #81 spec |
| **逐层 utilization 偏差 ≤5pp** | L0/L1/L2 利用率 ≥95%, 跨层偏差 ≤5pp | Issue #81 spec |
| **禁止直接跳 Stage 3/4** | Gate 2 PASS 之前禁止 Stage 3/4 | Issue #81 spec |

## 3. Issue #81 Gate 决策 (R17/§19)

| R17 Gate | 决策 | 状态 |
|------|------|------|
| Gate 1 (= Stage 1 RQ-VAE/HRQVAE) | ⏸ 已 PASS (Issue #78 闭环, verdict 已修正 commit hash) | per Issue #81 spec |
| **Gate 2 (= Stage 2 Sinkhorn)** | ⏳ 进行中 | Sinkhorn 实施 + 4-digit SID 验证 |
| Gate 3 (= Stage 3 T5-mini) | ⏸ STOP per spec | Issue #81 spec "Gate 2 PASS 之前禁止 Stage 3" |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP per spec | Issue #81 spec "Gate 2 PASS 之前禁止 Stage 4" |

---

result: Issue #81 [方向A Gate2 κ-freeze产物SID验证] R22 + R21 立即开工. 3 步: 1) 修正 verdict/task371 pending → 8a761f6; 2) 实施 Stage 2 Sinkhorn + 4-digit SID 验证; 3) commit + push + comment(含 hash) + close. ⏳ 进行中.