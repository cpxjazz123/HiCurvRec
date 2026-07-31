# Task #377 / Issue #84 [方向A Gate2] κ-aware codebook anti-collapse与SID恢复验证

**日期**: 2026-07-31
**触发**: Issue #84 [方向A Gate2] — R16 强制 GitHub OPEN 处理
**前置**: Issue #81 Gate 2 NO-GO (commit a748503, Phase 0 mode collapse 复刻, 4-digit SID unique 1/9922)
**任务**: 1. R18 4 维度 vs #81 对比; 2. κ-aware anti-collapse 诊断 (codebook norm/effective radius/distance range/assignment entropy); 3. Stage 1 50 epoch + Stage 2 Sinkhorn + 4-digit SID 验证
**结果**: ⏳ 进行中

---

## 1. R18 4 维度对比 (Issue #84 vs Issue #81)

| 维度 | Issue #81 (Gate2 κ-freeze 复刻 FAIL) | Issue #84 (κ-aware anti-collapse) | 是否一致 |
|------|------|------|------|
| **D1 spec 摘录** | κ-freeze warmup + Sinkhorn | κ-aware codebook anti-collapse (norm/entropy/distance range 受控) | ❌ 不同 (Issue #84 要求**新机制**) |
| **D2 实施核心** | LockableVQ + 1+8 epoch 短跑 | κ-aware 诊断 + 最小修复 (norm clipping/entropy reg) | ❌ 不同 |
| **D3 失败机制** | Phase 0 mode collapse 1+8 epoch 不恢复 | 必须避免坍缩 + 同步约束 codebook norm/distance | ❌ 不同 |
| **D4 引用文献** | arXiv:2405.13979v4 | 同 (Curvature-Aware Optimization) | ✅ 相同 |

→ **R18 3/4 维度不一致, 必须做新机制实验 (R18 强制). Issue #84 要求 κ-aware anti-collapse 诊断, 不能只延长 κ-freeze epoch 或重跑 Sinkhorn**.

## 2. Issue #84 spec 要求

| 要求 | 内容 | 来源 |
|------|------|------|
| **保留三层 learnable θ_l→κ_l** | L0 K64, L1 K128, L2 K256 | Issue #84 spec 框架合规 |
| **复用 #47 unified κ 距离公式** | κ-Stereographic distance | Issue #84 spec |
| **禁止 κ-freeze 复刻** | 不能只延长 epoch, 必须新机制 | Issue #84 spec "避开重复机制" |
| **κ-aware 诊断输出** | codebook norm / effective radius / distance range / assignment entropy / utilization / collision_rate | Issue #84 spec "必须记录" |
| **Gate 1 PASS 条件** | theta/scale/codebook 同步更新, 梯度有限非零, 无 NaN/Inf | Issue #84 spec Gate1 |
| **Gate 2 PASS 条件** | 4-digit SID unique ≥9500/9922, L0/L1/L2 util ≥90%, 偏差 ≤5pp | Issue #84 spec Gate2 |

## 3. Issue #84 Gate 决策 (R17/§19)

| R17 Gate | 决策 | 状态 |
|------|------|------|
| **Gate 1 (= Stage 1)** | ⏳ 进行中 | κ-aware 50 epoch 训练 + norm/entropy 诊断 |
| **Gate 2 (= Stage 2 Sinkhorn)** | ⏸ STOP per spec (Gate1 未 PASS 前) | Issue #84 spec |
| Gate 3 (= Stage 3 T5-mini) | ⏸ STOP per spec | Issue #84 spec |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP per spec | Issue #84 spec |

---

result: Issue #84 [方向A Gate2 κ-aware codebook anti-collapse] R22 + R19 立即并行启动 (GPU 0). 3 步: 1) κ-aware 诊断 (codebook norm/entropy/distance range); 2) Stage 1 50 epoch + Stage 2 Sinkhorn; 3) 4-digit SID unique ≥9500 验证 + commit + push + comment(含 hash) + close. ⏳ 进行中.