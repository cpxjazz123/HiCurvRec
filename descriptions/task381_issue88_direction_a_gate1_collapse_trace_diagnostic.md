# Task #381 / Issue #88 [方向A Gate1] κ/codebook 坍缩时间点根因 trace

**日期**: 2026-07-31
**触发**: Issue #88 [方向A Gate1] κ/codebook 坍缩时间点根因 trace — R16 + R22 强制立即开工
**前置**: Issue #84 CLOSED (commit 09f78a8, κ-aware anti-collapse FAIL Gate1/Gate2)
**任务**: 1. R18 4 维度对比 vs Issue #84; 2. 10 epoch 训练 + per-step trace; 3. 定位"最早坍缩点"; 4. Gate 1 验证
**结果**: ⏳ 进行中

---

## 1. R18 4 维度对比 (Issue #88 vs Issue #84)

| 维度 | Issue #84 (κ-aware anti-collapse FAIL) | Issue #88 (坍缩根因 trace) | 是否一致 |
|------|------|------|------|
| **D1 spec 摘录** | norm clipping + entropy reg + 50 epoch 训练 | 10 epoch 训练 + per-step trace θ/κ/scale/codebook norm/pairwise dist/top-k margin/entropy/util/collision/grad | ❌ 不同 (#88 是诊断, 不是修复) |
| **D2 实施核心** | FreeCurvHRQVAE 50 epoch + norm clipping + entropy reg | FreeCurvHRQVAE 10 epoch + per-step diagnostic trace (无 norm clipping / 无 entropy reg) | ❌ 不同 |
| **D3 失败机制假设** | norm/entropy 不足 → 码字坍缩 | κ 更新 / codebook scale / distance logits / assignment 时序不同步 → 坍缩 | ❌ 不同 |
| **D4 引用文献** | arXiv:2405.13979v4 | 同 | ✅ 相同 |

→ **R18 3/4 维度不一致, 必须做 trace 诊断实证 (R18 强制)**.

## 2. Issue #88 spec 要求

| 要求 | 内容 | 来源 |
|------|------|------|
| **10 epoch 训练** | 不需 50 epoch 完整训练, 10 epoch 足够定位坍缩点 | Issue #88 spec Gate1 |
| **per-step trace** | θ, κ, scale, codebook norm, pairwise dist quantiles, top-k margin, entropy, util, collision, grad norm | Issue #88 spec |
| **定位"最早坍缩点"** | 首次出现单码字承载 >50% / >90% / >99% item 的 step/epoch | Issue #88 spec |
| **3 选 1 判定** | optimizer-step / projection-clipping / distance-logits / argmin 哪个最早坍缩 | Issue #88 spec |
| **PASS** | 明确定位最早坍缩点, trace 无 NaN/Inf, verdict/commit 可追踪 | Issue #88 spec Gate1 |
| **禁止** | 只报告最终 util / 无法定位坍缩点 / 缺 L0/L1/L2 任一层 / 复用 #84 机制但无新增诊断维度 | Issue #88 spec 框架合规 |

## 3. Issue #88 Gate 决策 (R17/§19)

| R17 Gate | 决策 | 状态 |
|------|------|------|
| **Gate 1 (= Stage 1 trace)** | ⏳ 进行中 | 10 epoch 训练 + per-step trace + 定位最早坍缩点 |
| Gate 2 (= Stage 2 Sinkhorn) | ⏸ STOP per Issue #88 spec | Gate 1 PASS (定位) 后才允许 |
| Gate 3 (= Stage 3 T5-mini) | ⏸ STOP per Issue #88 spec | Gate 1+2 PASS 之后 |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP per Issue #88 spec | Gate 1+2+3 PASS 之后 |

---

## 4. 实施策略 (R11.5 自主决策)

- 复用 task378 (Issue #85) FreeCurvHRQVAE 50 epoch 训练循环, 改成 10 epoch + per-step trace
- trace 输出: products/task381_issue88_collapse_trace/trace_per_step.jsonl (每 step 一行, per-layer)
- 定位: 单码字承载 >50% / >90% / >99% item 的最早 step
- GPU 0 立即启动 (R7 + R19 激进, 4 卡空闲)

---

result: Issue #88 [方向A Gate1 κ/codebook 坍缩时间点根因 trace] R22 + R19 立即开工 (GPU 0). 3 步: 1) 10 epoch 训练 + per-step trace (θ/κ/scale/codebook norm/pairwise dist/top-k margin/entropy/util/collision/grad); 2) 定位最早坍缩点 (optimizer/projection/distance logits/argmin); 3) Gate 1 验证 + commit + push + comment(含 hash) + close. ⏳ 进行中.