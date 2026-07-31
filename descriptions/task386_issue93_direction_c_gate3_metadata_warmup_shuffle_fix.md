# Task #386 / Issue #93 [方向C Gate3] metadata warm-up + 同模型 shuffle 修复验证

**日期**: 2026-07-31
**触发**: Issue #93 [方向C Gate3] metadata warm-up 与同模型 shuffle 修复验证 — R16 + R22 强制立即开工
**前置**: Issue #90 Gate 3 FAIL (commit c9cdb60, shuffle_metadata_diff=0, metadata path 未真正生效)
**任务**: 1. R18 4 维度对比 vs Issue #90; 2. zero-equivalent init + metadata scale warm-up + 同模型 on/off/shuffle 对比; 3. Gate 3 PASS 验证
**结果**: ⏳ 进行中

---

## 1. R18 4 维度对比 (Issue #93 vs Issue #90)

| 维度 | Issue #90 (受控消融 FAIL) | Issue #93 (warm-up + 同模型 shuffle 修复) | 一致? |
|------|------|------|------|
| **D1 spec** | 受控消融 C0/C1/C2 100 steps | warm-up schedule + 同模型 on/off/shuffle 对比, 训练更长 | ❌ |
| **D2 实施** | metadata_proj 默认 init + 100 steps | metadata_proj zero-equivalent init + warm-up scale 0→1 + 500 steps + 同模型三态对比 | ❌ |
| **D3 失败机制** | metadata path 未真正生效 (shuffle_diff=0) | zero-equivalent init 让 scale=0 时 metadata 完全不生效, warm-up 让 metadata 缓慢接入, 同模型对比消除 init 噪声 | ❌ (修复 D3 假设) |
| **D4 引用文献** | arXiv:2309.04082 (Curve Your Attention) | 同 | ✅ |

→ R18 3/4 维度不一致, 必须做修复实证 (R18 强制).

## 2. Issue #93 spec 要求

| 要求 | 内容 | 来源 |
|------|------|------|
| **zero-equivalent init** | metadata_proj weight=0, bias=0; scale=0 时 metadata 完全不生效 | Issue #93 spec Gate3 |
| **warm-up schedule** | metadata_scale 从 step 0=0 → step N=1.0 线性增 | Issue #93 spec Gate3 |
| **同模型对比** | 同一 ckpt 上做 metadata on/off/shuffle 三态对比 | Issue #93 spec Gate3 (避免 #90 独立 init 噪声) |
| **更长训练** | ≥500 steps (vs #90 100 steps) | Issue #93 spec Gate3 |
| **报告** | 同模型 on/off logits diff, shuffle metadata diff, grad norm, scale/weight norm, loss | Issue #93 spec Gate3 |
| **PASS** | shuffle_metadata_diff ≠ 0 且方向可重复; metadata path 梯度有限非零; on/off diff 非零 (排除 init 噪声) | Issue #93 spec Gate3 PASS |
| **FAIL** | shuffle_metadata_diff 仍为 0; metadata path detach; 或 on/off diff 来自独立 init | Issue #93 spec Gate3 FAIL |

## 3. Issue #93 Gate 决策 (R17/§19)

| R17 Gate | 决策 | 状态 |
|------|------|------|
| Gate 1 (= Stage 1 metadata 提取) | ✅ 复用 #86 PASS (commit 4ad7890) | 沿用 |
| Gate 2 (= Stage 2 SID+metadata) | ❌ FAIL 沿用 #87 (SID unique=256/9922) | 受控消融前提 |
| **Gate 3 (= Stage 3 T5 修复验证)** | ⏳ 进行中 | zero-equivalent init + warm-up + 同模型对比 |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP per Issue #93 spec | Gate 3 PASS 后才允许 |

## 4. 实施策略 (R11.5 自主决策)

- 复用 task383 #90 的 C1 (metadata embedding only) 配置 + sid_metadata 复用
- 修复 1: metadata_proj zero-equivalent init (nn.Linear weight=zero, bias=zero)
- 修复 2: metadata_scale warm-up (step 0=0 → step 200=1.0, 之后固定 1.0)
- 修复 3: 同模型三态对比 (on/off/shuffle), 复用同一 ckpt 推理三次
- 训练 500 steps (vs #90 100 steps) + T5-small 60.5M params
- GPU 0 立即启动 (R7 + R19 激进, GPU 0 空闲)

---

result: Issue #93 [方向C Gate3 metadata warm-up + 同模型 shuffle 修复] R22 + R19 立即开工 (GPU 0). 3 步: 1) zero-equivalent init + warm-up scale 0→1 + 500 steps; 2) 同模型 on/off/shuffle 三态对比; 3) Gate 3 验证 + commit + push + comment(含 hash) → close. ⏳ 进行中.