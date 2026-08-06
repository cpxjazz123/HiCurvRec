---
type: precheck
issue: 13
status: "FAIL"
created: 2026-08-02
tags:
  - misc
up: "[[index]]"
---
# Issue #13 Precheck — R18 4 维度对比

**Issue #13**: [方向B] taskB 协议 audit 补做与 P2 P4 修复, canary 非零复验, 混合分量诊断, beam search 六指标产出

**Status**: precheck ✅ PASS (per #13 引用) / **实际 Stage 4 decode FAIL** (per Issue #5 不可采信 + 方向A 同款 d6bbdcc canary_argmax FAIL 推论)

---

## R18 4 维度对比 (vs Issue #12, 跨方向平行, 跟方向A 高度同构)

| 维度 | Issue #12 (方向A, 已关 380228f) | Issue #13 (方向B, 新) | 差异 |
|---|---|---|---|
| **D1 spec 摘录** | 修 P2/P4 + canary + beam search K=20 | 修 P2/P4 + canary + 混合分量诊断 + beam search K=20 | **D1 差异**: #13 多了 Step 1 audit + Step 5 混合分量诊断 (taskB 特异) |
| **D2 实施核心** | 改 taskA_stage4_resume.py 用 autoregressive_predict (HG_Rec_with_BoundedAdapter) | 改 taskB_stage4_resume.py 用 autoregressive_predict (HG_Rec_with_BoundedMixedAdapter, 含 weight_mlp mixing) | **D2 差异**: wrapper class 不同 (BoundedMixedAdapter 加 mixing weights), decode 路径同构 |
| **D3 Gate 1 失败机制** | P2 (vocab_size=1025 vs 451) + P4 (parallel argmax 无 layer-wise mask, validity 25%) + 6 指标恒等 | 平行: P2 + P4 + 6 指标恒等. 方向A 已实测 FAIL 推论方向B 同款实现 (共用 autoregressive_predict + get_layer_ranges) 也 FAIL | **D3 同**: 失败机制同构 |
| **D4 引用文献** | recometrics + mRHR + Issue #47 κ-codebook 公式 + HG-Rec paper §3.2 | recometrics + mRHR + **Riemannian curvature tensor of almost-product manifold** (DOI:10.21136/cmj.1986.102068) + Issue #47 | **D4 差异**: #13 多了乘积流形曲率 (混合双曲+欧氏分量的数学背景) |

**D1+D2+D4 都不同** (D3 同). R18 强制: **必须做实验** (D3 同不豁免, 因 D1/D2/D4 不同 + 实施路径独立).

---

## 失败机制根因分析 (跟方向A 同构, 但 taskB 缺实测)

### P2 (vocab 映射) FAIL 根因 (类比方向A)
- 训练端: T5 配置 `vocab_size=1025` (同 taskA)
- 合法 SID 空间: 449 token (同 taskA)
- taskB 训练端用全 1025 表, 但合法生成空间只占 449
- Stage 4 decode 不约束 → 模型可输出 1025 中任意 token → 多数越界

### P4 (合法 SID 约束) FAIL 根因 (类比方向A)
- 当前 `taskB_stage4_resume.py` (推测, 同 taskA 模式) 用 parallel argmax
- 没用 layer-wise mask
- **taskB 缺实测 canary_argmax audit** (per #13 Step 1 要求补做)
- 推论 (方向A 25% validity 推论): taskB 也是 ~25% validity

### Canary R@10=0 根因 (跟方向A 同源)
- 方向A canary_argmax (d6bbdcc) R@10=0.0 + P2/P4 FAIL
- 方向B 缺独立 canary audit, 推论同款 FAIL

### 6 指标恒等根因
- argmax 单 candidate 时 R@K≡R@1 (跟方向A 同)

### 混合分量诊断缺位 (taskB 特异)
- taskB wrapper 是 `HG_Rec_with_BoundedMixedAdapter` (含 weight_mlp 3-way mixing: 双曲/欧氏/混合)
- 方向B 定义: 三层独立 learnable κ + 固定双曲/欧氏分量 + 可学习 mixing weights
- 需要诊断: 每层 (κ_l, alpha/beta/gamma_l) 梯度 + 范数占比, 证明分量进 forward 流
- 严禁: 退化为 fixed weights / 删欧氏分量 / 加硬性下限

---

## 实施计划 (6 步, 严格按序)

### Step 1: 对 taskB 补做与方向A同款 protocol audit
- 写 `taskB/stage4/taskB_stage4_canary_argmax.py` (跟 taskA 镜像, 但用 taskB wrapper)
- 跑 100 samples, 产出 `taskB/stage4/taskB_stage4_canary_argmax/canary_verdict.json` 含 P1/P2/P3/P4 + sid_sha_match + metrics + decision
- 验证 taskB P4 真实状态 (不沿用旧文字断言)

### Step 2: P2 vocab 映射 hash (类比 #12 Step 1)
- 训练端 + Stage 4 端用 `get_layer_ranges` (taskB long-run script)
- layer_ranges 同 taskA `[(1, 64), (65, 192), (193, 448), (449, 449)]`
- mapping hash = aacb3085416491cd (同 taskA, 共享 CODEBOOK_SIZE)
- 落 `verdicts/issue13_p2_token_sid_mapping.json`

### Step 3: P4 修复 (类比 #12 Step 2)
- 改 `taskB/stage4/taskB_stage4_resume.py` main flow 用 `_m_lr.autoregressive_predict` 替换 parallel argmax
- taskB long-run script 提供 autoregressive_predict (跟 taskA 同源)
- py_compile 验证

### Step 4: 重做 canary (类比 #12 Step 3)
- 用 taskB long-run best_adapter.pt @ epoch 50 + 100 samples
- 验证 R@10 > 0 + validity = 100%
- 落 `verdicts/issue13_step3_step4_canary_result.json`

### Step 5: 混合分量可审计诊断 (taskB 特异)
- 写 `taskB/stage4/taskB_stage4_mixing_diagnostic.py`
- 短程 (不训练), 加载 taskB long-run best_adapter.pt
- 记录每层 (κ_l, alpha/beta/gamma_l):
  - dL/dκ_l (梯度)
  - dL/dmix_l (mixing 梯度)
  - 有限差分敏感度 (κ + 0.01 vs κ - 0.01, 比较 loss 差)
  - 固定双曲分量范数占比 ‖hyp‖² / (‖hyp‖² + ‖euc‖²)
  - 固定欧氏分量范数占比 ‖euc‖² / (‖hyp‖² + ‖euc‖²)
- 证明三者进 forward 流 (梯度非零 + 范数 > 0)
- 落 `verdicts/issue13_step5_mixing_diagnostic.json`

### Step 6: beam search K=20 (类比 #12 Step 4)
- 写 `taskB/stage4/taskB_stage4_beam_search_issue13_step6.py` (跟 taskA 镜像)
- per-sample 4-step beam + layer-wise mask
- 验证 6 指标互不恒等 + R@K 单调非降
- 落 `verdicts/issue13_step6_beam_search_result.json`

### Step 7: 单 seed Task84 全 eval (待 owner 派工, 不在本 issue 范围)

---

## Precheck 状态

| 项 | 状态 |
|---|---|
| precheck (训练端 Stage 1+2+3) | ✅ PASS (per #13 引用: taskB_stage2_weighted_mixed.py L111/122/181/262 三层独立 κ + mixing weights 梯度非零) |
| Stage 4 decode (parallel argmax) | ❌ FAIL 推论 (方向A 同款实现已 FAIL, taskB 缺独立 audit 待 Step 1) |
| 修复路径明确 | ✅ (跟 #12 平行, autoregressive_predict 替换 parallel argmax) |
| Gate 1 状态 | ✅ PASS (gate1_evidence.json, sid_sha256=2dab29, emb_sha256=0fe7d949 — taskA/taskB 共享) |
| Gate 2 状态 | ✅ PASS (三层独立 nn.Parameter κ + mixing weights 梯度非零) |
| Gate 3 状态 | ❌ conditional (P2 FAIL 推论, P4 未实测) — 本 issue Step 1-3 修复 |
| Gate 4 状态 | ❌ blocked-no-canary (R@10=0 推论) — 本 issue Step 4-6 修复 |

**结论**: 4 维度对比 + 失败机制 + 修复路径明确, **precheck 通过**. R19 立即启动 Step 1 audit (本 tick), Step 2-6 后续 tick.

---

## 跟 #12 实施同步性

| 项 | #12 (taskA) | #13 (taskB) |
|---|---|---|
| 实施时间 | 上一 tick 已完成 | 本 tick 启动 |
| Step 1 audit | canary_argmax 早存在 (d6bbdcc) | **缺**, 需补做 |
| P2 mapping hash | aacb3085 (已落) | 同 (共享 layer_ranges) |
| P4 修复 | autoregressive_predict 已改 | 同 (平行改) |
| canary 验证 | R@10=0.01 + validity 100% | 同 (待跑) |
| beam search K=20 | 6 指标互不恒等 PASS | 同 (待跑) |
| 混合分量诊断 | N/A (方向A 不涉及) | **taskB 特异, Step 5 必做** |

**关键差异**: #13 多了 Step 5 混合分量诊断 (taskB 三层独立 κ + 固定双曲/欧氏分量 + 可学习 mixing weights), 是 taskB 唯一性约束. 严禁退化为 fixed weights / 删欧氏分量 / 加硬性下限.
