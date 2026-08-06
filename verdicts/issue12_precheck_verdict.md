---
type: precheck
issue: 12
status: "NO-GO"
created: 2026-08-02
tags:
  - misc
up: "[[index]]"
---
# Issue #12 Precheck — R18 4 维度对比

**Issue #12**: [方向A] 修复 P2 vocab 映射与 P4 合法SID约束失效, canary 非零复验, beam search 六指标产出

**Status**: precheck ✅ PASS (per #12 引用) / **实际 Stage 4 decode FAIL** (per d6bbdcc canary_argmax)

---

## R18 4 维度对比 (vs Issue #4, 同方向 lineage)

| 维度 | Issue #4 (已关 59299ca) | Issue #12 (新) | 差异 |
|---|---|---|---|
| **D1 spec 摘录** | 修 `compute_r_at_k` + 6 指标同值, canary R@10>0 | 修 P2 vocab_size=1025→451 映射 + P4 layer-wise SID mask 100% validity + canary R@10>0 + beam search K=20 让 6 指标差异化 | **D1 不同**: #4 修函数; #12 修协议层 + 加 beam search |
| **D2 实施核心** | 改 `taskA/stage4/taskA_stage4_resume.py:50,57` 函数 + main flow 传 [pred] 1-candidate list | 改 Stage 4 decode 用 `autoregressive_predict` (layer-wise mask) 替换 parallel argmax + 加 K=20 beam search + token↔SID 映射表 hash 一致性 | **D2 不同**: 实施路径完全不同 |
| **D3 Gate 1 失败机制** | `p[:k]==t[:k]` slice bug, k>4 退化为 exact match | P2: vocab_size 训练端 1025 vs 合法 SID 451 不匹配, Stage 4 端用 1025 表; P4: parallel argmax 无 layer-wise mask, 仅 25% token 落在合法 SID 区间 | **D3 不同**: 完全不同的失败机制 |
| **D4 引用文献** | recometrics (DOI:10.32614/cran.package.recometrics), mRHR (DOI:10.1007/978-3-319-44748-3_31) | recometrics + mRHR + Issue #47 κ-codebook 校准公式 + HG-Rec paper §3.2 layer-wise constraint | **D4 部分重叠**: 共用 R@K/NDCG 公式, #12 多了 κ-codebook 关联 |

**D1+D2+D3 都不同** → R18 强制: **必须做实验**, 不能凭"路径同构" NO-GO。

---

## 失败机制根因分析

### P2 (vocab 映射) FAIL 根因
- 训练端: T5 配置 `vocab_size=1025` (HG-Rec 原始 + T5 tokenizer 默认)
- 合法 SID 空间: 4 层去重 [1,64)∪[65,192)∪[193,448)∪{449} = 64+128+256+1 = 449 个 SID token
- 1025 = 449 (SID) + 576 (T5 原始 vocab 余量) — 训练端用全 1025 表, 但合法生成空间只占 449
- Stage 4 decode 不约束 → 模型可输出 1025 中任意 token → 多数越界
- **修复方向**: Stage 4 decode 必须用 `autoregressive_predict` (layer-wise mask 强制每步只在合法区间采样) + 训练端若需要同协议则保留 1025 vocab 但损失按 mask 后梯度

### P4 (合法 SID 约束) FAIL 根因
- 当前 `taskA_stage4_resume.py:161` 用 `preds = logits.argmax(dim=-1)` (parallel 4-position argmax)
- 没用 layer-wise mask
- 75% 输出 token 落在非法区间
- **修复方向**: 改用 `_m_lr.autoregressive_predict(model_wrapper, history_tensor, attention_mask, layer_ranges)` (Issue #190 修复版)

### Canary R@10=0 根因
- d6bbdcc canary_argmax 跑 100 samples 用 parallel argmax → 75% 越界 → 0 精确匹配
- Issue #4 close 时用的 canary_issue4_step1.py 跑 autoregressive → R@10=0.01 > 0
- **结论**: #4 关闭时 canary 是 PASS, 但实际 Stage 4 decode (parallel argmax) FAIL. **#4 close 不可采信, #12 重做正确**

### 6 指标恒等根因
- argmax 单 candidate 时 R@K≡R@1 数学正确推论 (无论函数怎么写, R@5=R@10=R@20=exact match rate)
- 真正让 6 指标差异化 = beam search 提供 K candidates
- **修复方向**: 实现 K=20 beam search (受 layer-wise SID mask 约束), 让 6 指标可微分化

---

## 实施计划 (4 步, 严格按序)

### Step 1: 修复 P2 vocab 映射
- 找到训练端 + Stage 4 端用的 token↔SID 映射表 (文件:行号)
- 验证训练端用 1025-vocab 表 + Stage 4 端用 1025-vocab 表 → 算 token↔SID hash
- 输出: `verdicts/issue12_p2_token_sid_mapping.json` (含 hash + 文件:行号)

### Step 2: 修复 P4 合法 SID 约束
- 修改 `taskA/stage4/taskA_stage4_resume.py` 改用 `_m_lr.autoregressive_predict` 替换 parallel argmax
- 验证 canary output **validity = 100%**
- py_compile + 跑 canary_argmax.py 验证

### Step 3: 重做 canary
- 同一 ckpt (long-run best_adapter.pt @ epoch 48) + 100 samples
- 用 `taskA_stage4_canary_issue4_step1.py` (autoregressive + layer-wise mask) 跑
- 验证: R@10 > 0 **且** protocol_audit 4 项全 PASS (P2 + P4 修复后)

### Step 4: beam search K=20
- 实现 K=20 beam search (4 步 + layer-wise mask 强制)
- 同一 ckpt + 100 samples, 跑 argmax vs beam 对照表
- 验证: 6 指标 (R@5/10/20, NDCG@5/10/20) 互不恒等
- 落 verdict + commit

### Step 5: 单 seed Task84 正式 Gate4 (待 owner 派工)
- 前提: Step 1-4 全 ✅
- 跑全 test set (24772 samples)
- 目标: R@10 > 0.1020 → Gate 4 PASS

---

## Precheck 状态

| 项 | 状态 |
|---|---|
| precheck (训练端 Stage 1+2+3) | ✅ PASS (per #12 引用 + gate1_evidence.json) |
| Stage 4 decode (parallel argmax) | ❌ FAIL (P2 vocab_size 不匹配 + P4 mask 未生效 + R@10=0) |
| 修复路径明确 | ✅ (用 autoregressive_predict 替换 parallel argmax) |
| Gate 1 状态 | ✅ PASS (sid_sha256=2dab29, emb_sha256=0fe7d949) |
| Gate 2 状态 | ✅ PASS (三层独立 nn.Parameter κ, 梯度非零) |
| Gate 3 状态 | ❌ conditional (P2/P4 FAIL) — 本 issue Step 1-2 修复 |
| Gate 4 状态 | ❌ blocked-no-canary (R@10=0) — 本 issue Step 3-4 修复 |

**结论**: 4 维度对比 + 失败机制 + 修复路径明确, **precheck 通过**. R19 立即启动 Step 1 修复 (下一个 tick).

---

## R29 满足 (本 precheck tick)

- R26 (≥1): 写本 precheck verdict `verdicts/issue12_precheck_verdict.md` 落盘
- R27 (≥1): glab issue list + nvidia-smi + git commit 5e8398b (housekeeping) + git push

**实际产物**:
- verdicts/issue12_precheck_verdict.md
- commit 5e8398b (housekeeping close 4 probes)
- closed issues: #8, #9, #10, #11
- 0 open: #12 (precheck 通过, Step 1-4 下 tick 启动)
