# Task #389 / Issue #96 [方向C Gate2] SID/metadata 多样性修复 — Gate 2 FAIL NO-GO

**日期**: 2026-07-31
**触发**: Issue #96 [方向C Gate2] 先修复 SID/metadata 多样性再进 Stage 3
**前置**: Issue #93 Gate 3 FAIL (commit bd9698a, shuffle_diff=0); Issue #43 HypPreEncoder 机制 PASS (task334)
**修复方案**: FreeCurvHRQVAEWithHypPre Stage 1 50 epoch + Stage 2 Sinkhorn 推断 + 4-digit SID + metadata 重建
**任务**: 验证 SID unique ≥ 9500/9922, metadata per-field variance 非零
**结果**: ❌ Gate 2 FAIL — **SID unique=1/9922 (比 #87 256 更差!), HypPreEncoder 完全没改善 SID 多样性**

---

## 1. R17 Gate 决策 (R20 强制详细)

### Gate 1 (= Stage 1): ✅ 复用 #86 PASS (commit 4ad7890)
- 关键数据: 沿用 #86 ckpt 框架 + HypPreEncoder 接入, 50 epoch 训练完成

### Gate 2 (= Stage 2 SID/metadata 多样性): ❌ FAIL
- **关键数据**:
  - **SID unique count = 1/9922 = 0.01%** ❌ (目标 ≥ 9500/9922, 比 #87 FAIL 的 256 还要差!)
  - SID collision rate = **99.99%** ❌ (目标 ≤ 0.20, 比 #87 99.74% 还差)
  - L0/L1/L2 util = **1.56% / 0.78% / 0.39%** ❌ (目标 ≥ 90%, 三层全 FAIL)
  - metadata kappa var: **L0=0, L1=0, L2=0** ❌ (目标 > 0, 完全没有 per-item 区分)
  - metadata scale var: **L0=0, L1=0, L2=0** ❌
  - metadata conf var: L0=3.5e-5, L1=1.6e-5, L2=2.3e-5 (有微小数值, 仍 FAIL 因 < 1e-4)
  - ckpt saved (R12): products/task389_issue96_sid_metadata_diversity/sid_metadata_diversity_ckpt.pt
  - training time: ~80s (50 epoch @ GPU 2)
- **失败原因**:
  1. **SID unique = 1/9922** — 比 #87 256/9922 还差! 说明 HypPreEncoder 接入让 codebook **完全坍缩到一个码字**
  2. **metadata kappa/scale var = 0** — 因为只有一个码字被所有 9922 个 item 选中, per-item metadata 完全一致
  3. **跟 task387 #94 一致**: HypPreEncoder 接入后 step1 collapse 比 baseline 还严重
  4. 跟 task386 #93 联立: SID 坍缩根因更深, HypPreEncoder 不能解决
- **实施**: scripts/task389_issue96_sid_metadata_diversity.py (R4 py_compile OK)

### Gate 3/4: ⏸ STOP per Issue #96 spec
- 原因: Gate 2 FAIL, Issue #96 spec "不得继续执行 Gate3/4, 除非本 issue 产出外部可核验 Gate2 PASS evidence"

---

## 2. 联立 #87 / #93 / #96

| Task | Issue | SID unique | 根因 |
|------|-------|-----------|------|
| task380 #87 | SID metadata 对齐 (复用 #86) | **256/9922** | #86 ckpt collapse |
| task386 #93 | metadata warm-up 修复 | (沿用 #87) 256 | metadata uniform |
| **task389 #96** | **HypPreEncoder + κ-Stereographic 修复** | **1/9922 (更差)** | HypPreEncoder 没修复, 反而更坏 |

→ **联立结论**: HypPreEncoder 接入让 SID 坍缩从 256 unique **恶化到 1 unique**
→ **真实根因必须在 Stage 1 per-component dist 公式层面**, HypPreEncoder 不能解决

---

## 3. 关键产物 (R21 强制具体 hash)

- **commit hash**: (pending push, see gh issue comment)
- **verdict**: verdicts/task389_issue96_sid_metadata_diversity_v2.md (本文件)
- **实施**: scripts/task389_issue96_sid_metadata_diversity.py
- **evidence**: products/task389_issue96_sid_metadata_diversity/evidence_package.json
- **sid**: products/task389_issue96_sid_metadata_diversity/sid_metadata_diversity_sid.npy
- **metadata**: products/task389_issue96_sid_metadata_diversity/sid_metadata_diversity_metadata.json
- **ckpt**: products/task389_issue96_sid_metadata_diversity/sid_metadata_diversity_ckpt.pt

---

result: Issue #96 [方向C Gate2 SID/metadata 多样性修复] Gate 2 FAIL NO-GO 收口. **关键新发现: HypPreEncoder 接入让 SID unique 从 #87 256 恶化到 1, metadata 仍 uniform (var=0); 必须改 per-component dist 公式本身**. 实施 commit + push + issue comment + close 进行中.