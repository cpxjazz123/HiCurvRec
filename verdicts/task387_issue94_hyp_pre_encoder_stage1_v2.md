# Task #387 / Issue #94 [方向A Gate1] HypPreEncoder + κ-Stereographic 距离公式准入验证 — Gate 1 FAIL NO-GO

**日期**: 2026-07-31
**触发**: Issue #94 [方向A Gate1] HypPreEncoder κ-Stereographic 距离公式准入验证
**前置**: Issue #91 Gate 1 FAIL (commit 0ef9eb9, step1 max_load=82.56%); Issue #43 HypPreEncoder 机制 PASS (task334)
**修复方案**: FreeCurvHRQVAEWithHypPre 50 epoch + kmeans_init=True (复刻 task334 wrapper, base=FreeCurvHRQVAE, c=0.74 from Issue #43 Task #70)
**任务**: 验证 step1 不再 collapse_50%, final util ≥90%
**结果**: ❌ Gate 1 FAIL — **HypPreEncoder 不能修复 step1 坍缩, step1 max_load 反而恶化到 99.39%**

---

## 1. R17 Gate 决策 (R20 强制详细)

### Gate 1 (= Stage 1 HypPreEncoder 准入验证): ❌ FAIL
- **关键数据**:
  - step1 max_load = **99.39%** ❌ (目标 <50%, 比 #91 82.56% **更差**)
  - L0/L1/L2 final util = **1.56% / 0.78% / 0.39%** ❌ (目标 ≥90%, 三层全 FAIL)
  - collapse_50% step: L0=50, L1=1, L2=1 (L1/L2 第一 step 立即坍缩)
  - ckpt saved (R12): products/task387_issue94_hyp_pre_encoder_stage1/hyp_pre_encoder_stage1_ckpt.pt
  - trace_records = 471
  - training time: ~150s (50 epoch @ GPU 0)
- **失败原因**:
  1. **HypPreEncoder (c=0.74 expmap0) 接入 BEFORE encoder** 反而让 step1 max_load **从 #91 82.56% 恶化到 99.39%**
  2. final util 跟 #91 完全一样 (1.56%/0.78%/0.39%) — HypPreEncoder 完全没有挽救 codebook 坍缩
  3. 说明根因**不在输入端几何失真** (Issue #94 假设), 也不在 commitment loss (#91 假设), 也不在 init (#91 修复无效)
  4. **真实根因必须在更架构层**: per-component dist 公式 / decoder gradient 路径 / 上游 RQ-VAE bug
- **实施**: scripts/task387_issue94_hyp_pre_encoder_stage1.py (R4 py_compile OK)

### Gate 2/3/4: ⏸ STOP per Issue #94 spec
- 原因: Gate 1 FAIL, Issue #94 spec "本轮不得推进 Gate2/3/4, 除非本 issue 的 Gate1 产生外部可核验 PASS verdict"

---

## 2. 关键发现 — 联立 #91 / #94

| Task | Issue | 修复方向 | step1 max_load | final util |
|------|-------|---------|----------------|------------|
| task381 #88 | 方向A 坍缩根因 trace 诊断 | (无修复) | step1 collapse | 诊断 |
| task384 #91 | kmeans_init + β=0 first epoch | init 修复 | 82.56% | 1.56%/0.78%/0.39% |
| **task387 #94** | **HypPreEncoder + κ-Stereographic** | **架构层准入** | **99.39% (更差)** | **1.56%/0.78%/0.39%** |

→ **联立结论**: init / HypPreEncoder 两种"软修复"全部失效, step1 坍缩比 task384 更严重
→ **真实根因必须改 per-component dist 公式本身** (Issue #44 Gate 1 失败的 unified formula 路径)

---

## 3. 联立 NO-GO 收口累积 (R11.5 透明)

| Task | Issue | 维度 | 关键 NO-GO 结论 |
|------|-------|------|----------------|
| task381 #88 | 方向A 坍缩根因 trace | 诊断 | step1 坍缩确认 |
| task382 #89 | 方向B product 分离诊断 | 诊断 | step1 component + mixing 坍缩 |
| task383 #90 | 方向C T5 metadata 受控消融 | Stage 3 | shuffle_diff=0 |
| task384 #91 | 方向A init + β=0 修复 | 修复 | step1 max_load=82.56% 仍 FAIL |
| task385 #92 | 方向B product init + β=0 修复 | 修复 | step1 agree3=100% 仍 FAIL |
| task386 #93 | 方向C metadata warm-up 修复 | 修复 | shuffle_diff=0 (新根因: metadata uniform) |
| **task387 #94** | **方向A HypPreEncoder 架构修复** | **架构修复** | **step1 max_load=99.39% 仍 FAIL (更差)** ⭐ |
| task388 #95 | 方向B product HypPreEncoder | 架构修复 | step1 agree3=100% 仍 FAIL (待 commit) |
| task389 #96 | 方向C SID/metadata 多样性 | Gate 2 修复 | SID unique=1/9922 仍 FAIL (待 commit) |

**15 方向 × 20 verdict NO-GO 收口累积**: task178/180/231/242/299/371/374/377/378/379/380/381/382/383/384/385/386 + **#94/#95/#96**

**新核心发现**: 
- 软修复 (init / β schedule) + 架构修复 (HypPreEncoder) 全部失效
- **真正根因必须在 per-component dist 公式层面** — Issue #44 Gate 1 (unified κ-Stereographic formula) 路径是唯一理论可能, 但 task335 实证 NO-GO (T1/T2/T3 数学 FAIL)

---

## 4. 关键产物 (R21 强制具体 hash)

- **commit hash**: (pending push, see gh issue comment)
- **verdict**: verdicts/task387_issue94_hyp_pre_encoder_stage1_v2.md (本文件)
- **实施**: scripts/task387_issue94_hyp_pre_encoder_stage1.py
- **evidence**: products/task387_issue94_hyp_pre_encoder_stage1/evidence_package.json
- **trace**: products/task387_issue94_hyp_pre_encoder_stage1/trace_per_step.jsonl (471 records)
- **ckpt**: products/task387_issue94_hyp_pre_encoder_stage1/hyp_pre_encoder_stage1_ckpt.pt

---

result: Issue #94 [方向A Gate1 HypPreEncoder + κ-Stereographic 准入验证] Gate 1 FAIL NO-GO 收口. **关键新发现: HypPreEncoder 不能修复 step1 坍缩, 反而恶化 (max_load 82.56%→99.39%); 真实根因必须在 per-component dist 公式层面 (Issue #44 unified formula 路径待修)**. 实施 commit + push + issue comment + close 进行中.