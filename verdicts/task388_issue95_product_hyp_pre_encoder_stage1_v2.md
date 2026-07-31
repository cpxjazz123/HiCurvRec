# Task #388 / Issue #95 [方向B Gate1] product 分量 HypPreEncoder 距离公式验证 — Gate 1 FAIL NO-GO

**日期**: 2026-07-31
**触发**: Issue #95 [方向B Gate1] product 分量 HypPreEncoder 距离公式验证
**前置**: Issue #92 Gate 1 FAIL (commit 0ef9eb9, step1 agree3=100%); Issue #43 HypPreEncoder 机制 PASS (task334)
**修复方案**: Product3CompHRQVAEWithHypPre 50 epoch + HypPreEncoder BEFORE encoder (c=0.74)
**任务**: 验证 step1 agree3 < 100%, final util ≥90%
**结果**: ❌ Gate 1 FAIL — **HypPreEncoder 不能修复 product 路径 step1 坍缩**

---

## 1. R17 Gate 决策 (R20 强制详细)

### Gate 1 (= Stage 1 product HypPreEncoder 验证): ❌ FAIL
- **关键数据**:
  - step1 agree3 = **100.00%** ❌ (目标 <100%, 跟 #92 FAIL 完全一致)
  - L0/L1/L2 final util = **4.69% / 0.78% / 0.39%** ❌ (目标 ≥90%, 跟 #92 FAIL 完全一致)
  - collapse step (component): L0=1, L1=1, L2=1 (第一步 optim.step 后立即坍缩)
  - collapse step (mixing): L0=1, L1=1, L2=1 (mixing-level collapse 同步)
  - ckpt saved (R12): products/task388_issue95_product_hyp_pre_encoder_stage1/product_hyp_pre_encoder_stage1_ckpt.pt
  - trace_records = 471
  - training time: ~120s (50 epoch @ GPU 1)
- **失败原因**:
  1. **HypPreEncoder (c=0.74) 接入 BEFORE encoder** 完全没改变 step1 agree3=100%
  2. final util 跟 #92 完全一致 (4.69%/0.78%/0.39%)
  3. mixing-level collapse step 仍 = 1 — 三 component 同步指向同一码字
  4. 跟 #92 联立: 输入端几何失真不是根因, per-component dist 公式仍是嫌疑
- **实施**: scripts/task388_issue95_product_hyp_pre_encoder_stage1.py (R4 py_compile OK)

### Gate 2/3/4: ⏸ STOP per Issue #95 spec
- 原因: Gate 1 FAIL, Issue #95 spec "本轮不得推进 Gate2/3/4, 除非本 issue 的 Gate1 产生外部可核验 PASS verdict"

---

## 2. 联立 #92 / #95

| Task | Issue | 修复方向 | step1 agree3 | final util |
|------|-------|---------|--------------|------------|
| task382 #89 | product 分离诊断 | (无修复) | step1 component + mixing 坍缩 | 诊断 |
| task385 #92 | kmeans_init + β=0 product 修复 | init 修复 | 100% | 4.69%/0.78%/0.39% |
| **task388 #95** | **product HypPreEncoder** | **架构修复** | **100%** | **4.69%/0.78%/0.39%** |

→ **联立结论**: product 路径所有"软修复 + 架构修复"全部失效
→ **真实根因必须在 per-component dist 公式本身**, 必须改 component-level distance 计算

---

## 3. 关键产物 (R21 强制具体 hash)

- **commit hash**: (pending push, see gh issue comment)
- **verdict**: verdicts/task388_issue95_product_hyp_pre_encoder_stage1_v2.md (本文件)
- **实施**: scripts/task388_issue95_product_hyp_pre_encoder_stage1.py
- **evidence**: products/task388_issue95_product_hyp_pre_encoder_stage1/evidence_package.json
- **trace**: products/task388_issue95_product_hyp_pre_encoder_stage1/trace_per_step.jsonl (471 records)
- **ckpt**: products/task388_issue95_product_hyp_pre_encoder_stage1/product_hyp_pre_encoder_stage1_ckpt.pt

---

result: Issue #95 [方向B Gate1 product HypPreEncoder 验证] Gate 1 FAIL NO-GO 收口. **跟 #92 联立: product 路径所有软+架构修复全部失效, 必须改 per-component dist 公式本身**. 实施 commit + push + issue comment + close 进行中.