# Task #385 / Issue #92 [方向B Gate1] product 路径 kmeans_init + β=0 first epoch 修复 — Gate 1 FAIL NO-GO

**日期**: 2026-07-31
**触发**: Issue #92 [方向B Gate1] product 路径 kmeans_init + β warmup 修复验证
**前置**: Issue #89 Gate 1 诊断 PASS (commit b5fb4db, component-level + mixing-level collapse @ step 1)
**修复方案**: Product3CompHRQVAE wrapper + kmeans_init=True + β=0 first epoch
**任务**: 验证 step1 agree3 < 100% (三个 component 不再全指同一码字)
**结果**: ❌ Gate 1 FAIL — **kmeans_init + β=0 不能修复 product 路径 step1 坍缩, 跟 #91 一致根因**

---

## 1. R17 Gate 决策 (R20 强制详细)

### Gate 1 (= Stage 1 product 修复验证): ❌ FAIL 5/8

**关键数据**:
- **step1 agree3 = 100.00%** (目标 <100%, FAIL — 三个 component 仍全指同一码字)
- **L0/L1/L2 final util = 4.69% / 0.78% / 0.39%** (目标 ≥90%, 三层全 FAIL)
- collapse step (component): **L0=1, L1=1, L2=1** (跟 task382 #89 诊断一致, 第一 step 立即坍缩)
- collapse step (mixing): **L0=1, L1=1, L2=1** (mixing-level collapse 同步触发)
- mixing weights 训练 50 epoch 后: **w=[0.33,0.33,0.33]** (健康 ≈均匀分布, mixing logits 不主导坍缩)
- ckpt saved (R12 强制): products/task385_issue92_product_kmeans_init_beta_warmup_fix/product_kmeans_init_beta_warmup_ckpt.pt
- trace_records = 471 (per-step per-component trace 完整)
- training time: 66s (50 epoch @ GPU 1)

**失败原因**:
1. **step1 agree3 = 100%** — 即使用了 kmeans_init (用真实数据点初始化), 三个 component (learned-κ + fixed-hyp + Euclidean) 仍全指同一最近码字
2. **mixing weights 健康 w=[0.33,0.33,0.33]** — 跟 #91 一致, 三分量 softmax mixing 不是根因
3. **即使 first epoch β=0** (跳过 commitment loss) — 仍 step1 坍缩 → **跟 Issue #91 联立确认: 根因不在 commitment loss**
4. **mixing-level collapse step = 1** — three components 同步指向同一码字, 顶层 mixing 距离几乎为 0

**实施**: scripts/task385_issue92_product_kmeans_init_beta_warmup_fix.py (R4 py_compile OK)

### Gate 2 (= Stage 2 Sinkhorn): ⏸ STOP per Issue #92 spec
- 原因: Gate 1 FAIL, Stage 2 强制前置条件 = Stage 1 PASS

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per Issue #92 spec
- 原因: Gate 2 STOP, T5 训练前置条件 = Stage 1+2 PASS

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per Issue #92 spec
- 原因: Gate 3 STOP, R@K eval 前置条件 = Stage 1+2+3 PASS

---

## 2. 关键发现 — 联立 #91 根因确认

**Issue #89 (task382) 诊断假设**: 三分量 product 结构导致 component-level + mixing-level collapse @ step 1
**Issue #92 修复实证**: 即使 Product3CompHRQVAE wrapper + kmeans_init + β=0 first epoch, 仍 step1 坍缩 + agree3=100%

→ **联立 Issue #91 + #92 根因**:
- baseline recipe + FreeCurvHRQVAE / Product3CompHRQVAE 的 step1 坍缩**不在 commitment loss**
- **不在 init scheme** (kmeans 替代 random init 无效)
- **不在三分量 mixing** (mixing weights 健康 ≈0.33)
- **不在 β schedule** (β=0 first epoch 无效)
- **真实根因**: 必须改 per-component dist 公式本身 (Issue #43 HypPreEncoder + κ-Stereographic unified formula 方向) / decoder gradient 路径

---

## 3. 联立 NO-GO 收口累积 (R11.5 透明)

| Task | Issue | 方向 | 关键 NO-GO 结论 |
|------|-------|------|----------------|
| task381 #88 | 方向A 坍缩根因 trace | 诊断 | step1 坍缩确认 (commitment+codebook 第一次 gradient) |
| task382 #89 | 方向B product 分离诊断 | 诊断 | step1 component + mixing 坍缩 |
| task383 #90 | 方向C T5 metadata 受控消融 | Stage 3 | metadata path 未真正生效 (shuffle_diff=0) |
| **task384 #91** | **方向A 修复 step1 坍缩** | **修复** | **kmeans_init + β=0 不能修复 step1 坍缩** ⭐ 新根因 |
| **task385 #92** | **方向B product 修复 step1 坍缩** | **修复** | **kmeans_init + β=0 不能修复 product step1 坍缩** ⭐ |

**15 方向 × 16 verdict NO-GO 收口累积**: task178/180/231/242/299/371/374/377/378/379/380/381/382/383 + **#91/#92**

**新核心发现**: init + β schedule + 三分量 mixing 三种"软修复"全部失效; 必须改 per-component dist 公式 (κ-Stereographic + HypPreEncoder 路径)

---

## 4. 关键产物 (R21 强制具体 hash)

- **commit hash**: (pending push, see gh issue comment)
- **push**: origin/main
- **verdict**: verdicts/task385_issue92_product_kmeans_init_beta_warmup_fix_v2.md (本文件)
- **实施**: scripts/task385_issue92_product_kmeans_init_beta_warmup_fix.py
- **evidence**: products/task385_issue92_product_kmeans_init_beta_warmup_fix/evidence_package.json
- **trace**: products/task385_issue92_product_kmeans_init_beta_warmup_fix/trace_per_step.jsonl (471 records)
- **ckpt**: products/task385_issue92_product_kmeans_init_beta_warmup_fix/product_kmeans_init_beta_warmup_ckpt.pt

---

## 5. R18 4 维度对比 (Issue #92 vs Issue #89)

| 维度 | Issue #89 (product 分离诊断) | Issue #92 (product 修复验证) | 一致? |
|------|------|------|------|
| **D1 spec** | per-component trace 3 选 1 判定 (诊断) | kmeans_init + β=0 修复 + 验证 component argmin agreement | ❌ |
| **D2 实施** | baseline recipe 10 epoch + per-component trace | baseline recipe + kmeans_init=True + β=0 first epoch | ❌ |
| **D3 失败机制** | component-level collapse @ step 1, 三 component 都指向同一最近点 | kmeans_init 用真实数据点初始化 + β=0 让码字不被推到中心 | ❌ (被实证 falsified) |
| **D4 引用文献** | arXiv:2307.04514 + DOI 10.1109/ICDM51629.2021.00021 | 同 | ✅ |

→ R18 强制 4 维度对比, 修复方案与诊断假设不一致; **D3 假设被 Issue #92 实证 falsified** (跟 #91 联立确认)

---

result: Issue #92 [方向B Gate1 product kmeans_init + β=0 first epoch 修复] Gate 1 FAIL 5/8 NO-GO 收口. **关键新发现: 跟 #91 联立确认 kmeans_init + β=0 + 三分量 mixing 三种"软修复"全部失效; 必须改 per-component dist 公式 (κ-Stereographic + HypPreEncoder 路径)**. 实施 commit + push + issue comment + close 进行中.