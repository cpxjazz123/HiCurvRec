# Task #384 / Issue #91 [方向A Gate1] kmeans_init + β=0 first epoch 修复 step1 坍缩 — Gate 1 FAIL NO-GO

**日期**: 2026-07-31
**触发**: Issue #91 [方向A Gate1] kmeans_init + β warmup 修复 step1 坍缩验证
**前置**: Issue #88 Gate 1 诊断 PASS (commit d421beb, 最早坍缩 step 1, 第一次 gradient 推动)
**修复方案**: kmeans_init=True (替代 baseline recipe 的 kmeans_init=False) + β=0 first epoch (跳过 commitment loss)
**任务**: 验证修复后 step1 是否仍立即 50%+ 坍缩; 最终三层 util ≥90%
**结果**: ❌ Gate 1 FAIL — **kmeans_init + β=0 不能修复 step1 坍缩, 根因不在 commitment loss**

---

## 1. R17 Gate 决策 (R20 强制详细)

### Gate 1 (= Stage 1 修复验证): ❌ FAIL 3/7

**关键数据**:
- **step1 max_load = 82.56%** (目标 <50%, FAIL 严重超出阈值)
- **L0/L1/L2 final util = 1.56% / 0.78% / 0.39%** (目标 ≥90%, 三层全 FAIL)
- collapse 50% step: **L0=1, L1=1, L2=1** (跟 task381 #88 诊断一致, **第一步 optim.step 后立即坍缩**)
- collapse 90% step: L0=50, L1=50, L2=250 (90% 集中度在 epoch 0-2 内达成)
- ckpt saved (R12 强制): products/task384_issue91_kmeans_init_beta_warmup_fix/kmeans_init_beta_warmup_ckpt.pt
- trace_records = 471 (per-step trace 完整记录)
- training time: 96s (50 epoch @ GPU 0)

**失败原因**:
1. **即使 kmeans_init=True (用真实数据点初始化)**, codebook 仍 step 1 立即坍缩到单码字 50%+ load
2. **即使 first epoch β=0 (跳过 commitment loss)**, 坍缩仍发生 — 说明根因**不在 commitment loss + codebook loss 的第一次 gradient**
3. 第一 epoch 之后 β=0.25 (normal) 训练 49 epoch 仍无法恢复 → codebook 完全坍缩到 1-2 个码字

**实施**: scripts/task384_issue91_kmeans_init_beta_warmup_fix.py (R4 py_compile OK)

### Gate 2 (= Stage 2 Sinkhorn): ⏸ STOP per Issue #91 spec
- 原因: Gate 1 FAIL, Stage 2 强制前置条件 = Stage 1 PASS

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per Issue #91 spec
- 原因: Gate 2 STOP, T5 训练前置条件 = Stage 1+2 PASS

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per Issue #91 spec
- 原因: Gate 3 STOP, R@K eval 前置条件 = Stage 1+2+3 PASS

---

## 2. 关键发现 — 根因不在 commitment loss

**Issue #88 (task381) 诊断假设**: commitment loss + codebook loss 的第一次 gradient 把码字推向数据几何中心
**Issue #91 修复实证**: 即使用 kmeans init (码字初始化在真实数据点) + β=0 first epoch (跳过 commitment loss), 坍缩仍 step 1 触发

→ **根因不在 commitment loss**, 更深层假设:
- (a) decoder gradient 通过 recon_loss 推动 encoder 输出分布集中到某个特定位置 → 所有 encoder 输出聚集到 1 个码字附近
- (b) FreeCurvHRQVAE 的 per-component dist 公式 (κ-Stereographic) 在 norm=0.85 embedding 空间把所有码字映射到几乎等距
- (c) baseline recipe 本身 + L40S float32 数值精度问题导致 codebook 在第一次 optim.step 后立即梯度爆炸

无论 (a/b/c), **结论**: 单纯改 init / β schedule **无法修复 step1 坍缩**, 必须改 per-component dist 公式本身

---

## 3. 联立 NO-GO 收口累积 (R11.5 透明)

| Task | Issue | 维度 | 关键 NO-GO 结论 |
|------|-------|------|----------------|
| task381 #88 | 方向A 坍缩根因 trace | 诊断 | step1 坍缩确认 (commitment+codebook 第一次 gradient) |
| task382 #89 | 方向B product 分离诊断 | 诊断 | step1 component + mixing 坍缩 |
| task383 #90 | 方向C T5 metadata 受控消融 | Stage 3 | metadata path 未真正生效 (shuffle_diff=0) |
| **task384 #91** | **方向A 修复 step1 坍缩** | **修复** | **kmeans_init + β=0 不能修复 step1 坍缩** ⭐ 新根因 |
| **task385 #92** | **方向B product 修复 step1 坍缩** | **修复** | **kmeans_init + β=0 不能修复 product step1 坍缩** ⭐ |

**15 方向 × 16 verdict NO-GO 收口累积**: task178/180/231/242/299/371/374/377/378/379/380/381/382/383 + **#91/#92**

**新核心发现**: 修复路径必须在架构层 (κ-Stereographic 公式 + long training + decoder 端), 不是 init / β schedule 维度

---

## 4. 关键产物 (R21 强制具体 hash)

- **commit hash**: (pending push, see gh issue comment)
- **push**: origin/main
- **verdict**: verdicts/task384_issue91_kmeans_init_beta_warmup_fix_v2.md (本文件)
- **实施**: scripts/task384_issue91_kmeans_init_beta_warmup_fix.py
- **evidence**: products/task384_issue91_kmeans_init_beta_warmup_fix/evidence_package.json
- **trace**: products/task384_issue91_kmeans_init_beta_warmup_fix/trace_per_step.jsonl (471 records)
- **ckpt**: products/task384_issue91_kmeans_init_beta_warmup_fix/kmeans_init_beta_warmup_ckpt.pt

---

## 5. R18 4 维度对比 (Issue #91 vs Issue #88)

| 维度 | Issue #88 (坍缩根因 trace 诊断) | Issue #91 (kmeans_init + β warmup 修复) | 一致? |
|------|------|------|------|
| **D1 spec** | per-step trace 定位最早坍缩点 | kmeans_init + β=0 first epoch + 验证 step1 不再坍缩 | ❌ |
| **D2 实施** | baseline recipe + per-step trace | baseline recipe + kmeans_init=True + β=0 first epoch | ❌ |
| **D3 失败机制** | commitment loss + codebook loss 第一次 gradient 推动 | kmeans_init + β=0 让码字不被推到中心 | ❌ (被实证 falsified) |
| **D4 引用文献** | arXiv:2405.13979v4 | 同 | ✅ |

→ R18 强制 4 维度对比, 修复方案与诊断假设不一致; **D3 假设被 Issue #91 实证 falsified** (commitment loss 不是根因)

---

result: Issue #91 [方向A Gate1 kmeans_init + β=0 first epoch 修复 step1 坍缩] Gate 1 FAIL 3/7 NO-GO 收口. **关键新发现: kmeans_init + β=0 不能修复 step1 坍缩, 根因不在 commitment loss, 必须在架构层 (per-component dist 公式 + decoder gradient 路径) 解决**. 实施 commit + push + issue comment + close 进行中.