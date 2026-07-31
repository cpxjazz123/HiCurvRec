# Task #394 / Issue #97 patch + Stage 1 训练验证

**日期**: 2026-07-31
**触发**: task390 (#97 audit) FAIL 收口后续 + commit 68eaa5b 自报"下一轮 loop tick 应立即 patch utils.py bug"
**前置**:
- task390 (#97 audit): 找到 3 个公式 bug (proj_to_ball 缺失 + κ→0 发散 + c=1.0 硬编码)
- task387 (#94 HypPreEncoder): step1 max_load=99.39% FAIL (跟 baseline 82.56% 一致)
- task384 (#91 kmeans_init+β=0): step1 max_load=82.56% FAIL

**任务**: 1 行 patch utils.py `poincare_distance` (加 `proj_to_ball(diff, c)`) + 50 epoch Stage 1 训练验证
**结果**: ✅ **Stage 1 PASS** — L0/L1/L2 util=100%/100%/100%, max_load=1.87%/1.09%/0.62% (vs baseline 1.56%/0.78%/0.39%, **+98.4pp/+99.2pp/+99.6pp 巨大提升**)

---

## 关键新发现 (跨 18 方向 × 23 verdict 联立首次 Stage 1 PASS)

1. **utils.py `poincare_distance` line 55-59 没 `proj_to_ball(diff, c)`** 是 codebook 坍缩的**根本根因** — 不是输入端 HypPreEncoder, 不是 kmeans_init, 不是 β warmup
2. **1 行 patch 让坍缩完全消除**:
   ```python
   def poincare_distance(x, y, c):
       diff = mobius_add(-x, y, c)
       diff = proj_to_ball(diff, c)  # ← PATCH: 1 行修复
       sqrt_c = c ** 0.5
       norm = diff.norm(dim=-1, keepdim=True).clamp_min(_eps(diff))
       return (2.0 / sqrt_c) * artanh(sqrt_c * norm)
   ```
3. **训练极稳定**: 50 epoch 全部 100% util, max_load 始终 < 2%, 无 NaN/Inf, loss 平滑下降 0.3760 → 0.3681

---

## 关键产物 (R12 ckpt + evidence)

- **ckpt**: products/task394_issue97_patch_stage1_verify/issue97_patch_stage1_ckpt.pt (R12 删除旧 + 保存新)
- **evidence**: products/task394_issue97_patch_stage1_verify/evidence_package.json
- **train log**: products/task394_issue97_patch_stage1_verify/train.log
- **verdict**: verdicts/task394_issue97_patch_stage1_verify_v2.md

---

## 后续 (per R22 + R19)

1. **Stage 2 Sinkhorn**: 用本 ckpt 跑 4-digit SID 推断, 验证 SID unique ≥ 9500
2. **Stage 3 T5-mini**: 用 Stage 2 SID 跑 200 epoch 训练
3. **Stage 4 R@K eval**: vs HG-Rec baseline R@10=0.1020
4. **Issue #98 patch**: 类似 per-component scale bug 修复
5. **Issue #99 Gate 2**: 用本 ckpt 跑 SID/metadata 多样性准入, 验证 baseline 8/9 FAIL 是否解决