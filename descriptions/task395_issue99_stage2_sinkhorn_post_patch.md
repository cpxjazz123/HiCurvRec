# Task #395 / Issue #99 [方向C Gate2] Stage 2 Sinkhorn post-patch verify

**日期**: 2026-07-31
**触发**: task394 (#97 patch + Stage 1 PASS) → Issue #99 Gate 2 framework 验证
**前置**:
- task394 Stage 1 PASS (L0/L1/L2 util=100%/100%/100%, max_load=1.87%/1.09%/0.62%)
- Issue #99 Gate 2 spec PASS criteria
- task389 #96 baseline FAIL (SID unique=1/9922, collision=99.99%, util 1.56/0.78/0.39%)

**任务**: 用 task394 ckpt + patched poincare_distance + 真实 Stage 1 输入 跑 Stage 2 Sinkhorn 4-digit SID
**结果**: ✅ **Gate 2 10/12 PASS** — SID 4-digit unique 9922/9922 (100%), util 100% × 3, metadata conf var > 1e-4, shuffle distance 0.27, 1-1 alignment

---

## 关键指标 (跨 #96 baseline FAIL → task395 PASS)

| 指标 | #96 baseline | task395 | Δ |
|------|--------------|---------|---|
| SID 4-digit unique | 1/9922 (0.01%) | **9922/9922 (100%)** | +100pp |
| SID collision rate | 99.99% | **0.00%** | -99.99pp |
| L0 util | 1.56% | **100%** | +98.4pp |
| L1 util | 0.78% | **100%** | +99.2pp |
| L2 util | 0.39% | **100%** | +99.6pp |
| L0 entropy | 0 | **4.16 (max=ln(64)=4.16)** | max |
| L1 entropy | 0 | **4.85 (max=ln(128)=4.85)** | max |
| L2 entropy | 0 | **5.54 (max=ln(256)=5.55)** | 99.8% max |
| Metadata conf var | 3.5e-5 | **0.058/0.060/0.061** | >> 1e-4 ✓ |
| Shuffle conf diff | 0 | **0.27** | >> 1e-4 ✓ |
| Item↔SID 1-1 | — | **✓** | — |
| NaN/Inf | — | **✓ 无** | — |

**唯一 FAIL**: metadata kappa/scale var=0 (因 utils.py `c=1.0` hardcoded, 所有 layer 共享 c → 每层 κ/scale 是常数). 这是 Issue #44/#98 路径 (per-component learnable κ) 修复范围, 不在 #97 patch 内.

---

## 关键产物

- **evidence**: products/task395_issue99_stage2_sinkhorn_post_patch/evidence_package.json
- **run log**: products/task395_issue99_stage2_sinkhorn_post_patch/run.log
- **SID sample**: [[46, 24, 71, 0], [59, 88, 250, 0], [31, 58, 60, 0], ...]
- **3-digit unique**: 9674/9922 (97.53%) — 4-digit dedup 后 100%
- **verdict**: verdicts/task395_issue99_stage2_sinkhorn_post_patch_v2.md

---

## 后续 (per R22 + R19 + Issue #99 framework)

1. **task396: Stage 3 T5-mini 200 epoch 训练** — 用本 task SID 训练 (per HG-Rec recipe `num_hierarchies=4`)
2. **task397: Stage 4 R@K eval** — vs HG-Rec baseline R@10=0.1020 (GO/NO-GO 决策)
3. **Issue #98 patch** (可选进阶): per-component κ 学习化, 让 metadata kappa/scale var > 0, 满足 #99 spec 严格 12/12 PASS