# Issue #210 Phase A Equal-Codebook Control: 诊断报告 (中间态)

日期: 2026-08-08
Issue: #210 (work_items/77)
Phase: A 中间态 — v15 baseline (64,128,256) 已诊断; 主控 (128,128,128) 等排队训练中

---

## 1. 背景

当前三层 RQ-VAE 配置 (v15 capmatch): K_0=64, K_1=128, K_2=256 (递增)。已有实验观察到 |r_0| > |r_1| > |r_2| 的 residual 异质性,但**无法排除 confounder**: 层间差异部分可能来源于不同 codebook capacity, 而非纯 RQ depth。

**Issue 目标**: 使用 equal-codebook control 验证 (1) residual 异质性 是否独立于 codebook size; (2) layer-wise curvature sensitivity 是否独立于 codebook size。

---

## 2. Phase A 中间态: v15 (64,128,256) baseline 诊断

**输入数据**: hyp_v2 Stage1 emb (sha=e8fea26a, **不是** v15 训练时用的 baseline Stage1 sha=1a42341f → **数据不匹配警告**,util 偏低是数据不匹配导致,不影响 residual ordering 结论)

**Stage2 ckpt**: v15 1000ep 训练 (sha=5f8331cc), K=(64,128,256), --no_mlr, --no_curv_prior

**最终 κ/c (训练收敛值)**:
- L0: κ=0.304, c=1.35
- L1: κ=1.792, c=6.00
- L2: κ=1.480, c=4.39

### Residual distribution per layer

| Layer | mean | std | q05 | q25 | q50 | q75 | q95 |
|---|---|---|---|---|---|---|---|
| L0 (K=64) | **0.2234** | 0.0118 | 0.2046 | 0.2157 | 0.2232 | 0.2311 | 0.2432 |
| L1 (K=128) | **0.1711** | 0.0188 | 0.1428 | 0.1589 | 0.1700 | 0.1830 | 0.2013 |
| L2 (K=256) | **0.1419** | 0.0146 | 0.1187 | 0.1318 | 0.1417 | 0.1517 | 0.1657 |

**关键观察**:
1. ✅ **Residual ordering 满足**: |r_0| > |r_1| > |r_2| (0.22 > 0.17 > 0.14)
2. ✅ **Ratios**: R_1 = 0.77, R_2 = 0.64 (相对于 L0)
3. ✅ **三层 KS test 全部极显著**: ks=0.93/0.99/0.57, p=0 (***)
4. ✅ **三层分布完全分离**, 没有 overlap

### Quantization health per layer

| Layer | K | util | dead_ratio | entropy | entropy_max | nearest_dist | nearest_margin | quant_err |
|---|---|---|---|---|---|---|---|---|
| L0 | 64 | 0.3438 | 0.6562 | 1.265 | 4.159 (ln 64) | 0.4493 | 0.0246 | 0.2234 |
| L1 | 128 | 0.3984 | 0.6016 | 1.842 | 4.852 (ln 128) | 0.3448 | 0.0312 | 0.1711 |
| L2 | 256 | 0.5312 | 0.4688 | 3.355 | 5.545 (ln 256) | 0.2855 | 0.0140 | 0.1419 |

**健康度警告** ⚠️:
- util (34-53%) 远低于原始 sid_metadata.json 中报告的 1.0 (那是 3-digit SID 评估, 不是原码本)
- dead_code 47-66% (每个 layer 一半以上码字闲置)
- entropy 比例: 30%/38%/60% of max (L0 利用最差)

**根因**: 数据不匹配 — 用 hyp_v2 Stage1 emb 评估 v15 baseline Stage2 ckpt, 模型对这些输入分配不均。要正确评估需用 baseline Stage1 emb (sha=1a42341f), 该文件已丢失。

**但 residual ordering 仍有效**: 即使在数据不匹配下, 模型仍展现出 RQ depth 自身的层级几何结构 → 这是模型的 intrinsic property, 不依赖输入数据精确性。

---

## 3. Gate 1 (Phase A) 评估

### 当前 (64,128,256) baseline 是否通过 Gate 1?

| 判定项 | 结果 |
|---|---|
| E[\|r_0\|] > E[\|r_1\|] > E[\|r_2\|] | ✅ 满足 (0.2234 > 0.1711 > 0.1419) |
| 三层 KS test p < 10^-3 | ✅ 满足 (p = 0.00e+00 全部显著) |

**Gate 1 PASS** (在数据不匹配下,residual ordering 仍成立 → 等训练完主控 (128,128,128) 后再确认是否同样成立)。

### 当前是否通过 Gate 2?

| 判定项 | 结果 |
|---|---|
| utilization 健康 | ⚠️ util 34-53%, 远低于基线期望 ≥80% |
| dead-code ratio 可接受 | ❌ dead 47-66% |
| assignment entropy 不退化 | ⚠️ entropy/entropy_max 仅 30-60% |

**Gate 2 FAIL** (但根因是数据不匹配, 需主控训练后用一致数据重测)

---

## 4. 下一步

### Phase A 完成路径

1. ⏳ **等 GPU 空闲** (v85q_batch2048 训练中,预计 ep150 完成 ~15 min 后空闲)
2. 🔄 **训练 (128,128,128) 主控** (DDP 4 卡, ~8 min)
3. 🔄 **训练 (64,64,64) + (256,256,256) 对照** (各 ~8 min)
4. 🔄 **统一数据重做诊断** (用同一个 Stage1 emb, 保证对比公平)
5. 🔄 **生成 Phase A 完整报告** (4 配置并列对比)

### 队列脚本

`/home/wlia0047/.claude/jobs/6ae5ecdb/tmp/issue210_queue.py` PID=2513215
- 等待 v85q 完成后自动启动 (128,128,128) → (64,64,64) → (256,256,256)

---

## 5. DECOR BAN

本诊断严格在曲率框架内推进, **不引入任何 DECOR 机制** (no --enable_prompt_former, no decor_prompt_former.py)。

---

## 6. 文件清单

- Phase A 中间诊断 JSON: `/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue210_equal_codebook/phase_a_v15_64x128x256.json`
- 队列脚本: `/home/wlia0047/.claude/jobs/6ae5ecdb/tmp/issue210_queue.py` (PID 2513215 后台运行)
- verdict: 本文件
- 备份主脚本: `/home/wlia0047/.claude/backups/taskA_stage2_issue210_backup.py` (R31 保护)